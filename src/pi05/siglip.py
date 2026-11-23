"""Exact Flax-parity PyTorch port of OpenPI's `SigLIP-So400m/14` vision encoder (`openpi/models/siglip.py`).

Matches every detail of `openpi/models/siglip.py`:
- Patch extraction (`14x14` valid conv) + learned `pos_embedding` `(1, 256, 1152)` computed in `float32`,
  then cast to `dtype_mm` (`bfloat16` or `float32`).
- 27 Transformer blocks with `LayerNorm(eps=1e-6)` (Flax default epsilon, NOT PyTorch default `1e-5`),
  16-head MHA (`head_dim=72`), and MLP (`1152 -> 4304 -> 1152`) with `gelu(approximate="tanh")`.
- Post-encoder `LayerNorm(eps=1e-6)` + linear projection `head` (`1152 -> 2048`), returning unpooled
  `(B, 256, 2048)` patch embeddings.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import Pi05Config


class SigLIPAttention(nn.Module):
    """Multi-head self-attention matching `flax.linen.MultiHeadDotProductAttention`."""

    def __init__(self, width: int = 1152, num_heads: int = 16) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width, bias=True)
        self.v_proj = nn.Linear(width, width, bias=True)
        self.out_proj = nn.Linear(width, width, bias=True)

    def forward(self, x: Tensor, *, exact_fp32_attn: bool = False) -> Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if exact_fp32_attn or x.dtype == torch.float32:
            logits = torch.matmul(q.float() * self.scale, k.float().transpose(-2, -1))
            probs = F.softmax(logits, dim=-1).to(x.dtype)
            attn_out = torch.matmul(probs, v)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.width)
        return self.out_proj(attn_out)


class SigLIPMLP(nn.Module):
    """Feed-forward block matching `siglip.MlpBlock` (`nn.gelu` = tanh approximation)."""

    def __init__(self, width: int = 1152, mlp_dim: int = 4304) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, mlp_dim, bias=True)
        self.fc2 = nn.Linear(mlp_dim, width, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        return self.fc2(x)


class SigLIPEncoderLayer(nn.Module):
    """Single pre-norm ViT encoder block matching `siglip.Encoder1DBlock`."""

    def __init__(self, width: int = 1152, mlp_dim: int = 4304, num_heads: int = 16, eps: float = 1e-6) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(width, eps=eps)
        self.self_attn = SigLIPAttention(width=width, num_heads=num_heads)
        self.layer_norm2 = nn.LayerNorm(width, eps=eps)
        self.mlp = SigLIPMLP(width=width, mlp_dim=mlp_dim)

    def _ln(self, ln: nn.LayerNorm, x: Tensor) -> Tensor:
        var_fp32 = F.layer_norm(x.float(), (x.shape[-1],), None, None, ln.eps)
        out = var_fp32.to(x.dtype) * ln.weight.to(x.dtype) + ln.bias.to(x.dtype)
        return out

    def forward(self, x: Tensor, *, exact_fp32_attn: bool = False) -> Tensor:
        x = x + self.self_attn(self._ln(self.layer_norm1, x), exact_fp32_attn=exact_fp32_attn)
        x = x + self.mlp(self._ln(self.layer_norm2, x))
        return x


class SigLIPVisionTower(nn.Module):
    """`SigLIP-So400m/14` (`224x224 -> 256 x 2048`) matching `PaliGemma.img` in `HistoryPi0`."""

    def __init__(self, config: Pi05Config) -> None:
        super().__init__()
        grid = config.image_size // config.siglip_patch_size
        self.num_patches = grid * grid
        self.width = config.siglip_width
        self.out_dim = config.paligemma_width

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=config.siglip_width,
            kernel_size=config.siglip_patch_size,
            stride=config.siglip_patch_size,
            padding=0,
            bias=True,
        )
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, config.siglip_width))

        self.layers = nn.ModuleList(
            [
                SigLIPEncoderLayer(
                    width=config.siglip_width,
                    mlp_dim=config.siglip_mlp_dim,
                    num_heads=config.siglip_num_heads,
                    eps=config.layernorm_eps,
                )
                for _ in range(config.siglip_depth)
            ]
        )
        self.post_layernorm = nn.LayerNorm(config.siglip_width, eps=config.layernorm_eps)
        self.head = nn.Linear(config.siglip_width, config.paligemma_width, bias=True)
        self.gradient_checkpointing: bool = False

    def forward(self, image: Tensor, *, exact_fp32_attn: bool = False) -> Tensor:
        """Encode `image` in `[-1, 1]` (`[B, H, W, 3]` or `[B, 3, H, W]`) into `[B, 256, 2048]`."""
        if image.ndim != 4:
            raise ValueError(f"Expected 4D image batch, got shape {tuple(image.shape)}")
        if image.shape[-1] == 3 and image.shape[1] != 3:
            img_nchw = image.permute(0, 3, 1, 2).contiguous()
        elif image.shape[1] == 3:
            img_nchw = image
        else:
            raise ValueError(f"Expected 3 channels on axis 1 or -1, got shape {tuple(image.shape)}")

        # Patch extraction + pos_embedding in float32 (matches `openpi/models/siglip.py` L213-229)
        img_fp32 = img_nchw.float()
        x = F.conv2d(
            img_fp32,
            self.patch_embedding.weight.float(),
            self.patch_embedding.bias.float(),
            stride=self.patch_embedding.stride,
            padding=0,
        )
        # NCHW (B, C, 16, 16) -> row-major (B, 256, C) matching NHWC reshape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding.float()

        dtype_mm = self.layers[0].self_attn.q_proj.weight.dtype
        x = x.to(dtype_mm)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, use_reentrant=False, preserve_rng_state=False
                )
            else:
                x = layer(x, exact_fp32_attn=exact_fp32_attn)

        x_norm = F.layer_norm(x.float(), (x.shape[-1],), None, None, self.post_layernorm.eps)
        x = x_norm.to(dtype_mm) * self.post_layernorm.weight.to(dtype_mm) + self.post_layernorm.bias.to(dtype_mm)
        return self.head(x)


def pool_vision_tokens(tokens: Tensor, target_tokens: int = 64, pool_type: str = "mean") -> Tensor:
    """Pool `(..., 256, D)` SigLIP patch tokens to `(..., target_tokens, D)` (`64=8x8`, `16=4x4`, `4=2x2`).

    Matches `mme_vla_suite.shared.data_utils.pool_tokens_to_size` identically.
    """
    p = tokens.shape[-2]
    d = tokens.shape[-1]
    if p == target_tokens:
        return tokens
    h = int(math.isqrt(p))
    target_h = int(math.isqrt(target_tokens))
    if h * h != p or target_h * target_h != target_tokens or h % target_h != 0:
        raise ValueError(f"Cannot pool {p} tokens into {target_tokens} square grid tokens")
    k = h // target_h

    lead_shape = tokens.shape[:-2]
    x = tokens.reshape(-1, h, h, d).permute(0, 3, 1, 2)  # (N, D, H, W)
    orig_dtype = x.dtype
    if pool_type == "mean":
        pooled = F.avg_pool2d(x.float(), kernel_size=k, stride=k).to(orig_dtype)
    elif pool_type == "max":
        pooled = F.max_pool2d(x.float(), kernel_size=k, stride=k).to(orig_dtype)
    else:
        raise ValueError(f"Unsupported pool_type={pool_type!r}")
    return pooled.permute(0, 2, 3, 1).reshape(*lead_shape, target_tokens, d)
