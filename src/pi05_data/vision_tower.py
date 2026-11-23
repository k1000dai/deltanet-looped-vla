"""Exact PyTorch `π0.5` (`PaliGemma` `So400m/14`) SigLIP vision tower & RGB preprocessing.

Matches `openpi/models/siglip.py` (`variant="So400m/14"`, `num_classes=2048`, `pool_type="none"`,
`dtype_mm="bfloat16"`) and `openpi/shared/image_tools.py:resize_with_pad_torch` line-for-line:
  1. `uint8 [0, 255]` -> `float32 / 255.0 * 2.0 - 1.0` in `[-1, 1]` -> `resize_with_pad_torch(224, 224)`.
  2. `patch_embed` (`14x14` stride `14` conv) + learned `pos_embed (1, 256, 1152)` in `float32` ("Kevin edit").
  3. Cast to `bfloat16` through 27 `Encoder1DBlock` layers (`LayerNorm(eps=1e-6)`, 16-head MHA `head_dim=72`,
     `MlpBlock` `1152 -> 4304 -> 1152` with `gelu(approximate="tanh")`) + `encoder_norm(eps=1e-6)`.
  4. Linear multimodal projection `head: Linear(1152, 2048)` -> `(B, 256, 2048)` (`16x16` spatial grid).
"""

from __future__ import annotations

import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DEFAULT_MIYABI_SIGLIP_PKL = "/work/gj26/b20098/checkpoints/pi05_init/pi05_vision_encoder/siglip_params.pkl"
DEFAULT_MIYABI_PI05_BASE_SAFETENSORS = (
    "/work/gj26/b20098/hub/models--lerobot--pi05_base/snapshots/"
    "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba/model.safetensors"
)


def resize_with_pad_nchw(
    images: torch.Tensor,
    height: int = 224,
    width: int = 224,
) -> torch.Tensor:
    """Exact NCHW float32 `[-1, 1]` equivalent of `openpi.shared.image_tools.resize_with_pad_torch`."""
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW tensor of rank 4, got shape {tuple(images.shape)}")
    _, _, cur_h, cur_w = images.shape
    if cur_h == height and cur_w == width:
        return images.clamp(-1.0, 1.0)

    ratio = max(cur_w / float(width), cur_h / float(height))
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)

    resized = F.interpolate(images, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    resized = resized.clamp(-1.0, 1.0)

    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w
    if pad_h0 == 0 and pad_h1 == 0 and pad_w0 == 0 and pad_w1 == 0:
        return resized
    return F.pad(resized, (pad_w0, pad_w1, pad_h0, pad_h1), mode="constant", value=-1.0)


def preprocess_pi05_rgb(
    rgb: np.ndarray | torch.Tensor,
    *,
    image_size: tuple[int, int] = (224, 224),
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert `uint8 [0, 255]` (or float `[-1, 1]`) RGB images to `(N, 3, 224, 224)` `float32` in `[-1, 1]`.

    Accepts `(H, W, 3)`, `(N, H, W, 3)`, `(3, H, W)`, or `(N, 3, H, W)`.
    """
    t = torch.from_numpy(rgb) if isinstance(rgb, np.ndarray) else rgb
    if device is not None:
        t = t.to(device=device, non_blocking=True)

    if t.ndim == 3:
        t = t.unsqueeze(0)
    if t.ndim != 4:
        raise ValueError(f"Expected 3D or 4D RGB input, got shape {tuple(t.shape)}")

    if t.shape[-1] == 3 and t.shape[1] != 3:
        t = t.permute(0, 3, 1, 2).contiguous()
    elif t.shape[1] != 3:
        raise ValueError(f"Expected channel dim of size 3, got shape {tuple(t.shape)}")

    if t.dtype == torch.uint8:
        x = t.to(dtype=torch.float32) * (2.0 / 255.0) - 1.0
    else:
        x = t.to(dtype=torch.float32)
        # If passed in [0, 1] float, caller should pre-scale; check standard [-1, 1] bounds
        if float(x.min().item()) >= 0.0 and float(x.max().item()) > 1.0:
            x = x * (2.0 / 255.0) - 1.0

    return resize_with_pad_nchw(x, height=int(image_size[0]), width=int(image_size[1]))


def pool_siglip_tokens(
    tokens: torch.Tensor,
    target_grid: int = 8,
    *,
    pool_type: str = "mean",
) -> torch.Tensor:
    """Spatially pool `(..., 256, D)` `16x16` SigLIP patch tokens to `(..., target_grid^2, D)`.

    Matches `mme_vla_suite.shared.data_utils.pool_tokens_to_size` (`nnx.avg_pool` with non-overlapping windows).
    """
    orig_shape = tokens.shape
    num_patches = orig_shape[-2]
    dim = orig_shape[-1]
    target_patches = int(target_grid) * int(target_grid)
    if num_patches == target_patches:
        return tokens

    src_side = int(math.isqrt(num_patches))
    if src_side * src_side != num_patches:
        raise ValueError(f"Expected square patch count, got {num_patches}")
    if src_side % target_grid != 0:
        raise ValueError(f"src_side={src_side} must be divisible by target_grid={target_grid}")

    k = src_side // int(target_grid)
    flat = tokens.reshape(-1, src_side, src_side, dim).permute(0, 3, 1, 2)  # (M, D, H, W)
    orig_dtype = tokens.dtype
    if pool_type == "mean":
        pooled = F.avg_pool2d(flat.float(), kernel_size=k, stride=k).to(orig_dtype)
    elif pool_type == "max":
        pooled = F.max_pool2d(flat, kernel_size=k, stride=k)
    else:
        raise ValueError(f"Unsupported pool_type={pool_type!r}")
    out = pooled.permute(0, 2, 3, 1).reshape(*orig_shape[:-2], target_patches, dim)
    return out.contiguous()


def project_siglip_1152_to_2048(
    tokens_1152: torch.Tensor,
    head_weight: torch.Tensor,
    head_bias: torch.Tensor,
) -> torch.Tensor:
    """Project `1152-D` `encoder_norm` SigLIP tokens to `2048-D` `PaliGemma.img` tokens (`0.0` bit-diff!)."""
    w = head_weight.detach().to(device=tokens_1152.device, dtype=tokens_1152.dtype)
    b = head_bias.detach().to(device=tokens_1152.device, dtype=tokens_1152.dtype)
    if not tokens_1152.requires_grad:
        with torch.no_grad():
            return F.linear(tokens_1152.clone() if tokens_1152.is_inference() else tokens_1152, w, b)
    return F.linear(tokens_1152, w, b)


class _FlaxCompatibleLayerNorm(nn.Module):
    """Matches `src/pi05/siglip.py:_ln` & `flax.linen.LayerNorm(epsilon=1e-6, dtype=dtype_mm)` bit-for-bit."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var_fp32 = F.layer_norm(x.float(), (self.dim,), None, None, self.eps)
        return var_fp32.to(x.dtype) * self.weight.to(x.dtype) + self.bias.to(x.dtype)


class _SiglipSo400mBlock(nn.Module):
    def __init__(self, width: int = 1152, num_heads: int = 16, mlp_dim: int = 4304) -> None:
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.norm1 = _FlaxCompatibleLayerNorm(width, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width, bias=True)
        self.out_proj = nn.Linear(width, width, bias=True)
        self.norm2 = _FlaxCompatibleLayerNorm(width, eps=1e-6)
        self.fc1 = nn.Linear(width, mlp_dim, bias=True)
        self.fc2 = nn.Linear(mlp_dim, width, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, S, Dh)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.width)
        x = x + self.out_proj(attn)

        h2 = self.norm2(x)
        h2 = self.fc2(F.gelu(self.fc1(h2), approximate="tanh"))
        return x + h2


class Pi05SiglipVisionTower(nn.Module):
    """Standalone PyTorch `π0.5` SigLIP `So400m/14` (`1152-D` ViT + `2048-D` PaliGemma linear projector)."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 14,
        width: int = 1152,
        depth: int = 27,
        num_heads: int = 16,
        mlp_dim: int = 4304,
        proj_dim: int = 2048,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid_size = self.image_size // self.patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.width = int(width)
        self.depth = int(depth)
        self.proj_dim = int(proj_dim)
        self.compute_dtype = compute_dtype

        # Kept strictly in float32 per `openpi/models/siglip.py` lines 211-229 ("Kevin edit")
        self.patch_embed = nn.Conv2d(
            3, self.width, kernel_size=self.patch_size, stride=self.patch_size, padding=0, bias=True
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.width, dtype=torch.float32))

        self.blocks = nn.ModuleList(
            [_SiglipSo400mBlock(width=self.width, num_heads=num_heads, mlp_dim=mlp_dim) for _ in range(self.depth)]
        )
        self.encoder_norm = _FlaxCompatibleLayerNorm(self.width, eps=1e-6)
        self.head = nn.Linear(self.width, self.proj_dim, bias=True)

        self._cast_compute_weights(compute_dtype)

    def _cast_compute_weights(self, dtype: torch.dtype) -> None:
        self.compute_dtype = dtype
        self.patch_embed.float()
        self.pos_embed.data = self.pos_embed.data.float()
        for blk in self.blocks:
            blk.qkv.to(dtype=dtype)
            blk.out_proj.to(dtype=dtype)
            blk.fc1.to(dtype=dtype)
            blk.fc2.to(dtype=dtype)
            blk.norm1.float()
            blk.norm2.float()
        self.encoder_norm.float()
        self.head.to(dtype=dtype)

    @torch.no_grad()
    def load_from_siglip_pickle(self, pkl_path: str | Path = DEFAULT_MIYABI_SIGLIP_PKL) -> None:
        """Load exact `Yinpei/pi05_vision_encoder` (`siglip_params.pkl`) Flax weights."""
        with open(pkl_path, "rb") as f:
            sp: dict[str, Any] = pickle.load(f)

        def _t(arr: Any) -> torch.Tensor:
            # Match `Yinpei/pi05_baseline` (`79999/params`), where `PaliGemma/img/*` is stored in bfloat16
            t_fp32 = torch.from_numpy(np.asarray(arr, dtype=np.float32))
            if self.compute_dtype == torch.bfloat16:
                return t_fp32.to(torch.bfloat16).float()
            return t_fp32

        # Patch conv: Flax HWIO (14, 14, 3, 1152) -> PyTorch OIHW (1152, 3, 14, 14)
        self.patch_embed.weight.copy_(_t(sp["embedding"]["kernel"]).permute(3, 2, 0, 1))
        self.patch_embed.bias.copy_(_t(sp["embedding"]["bias"]))
        self.pos_embed.copy_(_t(sp["pos_embedding"]))

        enc = sp["Transformer"]["encoderblock"]
        for lyr in range(self.depth):
            blk: _SiglipSo400mBlock = self.blocks[lyr]  # type: ignore[assignment]
            blk.norm1.weight.copy_(_t(enc["LayerNorm_0"]["scale"][lyr]))
            blk.norm1.bias.copy_(_t(enc["LayerNorm_0"]["bias"][lyr]))

            attn = enc["MultiHeadDotProductAttention_0"]
            # Flax (1152, 16, 72) -> reshape(1152, 1152).T -> PyTorch Linear (1152, 1152)
            q_w = _t(attn["query"]["kernel"][lyr]).reshape(self.width, self.width).t()
            k_w = _t(attn["key"]["kernel"][lyr]).reshape(self.width, self.width).t()
            v_w = _t(attn["value"]["kernel"][lyr]).reshape(self.width, self.width).t()
            q_b = _t(attn["query"]["bias"][lyr]).reshape(self.width)
            k_b = _t(attn["key"]["bias"][lyr]).reshape(self.width)
            v_b = _t(attn["value"]["bias"][lyr]).reshape(self.width)
            blk.qkv.weight.copy_(torch.cat([q_w, k_w, v_w], dim=0).to(self.compute_dtype))
            blk.qkv.bias.copy_(torch.cat([q_b, k_b, v_b], dim=0).to(self.compute_dtype))

            # Flax out (16, 72, 1152) -> reshape(1152, 1152).T -> PyTorch Linear (1152, 1152)
            out_w = _t(attn["out"]["kernel"][lyr]).reshape(self.width, self.width).t()
            out_b = _t(attn["out"]["bias"][lyr])
            blk.out_proj.weight.copy_(out_w.to(self.compute_dtype))
            blk.out_proj.bias.copy_(out_b.to(self.compute_dtype))

            blk.norm2.weight.copy_(_t(enc["LayerNorm_1"]["scale"][lyr]))
            blk.norm2.bias.copy_(_t(enc["LayerNorm_1"]["bias"][lyr]))

            mlp = enc["MlpBlock_0"]
            blk.fc1.weight.copy_(_t(mlp["Dense_0"]["kernel"][lyr]).t().to(self.compute_dtype))
            blk.fc1.bias.copy_(_t(mlp["Dense_0"]["bias"][lyr]).to(self.compute_dtype))
            blk.fc2.weight.copy_(_t(mlp["Dense_1"]["kernel"][lyr]).t().to(self.compute_dtype))
            blk.fc2.bias.copy_(_t(mlp["Dense_1"]["bias"][lyr]).to(self.compute_dtype))

        enc_norm = sp["Transformer"]["encoder_norm"]
        self.encoder_norm.weight.copy_(_t(enc_norm["scale"]))
        self.encoder_norm.bias.copy_(_t(enc_norm["bias"]))

        self.head.weight.copy_(_t(sp["head"]["kernel"]).t().to(self.compute_dtype))
        self.head.bias.copy_(_t(sp["head"]["bias"]).to(self.compute_dtype))

    @torch.no_grad()
    def load_from_hf_safetensors(self, safetensors_path: str | Path = DEFAULT_MIYABI_PI05_BASE_SAFETENSORS) -> None:
        """Load vision_tower + multi_modal_projector directly from `lerobot/pi05_base` or converted `.safetensors`."""
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as sf:
            all_keys = set(sf.keys())

            def _find(suffix: str) -> torch.Tensor:
                for cand in (
                    f"paligemma_with_expert.paligemma.model.{suffix}",
                    f"model.paligemma_with_expert.paligemma.model.{suffix}",
                    f"paligemma.model.{suffix}",
                    suffix,
                ):
                    if cand in all_keys:
                        return sf.get_tensor(cand)
                matches = [k for k in all_keys if k.endswith(suffix)]
                if len(matches) == 1:
                    return sf.get_tensor(matches[0])
                raise KeyError(f"Could not find tensor ending with {suffix!r} in {safetensors_path}")

            self.patch_embed.weight.copy_(_find("vision_tower.vision_model.embeddings.patch_embedding.weight").float())
            self.patch_embed.bias.copy_(_find("vision_tower.vision_model.embeddings.patch_embedding.bias").float())
            pos_w = _find("vision_tower.vision_model.embeddings.position_embedding.weight").float()
            self.pos_embed.copy_(pos_w.unsqueeze(0) if pos_w.ndim == 2 else pos_w)

            for lyr in range(self.depth):
                pfx = f"vision_tower.vision_model.encoder.layers.{lyr}"
                blk: _SiglipSo400mBlock = self.blocks[lyr]  # type: ignore[assignment]
                blk.norm1.weight.copy_(_find(f"{pfx}.layer_norm1.weight").float())
                blk.norm1.bias.copy_(_find(f"{pfx}.layer_norm1.bias").float())
                q_w = _find(f"{pfx}.self_attn.q_proj.weight")
                k_w = _find(f"{pfx}.self_attn.k_proj.weight")
                v_w = _find(f"{pfx}.self_attn.v_proj.weight")
                q_b = _find(f"{pfx}.self_attn.q_proj.bias")
                k_b = _find(f"{pfx}.self_attn.k_proj.bias")
                v_b = _find(f"{pfx}.self_attn.v_proj.bias")
                blk.qkv.weight.copy_(torch.cat([q_w, k_w, v_w], dim=0).to(self.compute_dtype))
                blk.qkv.bias.copy_(torch.cat([q_b, k_b, v_b], dim=0).to(self.compute_dtype))
                blk.out_proj.weight.copy_(_find(f"{pfx}.self_attn.out_proj.weight").to(self.compute_dtype))
                blk.out_proj.bias.copy_(_find(f"{pfx}.self_attn.out_proj.bias").to(self.compute_dtype))
                blk.norm2.weight.copy_(_find(f"{pfx}.layer_norm2.weight").float())
                blk.norm2.bias.copy_(_find(f"{pfx}.layer_norm2.bias").float())
                blk.fc1.weight.copy_(_find(f"{pfx}.mlp.fc1.weight").to(self.compute_dtype))
                blk.fc1.bias.copy_(_find(f"{pfx}.mlp.fc1.bias").to(self.compute_dtype))
                blk.fc2.weight.copy_(_find(f"{pfx}.mlp.fc2.weight").to(self.compute_dtype))
                blk.fc2.bias.copy_(_find(f"{pfx}.mlp.fc2.bias").to(self.compute_dtype))

            self.encoder_norm.weight.copy_(_find("vision_tower.vision_model.post_layernorm.weight").float())
            self.encoder_norm.bias.copy_(_find("vision_tower.vision_model.post_layernorm.bias").float())
            self.head.weight.copy_(_find("multi_modal_projector.linear.weight").to(self.compute_dtype))
            self.head.bias.copy_(_find("multi_modal_projector.linear.bias").to(self.compute_dtype))

    def forward_features(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(encoded_1152, projected_2048)` of shapes `(B, 256, 1152)` and `(B, 256, 2048)`."""
        # Patch stem + position embedding in fp32
        x = self.patch_embed(pixel_values.float())  # (B, 1152, 16, 16)
        x = x.flatten(2).transpose(1, 2) + self.pos_embed.float()  # (B, 256, 1152)
        x = x.to(self.compute_dtype)
        for blk in self.blocks:
            x = blk(x)
        encoded_1152 = self.encoder_norm(x).to(self.compute_dtype)
        projected_2048 = self.head(encoded_1152)
        return encoded_1152, projected_2048

    def forward(self, pixel_values: torch.Tensor, *, return_pre_head: bool = False) -> torch.Tensor:
        encoded_1152, projected_2048 = self.forward_features(pixel_values)
        return encoded_1152 if return_pre_head else projected_2048
