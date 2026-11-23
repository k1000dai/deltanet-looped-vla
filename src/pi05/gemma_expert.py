"""PaliGemma-2B + Gemma-300M Action Expert with 18-Layer `HistoryBlock` Memory Hooks & LoRA.

Exact PyTorch port of:
- `openpi/models/gemma.py` (`RMSNorm`, `Attention`, `FeedForward`, `Block`, `_apply_rope`)
- `mme_vla_suite/models/integration/history_gemma.py` (`MemoryRMSNorm`, `MemoryAttention`, `HistoryBlock`)

Supports both:
1. `mem_mod_mode="modul_residual"` (default for `DeltaMem-π0.5` warm-start from `Yinpei/pi05_baseline`):
   - Zero-initialized residual FiLM modulation (`x + norm(x)*scale + shift`) over 18-layer `MemoryAttention`,
     guaranteeing **bit-exact `0.0` forward diff** vs `pi05_baseline` at step 0 and when `mem_tokens=None`,
     safe `0.0` handling when `mem_mask` has all-`False` rows, and non-zero step-0 gradient flow.
2. `mem_mod_mode="mme_vla_exact"`:
   - Exact upstream `HistoryBlock` (`MemoryRMSNorm(x, mem_attn(x, mem_seq, mem_mask))` before `pre_ffw_norm`)
     for bit-exact weight loading and closed-loop parity on `Yinpei/perceptual-framesamp-modul` (`44.51%` SR).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import MemModMode, Pi05Config

OPENPI_MASK_NEG: float = -2.3819763e38


def apply_rope(x: Tensor, positions: Tensor, max_wavelength: float = 10_000.0) -> Tensor:
    """Apply Gemma RoPE (`positions: [B, L]`) to `x: [B, L, H, D]`. Matches `openpi/models/gemma.py` L424."""
    d = x.shape[-1]
    freq_exponents = (2.0 / d) * torch.arange(d // 2, dtype=torch.float32, device=x.device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].float() / timescale[None, None, :]
    radians = radians[..., None, :]  # [B, L, 1, D/2]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    res = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return res.to(x.dtype)


def gated_residual(x: Tensor | None, y: Tensor | None, gate: Tensor | None) -> Tensor | None:
    if x is None:
        return None
    assert y is not None
    if gate is None:
        return x + y
    return x + y * gate


class LoRALinear(nn.Module):
    """Lightweight built-in LoRA wrapper (`y = base(x) + (x @ A @ B) * scaling`) with step-0 = `0.0` diff."""

    def __init__(self, base: nn.Linear, rank: int = 32, alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / float(rank)
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(in_features, rank, dtype=torch.float32, device=device))
        self.lora_b = nn.Parameter(torch.zeros(rank, out_features, dtype=torch.float32, device=device))
        nn.init.normal_(self.lora_a, mean=0.0, std=1.0 / math.sqrt(in_features))
        # Cast LoRA weights to match compute precision if needed, or keep fp32 master
        if dtype == torch.bfloat16:
            self.lora_a.data = self.lora_a.data.to(torch.bfloat16)
            self.lora_b.data = self.lora_b.data.to(torch.bfloat16)

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        lx = x.to(self.lora_a.dtype)
        delta = torch.matmul(torch.matmul(lx, self.lora_a), self.lora_b) * self.scaling
        return out + delta.to(out.dtype)


class Pi05RMSNorm(nn.Module):
    """Standard `(1 + weight)` RMSNorm (Stream 0) or Adaptive `(scale, shift, gate)` RMSNorm (Stream 1)."""

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: int | None = None) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.cond_dim = cond_dim
        if cond_dim is None:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.dense = None
        else:
            self.register_parameter("weight", None)
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)

    def _norm_fp32(self, x: Tensor) -> Tensor:
        xf = x.float()
        var = torch.mean(xf * xf, dim=-1, keepdim=True)
        return xf * torch.rsqrt(var + self.eps)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        dtype = x.dtype
        normed = self._norm_fp32(x)
        if self.dense is None or cond is None:
            assert self.weight is not None
            out = normed * (1.0 + self.weight.float())
            return out.to(dtype), None

        w_dtype = x.dtype if self.dense.weight.dtype == torch.bfloat16 else torch.float32
        modulation = F.linear(
            cond.to(w_dtype),
            self.dense.weight.to(w_dtype),
            self.dense.bias.to(w_dtype) if self.dense.bias is not None else None,
        )
        if x.ndim == 3 and modulation.ndim == 2:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        out = normed * (1.0 + scale.float()) + shift.float()
        return out.to(dtype), gate.to(dtype)


class MemoryRMSNorm(nn.Module):
    """Memory RMSNorm supporting both exact MME-VLA FiLM and bit-exact `0.0`-diff residual FiLM."""

    def __init__(
        self,
        dim: int = 1024,
        eps: float = 1e-6,
        *,
        adaptive: bool = False,
        residual_zero_init: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.adaptive = adaptive
        self.residual_zero_init = residual_zero_init
        if not adaptive:
            self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
            self.dense = None
        else:
            self.register_parameter("weight", None)
            self.dense = nn.Linear(dim, dim * 2, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)

    def _norm_fp32(self, x: Tensor) -> Tensor:
        xf = x.float()
        var = torch.mean(xf * xf, dim=-1, keepdim=True)
        return xf * torch.rsqrt(var + self.eps)

    def forward(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        dtype = x.dtype
        if not self.adaptive:
            assert self.weight is not None
            return (self._norm_fp32(x) * (1.0 + self.weight.float())).to(dtype)

        if cond is None:
            if self.residual_zero_init:
                return x
            return self._norm_fp32(x).to(dtype)

        assert self.dense is not None
        w_dtype = dtype if self.dense.weight.dtype == torch.bfloat16 else torch.float32
        modulation = F.linear(
            cond.to(w_dtype),
            self.dense.weight.to(w_dtype),
            self.dense.bias.to(w_dtype) if self.dense.bias is not None else None,
        )
        scale, shift = modulation.chunk(2, dim=-1)
        xf = x.float()
        rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        if self.residual_zero_init:
            # Scale-matched residual FiLM: x + (x * scale + rms * shift) == rms * (norm(x)*(1+scale) + shift).
            # Gives bit-exact 0.0 diff at step 0 (scale=0, shift=0) AND exact O(1) MME-VLA Jacobian into pre_ffw_norm!
            delta = (xf * scale.float() + rms * shift.float()).to(dtype)
            return x + delta
        normed = xf / rms
        return (normed * (1.0 + scale.float()) + shift.float()).to(dtype)


class MemoryAttention(nn.Module):
    """Cross-attention from ActionExpert (`[B, T_act, 1024]`) to memory (`[B, S_mem, 1024]`).

    Matches `mme_vla_suite/models/integration/history_gemma.py` (`MemoryAttention`, L45-109):
    - Shared `mem_rms_norm` applied to both `x` and `mem_tokens`.
    - GQA (`4` query heads, `1` KV head, `head_dim = 256`, width `1024`).
    - RoPE positions `0 .. S_mem-1` on keys and `S_mem .. S_mem+T_act-1` on queries.
    """

    def __init__(
        self,
        width: int = 1024,
        num_heads: int = 4,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        eps: float = 1e-6,
        rope_max_wavelength: float = 10_000.0,
        q_rope_offset: int | None = None,
        k_rope_span_frames: bool = False,
    ) -> None:
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rope_max_wavelength = rope_max_wavelength
        self.q_rope_offset = q_rope_offset
        self.k_rope_span_frames = k_rope_span_frames

        self.mem_rms_norm = MemoryRMSNorm(width, eps=eps, adaptive=False)
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, width, bias=False)
        self.reset_lecun_parameters()

    def reset_lecun_parameters(self) -> None:
        std_in = 1.0 / math.sqrt(self.width)
        std_out = 1.0 / math.sqrt(self.num_heads * self.head_dim)
        nn.init.trunc_normal_(self.q_proj.weight, std=std_in, a=-2 * std_in, b=2 * std_in)
        nn.init.trunc_normal_(self.k_proj.weight, std=std_in, a=-2 * std_in, b=2 * std_in)
        nn.init.trunc_normal_(self.v_proj.weight, std=std_in, a=-2 * std_in, b=2 * std_in)
        nn.init.trunc_normal_(self.out_proj.weight, std=std_out, a=-2 * std_out, b=2 * std_out)
        nn.init.zeros_(self.mem_rms_norm.weight)

    def precompute_kv(
        self,
        mem_tokens: Tensor,
        mem_mask: Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        """Precompute `(k_roped, v, mem_mask_bool, q_off)` once per chunk across all 10 Euler ODE steps."""
        bsz, mem_len, _ = mem_tokens.shape
        dev = mem_tokens.device
        target_dtype = dtype if dtype is not None else mem_tokens.dtype

        if mem_mask is None:
            mem_mask_bool = torch.ones((bsz, mem_len), dtype=torch.bool, device=dev)
        else:
            mem_mask_bool = mem_mask.to(device=dev, dtype=torch.bool)

        mem_norm = self.mem_rms_norm(mem_tokens.to(target_dtype))
        k = self.k_proj(mem_norm).view(bsz, mem_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(mem_norm).view(bsz, mem_len, self.num_kv_heads, self.head_dim)

        q_off = mem_len if self.q_rope_offset is None else int(self.q_rope_offset)
        if self.k_rope_span_frames and q_off > mem_len and (mem_len % 16 == 0) and (q_off % 16 == 0):
            n_kf = mem_len // 16
            max_kf = q_off // 16
            contig_slots = torch.arange(n_kf, device=dev, dtype=torch.long)
            strided_slots = (contig_slots * (max_kf - 1)) // max(1, n_kf - 1)
            is_full = mem_mask_bool.view(bsz, n_kf, 16)[:, :, 0].sum(dim=-1) >= n_kf
            f_slots = torch.where(is_full.unsqueeze(-1), strided_slots.unsqueeze(0), contig_slots.unsqueeze(0))
            patch_off = torch.arange(16, device=dev, dtype=torch.long).view(1, 1, 16)
            k_positions = (f_slots.unsqueeze(-1) * 16 + patch_off).reshape(bsz, mem_len)
        else:
            k_positions = torch.arange(0, mem_len, device=dev, dtype=torch.long).unsqueeze(0).expand(bsz, mem_len)

        k = apply_rope(k, k_positions, self.rope_max_wavelength)
        return k, v, mem_mask_bool, q_off

    def forward(
        self,
        x: Tensor,
        mem_tokens: Tensor | None,
        mem_mask: Tensor | None = None,
        *,
        safe_empty_mask: bool = True,
        cached_kv: tuple[Tensor, Tensor, Tensor, int] | None = None,
    ) -> Tensor:
        if cached_kv is None and (mem_tokens is None or mem_tokens.shape[1] == 0):
            return torch.zeros_like(x)

        bsz, x_len, _ = x.shape
        dtype = x.dtype

        if cached_kv is not None:
            k, v, mem_mask_bool, q_off = cached_kv
            if k.dtype != dtype:
                k = k.to(dtype)
                v = v.to(dtype)
        else:
            assert mem_tokens is not None
            k, v, mem_mask_bool, q_off = self.precompute_kv(mem_tokens, mem_mask, dtype=dtype)

        x_norm = self.mem_rms_norm(x)
        q = self.q_proj(x_norm).view(bsz, x_len, self.num_heads, self.head_dim)
        q_positions = torch.arange(q_off, x_len + q_off, device=x.device, dtype=torch.long).unsqueeze(0).expand(
            bsz, x_len
        )
        q = apply_rope(q, q_positions, self.rope_max_wavelength)
        q = q * (self.head_dim**-0.5)

        groups = self.num_heads // self.num_kv_heads
        q_g = q.view(bsz, x_len, self.num_kv_heads, groups, self.head_dim)
        logits = torch.einsum("btkgh,bskh->bkgts", q_g.float(), k.float())
        masked_logits = torch.where(mem_mask_bool[:, None, None, None, :], logits, OPENPI_MASK_NEG)
        probs = F.softmax(masked_logits, dim=-1).to(dtype)

        if safe_empty_mask:
            has_valid = mem_mask_bool.any(dim=-1).to(dtype)
            probs = probs * has_valid[:, None, None, None, None]

        encoded = torch.einsum("bkgts,bskh->btkgh", probs, v).reshape(bsz, x_len, self.num_heads * self.head_dim)
        out = self.out_proj(encoded)
        if safe_empty_mask:
            out = out * has_valid[:, None, None]
        return out


class GemmaGeGLUMLP(nn.Module):
    """GeGLU MLP matching `openpi/models/gemma.py` (`FeedForward`, `nn.gelu` tanh approx)."""

    def __init__(self, width: int, mlp_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaExpertStreamLayer(nn.Module):
    """Per-stream weights for either PaliGemma-2B (Stream 0) or ActionExpert-300M (Stream 1)."""

    def __init__(
        self,
        width: int,
        mlp_dim: int,
        num_heads: int = 8,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        eps: float = 1e-6,
        adarms_cond_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.input_layernorm = Pi05RMSNorm(width, eps=eps, cond_dim=adarms_cond_dim)
        self.q_proj = nn.Linear(width, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(width, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, width, bias=False)
        self.post_attention_layernorm = Pi05RMSNorm(width, eps=eps, cond_dim=adarms_cond_dim)
        self.mlp = GemmaGeGLUMLP(width, mlp_dim)


@dataclass
class PrefixKVCache:
    """Stacked 18-layer VLM prefix KV cache for fast Stage-A training & 10-step ODE sampling."""

    keys: Tensor  # [depth, B, L_prefix, num_kv_heads, head_dim]
    values: Tensor  # [depth, B, L_prefix, num_kv_heads, head_dim]
    pad_masks: Tensor  # bool [B, L_prefix]
    prefix_offsets: Tensor  # long [B, 1]

    def detach(self) -> PrefixKVCache:
        return PrefixKVCache(
            keys=self.keys.detach(),
            values=self.values.detach(),
            pad_masks=self.pad_masks.detach(),
            prefix_offsets=self.prefix_offsets.detach(),
        )

    def repeat_interleave(self, repeats: int) -> PrefixKVCache:
        return PrefixKVCache(
            keys=self.keys.repeat_interleave(repeats, dim=1),
            values=self.values.repeat_interleave(repeats, dim=1),
            pad_masks=self.pad_masks.repeat_interleave(repeats, dim=0),
            prefix_offsets=self.prefix_offsets.repeat_interleave(repeats, dim=0),
        )


class JointHistoryBlock(nn.Module):
    """Single joint PaliGemma + ActionExpert layer with optional `HistoryBlock` memory modulation."""

    def __init__(self, config: Pi05Config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.rope_max_wavelength = config.rope_max_wavelength
        self.mem_mod_mode: MemModMode = config.mem_mod_mode

        self.vlm = GemmaExpertStreamLayer(
            width=config.paligemma_width,
            mlp_dim=config.paligemma_mlp_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            eps=config.rms_norm_eps,
            adarms_cond_dim=None,
        )
        self.expert = GemmaExpertStreamLayer(
            width=config.expert_width,
            mlp_dim=config.expert_mlp_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            eps=config.rms_norm_eps,
            adarms_cond_dim=config.expert_width,
        )

        self.mem_attn: MemoryAttention | None = None
        self.mem_rms_norm_ffn: MemoryRMSNorm | None = None
        if config.mem_mod_mode != "none":
            self.init_memory_hook(config.mem_mod_mode, config)

    def init_memory_hook(self, mode: MemModMode, config: Pi05Config) -> None:
        self.mem_mod_mode = mode
        if mode == "none":
            self.mem_attn = None
            self.mem_rms_norm_ffn = None
            return
        if self.mem_attn is None:
            self.mem_attn = MemoryAttention(
                width=config.expert_width,
                num_heads=config.mem_num_heads,
                num_kv_heads=config.mem_num_kv_heads,
                head_dim=config.mem_head_dim,
                eps=config.rms_norm_eps,
                rope_max_wavelength=config.rope_max_wavelength,
            )
        residual_zero_init = mode == "modul_residual"
        if self.mem_rms_norm_ffn is None or self.mem_rms_norm_ffn.residual_zero_init != residual_zero_init:
            self.mem_rms_norm_ffn = MemoryRMSNorm(
                dim=config.expert_width,
                eps=config.rms_norm_eps,
                adaptive=True,
                residual_zero_init=residual_zero_init,
            )

    def _attend(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask_bool: Tensor,
        *,
        exact_fp32_attn: bool = False,
    ) -> Tensor:
        """Compute GQA attention (`q: [B, T, N, H]`, `k,v: [B, S, K, H]`, `attn_mask_bool: [B, T, S]`)."""
        bsz, t_len, _, h_dim = q.shape
        s_len = k.shape[1]
        dtype = q.dtype

        if exact_fp32_attn or dtype == torch.float32 or t_len <= 32:
            groups = self.num_heads // self.num_kv_heads
            q_g = q.view(bsz, t_len, self.num_kv_heads, groups, h_dim)
            logits = torch.einsum("btkgh,bskh->bkgts", q_g.float(), k.float())
            masked = torch.where(attn_mask_bool[:, None, None, :, :], logits, OPENPI_MASK_NEG)
            probs = F.softmax(masked, dim=-1).to(dtype)
            encoded = torch.einsum("bkgts,bskh->btkgh", probs, v)
            return encoded.reshape(bsz, t_len, self.num_heads * h_dim)

        # Fast Flash/Memory-Efficient SDPA path for long prefix prefill (`T ~ 576`)
        q_t = q.transpose(1, 2)  # [B, N, T, H]
        k_t = k.transpose(1, 2).expand(bsz, self.num_heads, s_len, h_dim)
        v_t = v.transpose(1, 2).expand(bsz, self.num_heads, s_len, h_dim)
        mask_4d = torch.where(attn_mask_bool[:, None, :, :], 0.0, OPENPI_MASK_NEG).to(dtype)
        # `q` already scaled by `head_dim ** -0.5` after RoPE (matching Flax `gemma.py` L204)
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask_4d, scale=1.0)
        return out.transpose(1, 2).contiguous().view(bsz, t_len, self.num_heads * h_dim)

    def _apply_expert_memory_mod(
        self,
        x_expert: Tensor,
        mem_tokens: Tensor | None,
        mem_mask: Tensor | None,
        mem_kv_layer: tuple[Tensor, Tensor, Tensor, int] | None = None,
    ) -> Tensor:
        if self.mem_mod_mode == "none" or self.mem_attn is None or self.mem_rms_norm_ffn is None:
            return x_expert
        if self.mem_mod_mode == "modul_residual":
            if mem_tokens is None and mem_kv_layer is None:
                return x_expert
            mem_mod_vec = self.mem_attn(
                x_expert, mem_tokens, mem_mask, safe_empty_mask=True, cached_kv=mem_kv_layer
            )
            out = self.mem_rms_norm_ffn(x_expert, mem_mod_vec)
            mask_ref = mem_kv_layer[2] if mem_kv_layer is not None else mem_mask
            if mask_ref is not None:
                has_valid = mask_ref.to(device=x_expert.device, dtype=torch.bool).any(dim=-1)
                out = torch.where(has_valid[:, None, None], out, x_expert)
            return out
        # "mme_vla_exact": exact upstream `HistoryBlock` behavior
        mem_mod_vec = self.mem_attn(
            x_expert, mem_tokens, mem_mask, safe_empty_mask=False, cached_kv=mem_kv_layer
        )
        return self.mem_rms_norm_ffn(x_expert, mem_mod_vec)

    def forward(
        self,
        vlm_x: Tensor | None,
        expert_x: Tensor | None,
        positions: Tensor,
        attn_mask_bool: Tensor,
        adarms_cond: Tensor | None = None,
        *,
        kv_cache_layer: tuple[Tensor, Tensor] | None = None,
        mem_tokens: Tensor | None = None,
        mem_mask: Tensor | None = None,
        mem_kv_layer: tuple[Tensor, Tensor, Tensor, int] | None = None,
        exact_fp32_attn: bool = False,
    ) -> tuple[Tensor | None, Tensor | None, tuple[Tensor, Tensor]]:
        qs: list[Tensor] = []
        ks: list[Tensor] = []
        vs: list[Tensor] = []
        vlm_gate: Tensor | None = None
        expert_gate: Tensor | None = None

        if vlm_x is not None:
            bsz, t0, _ = vlm_x.shape
            h0, vlm_gate = self.vlm.input_layernorm(vlm_x, None)
            qs.append(self.vlm.q_proj(h0).view(bsz, t0, self.num_heads, self.head_dim))
            ks.append(self.vlm.k_proj(h0).view(bsz, t0, self.num_kv_heads, self.head_dim))
            vs.append(self.vlm.v_proj(h0).view(bsz, t0, self.num_kv_heads, self.head_dim))

        if expert_x is not None:
            bsz, t1, _ = expert_x.shape
            h1, expert_gate = self.expert.input_layernorm(expert_x, adarms_cond)
            qs.append(self.expert.q_proj(h1).view(bsz, t1, self.num_heads, self.head_dim))
            ks.append(self.expert.k_proj(h1).view(bsz, t1, self.num_kv_heads, self.head_dim))
            vs.append(self.expert.v_proj(h1).view(bsz, t1, self.num_kv_heads, self.head_dim))

        q = torch.cat(qs, dim=1)
        k = torch.cat(ks, dim=1)
        v = torch.cat(vs, dim=1)

        q = apply_rope(q, positions, self.rope_max_wavelength) * (self.head_dim**-0.5)
        k = apply_rope(k, positions, self.rope_max_wavelength)

        if kv_cache_layer is not None:
            cache_k, cache_v = kv_cache_layer
            k = torch.cat([cache_k, k], dim=1)
            v = torch.cat([cache_v, v], dim=1)

        encoded = self._attend(q, k, v, attn_mask_bool, exact_fp32_attn=exact_fp32_attn)

        cursor = 0
        out_vlm: Tensor | None = None
        out_expert: Tensor | None = None

        if vlm_x is not None:
            t0 = vlm_x.shape[1]
            y0 = self.vlm.o_proj(encoded[:, cursor : cursor + t0])
            x0 = gated_residual(vlm_x, y0, vlm_gate)
            assert x0 is not None
            h0_ffn, g0_ffn = self.vlm.post_attention_layernorm(x0, None)
            y0_ffn = self.vlm.mlp(h0_ffn)
            out_vlm = gated_residual(x0, y0_ffn, g0_ffn)
            cursor += t0

        if expert_x is not None:
            t1 = expert_x.shape[1]
            y1 = self.expert.o_proj(encoded[:, cursor : cursor + t1])
            x1_res = gated_residual(expert_x, y1, expert_gate)
            assert x1_res is not None
            # Upstream `HistoryBlock` (history_gemma.py L177-202) modulates the FFN branch input `x`,
            # while keeping post-attention `xs[1]` (`x1_res`) as the clean skip connection!
            x1_mod = self._apply_expert_memory_mod(x1_res, mem_tokens, mem_mask, mem_kv_layer=mem_kv_layer)
            h1_ffn, g1_ffn = self.expert.post_attention_layernorm(x1_mod, adarms_cond)
            y1_ffn = self.expert.mlp(h1_ffn)
            out_expert = gated_residual(x1_res, y1_ffn, g1_ffn)

        return out_vlm, out_expert, (k, v)


class PaliGemmaWithActionExpert(nn.Module):
    """18-layer Dual-Stream PaliGemma-2B + ActionExpert-300M Transformer."""

    def __init__(self, config: Pi05Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.paligemma_width)
        self.layers = nn.ModuleList([JointHistoryBlock(config) for _ in range(config.depth)])
        self.vlm_norm = Pi05RMSNorm(config.paligemma_width, eps=config.rms_norm_eps, cond_dim=None)
        self.expert_norm = Pi05RMSNorm(
            config.expert_width, eps=config.rms_norm_eps, cond_dim=config.expert_width
        )
        self.gradient_checkpointing: bool = False

    def embed_language(self, token_ids: Tensor) -> Tensor:
        """Matches `gemma.Embedder.encode`: table lookup * `sqrt(width)` cast to embed dtype."""
        emb = self.embed_tokens(token_ids)
        scale = torch.tensor(math.sqrt(self.config.paligemma_width), dtype=emb.dtype, device=emb.device)
        return emb * scale

    def enable_memory_modulation(self, mode: MemModMode) -> None:
        self.config.mem_mod_mode = mode
        device = self.embed_tokens.weight.device
        target_dtype = self.layers[0].expert.q_proj.weight.dtype
        for block in self.layers:
            block.init_memory_hook(mode, self.config)
            if block.mem_attn is not None:
                block.mem_attn.to(device=device, dtype=target_dtype)
                block.mem_attn.mem_rms_norm.weight.data = block.mem_attn.mem_rms_norm.weight.data.float()
            if block.mem_rms_norm_ffn is not None:
                block.mem_rms_norm_ffn.to(device=device, dtype=target_dtype)

    def prefill_prefix(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_attn_mask_bool: Tensor,
        *,
        exact_fp32_attn: bool = False,
    ) -> tuple[Tensor, PrefixKVCache]:
        positions = torch.cumsum(prefix_pad_masks.long(), dim=1) - 1
        x0: Tensor | None = prefix_embs
        k_list: list[Tensor] = []
        v_list: list[Tensor] = []

        for block in self.layers:
            x0, _, (k, v) = block(
                x0,
                None,
                positions,
                prefix_attn_mask_bool,
                None,
                exact_fp32_attn=exact_fp32_attn,
            )
            k_list.append(k)
            v_list.append(v)

        assert x0 is not None
        x0_out, _ = self.vlm_norm(x0, None)
        cache = PrefixKVCache(
            keys=torch.stack(k_list, dim=0),
            values=torch.stack(v_list, dim=0),
            pad_masks=prefix_pad_masks,
            prefix_offsets=torch.sum(prefix_pad_masks.long(), dim=-1, keepdim=True),
        )
        return x0_out, cache

    def precompute_memory_kv(
        self,
        mem_tokens: Tensor | None,
        mem_mask: Tensor | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> list[tuple[Tensor, Tensor, Tensor, int] | None] | None:
        """Precompute per-layer `(K_mem, V_mem)` once per chunk across all 10 Euler ODE steps (`0.0` diff)."""
        if mem_tokens is None or mem_tokens.shape[1] == 0:
            return None
        out: list[tuple[Tensor, Tensor, Tensor, int] | None] = []
        for blk in self.layers:
            if blk.mem_mod_mode != "none" and blk.mem_attn is not None and blk.mem_rms_norm_ffn is not None:
                out.append(blk.mem_attn.precompute_kv(mem_tokens, mem_mask, dtype=dtype))
            else:
                out.append(None)
        return out

    def forward_suffix_with_kv(
        self,
        suffix_embs: Tensor,
        suffix_pad_masks: Tensor,
        suffix_attn_mask_bool: Tensor,
        adarms_cond: Tensor,
        kv_cache: PrefixKVCache,
        *,
        mem_tokens: Tensor | None = None,
        mem_mask: Tensor | None = None,
        mem_kv_cache: list[tuple[Tensor, Tensor, Tensor, int] | None] | None = None,
        exact_fp32_attn: bool = False,
    ) -> Tensor:
        bsz, s_len = suffix_pad_masks.shape
        p_len = kv_cache.pad_masks.shape[1]
        prefix_2d = kv_cache.pad_masks[:, None, :].expand(bsz, s_len, p_len)
        full_attn_mask = torch.cat([prefix_2d, suffix_attn_mask_bool], dim=-1)
        positions = kv_cache.prefix_offsets + torch.cumsum(suffix_pad_masks.long(), dim=-1) - 1

        x1: Tensor | None = suffix_embs
        for idx, block in enumerate(self.layers):
            kv_layer = (kv_cache.keys[idx], kv_cache.values[idx])
            m_kv_layer = mem_kv_cache[idx] if mem_kv_cache is not None else None
            if self.gradient_checkpointing and self.training:

                def _run_block(
                    inp: Tensor,
                    blk: JointHistoryBlock = block,
                    kvl: tuple[Tensor, Tensor] = kv_layer,
                    mkvl: tuple[Tensor, Tensor, Tensor, int] | None = m_kv_layer,
                ) -> Tensor:
                    _, out_e, _ = blk(
                        None,
                        inp,
                        positions,
                        full_attn_mask,
                        adarms_cond,
                        kv_cache_layer=kvl,
                        mem_tokens=mem_tokens,
                        mem_mask=mem_mask,
                        mem_kv_layer=mkvl,
                        exact_fp32_attn=exact_fp32_attn,
                    )
                    assert out_e is not None
                    return out_e

                assert x1 is not None
                x1 = torch.utils.checkpoint.checkpoint(
                    _run_block, x1, use_reentrant=False, preserve_rng_state=False
                )
            else:
                _, x1, _ = block(
                    None,
                    x1,
                    positions,
                    full_attn_mask,
                    adarms_cond,
                    kv_cache_layer=kv_layer,
                    mem_tokens=mem_tokens,
                    mem_mask=mem_mask,
                    mem_kv_layer=m_kv_layer,
                    exact_fp32_attn=exact_fp32_attn,
                )

        assert x1 is not None
        x1_out, _ = self.expert_norm(x1, adarms_cond)
        return x1_out

    def forward_joint(
        self,
        prefix_embs: Tensor,
        suffix_embs: Tensor,
        positions: Tensor,
        full_attn_mask_bool: Tensor,
        adarms_cond: Tensor,
        *,
        mem_tokens: Tensor | None = None,
        mem_mask: Tensor | None = None,
        exact_fp32_attn: bool = False,
    ) -> tuple[Tensor, Tensor]:
        x0: Tensor | None = prefix_embs
        x1: Tensor | None = suffix_embs

        for block in self.layers:
            if self.gradient_checkpointing and self.training:

                def _run_joint(
                    in0: Tensor,
                    in1: Tensor,
                    blk: JointHistoryBlock = block,
                ) -> tuple[Tensor, Tensor]:
                    o0, o1, _ = blk(
                        in0,
                        in1,
                        positions,
                        full_attn_mask_bool,
                        adarms_cond,
                        mem_tokens=mem_tokens,
                        mem_mask=mem_mask,
                        exact_fp32_attn=exact_fp32_attn,
                    )
                    assert o0 is not None and o1 is not None
                    return o0, o1

                assert x0 is not None and x1 is not None
                x0, x1 = torch.utils.checkpoint.checkpoint(
                    _run_joint, x0, x1, use_reentrant=False, preserve_rng_state=False
                )
            else:
                x0, x1, _ = block(
                    x0,
                    x1,
                    positions,
                    full_attn_mask_bool,
                    adarms_cond,
                    mem_tokens=mem_tokens,
                    mem_mask=mem_mask,
                    exact_fp32_attn=exact_fp32_attn,
                )

        assert x0 is not None and x1 is not None
        x0_out, _ = self.vlm_norm(x0, None)
        x1_out, _ = self.expert_norm(x1, adarms_cond)
        return x0_out, x1_out


def inject_lora_into_stream(
    layers: Sequence[JointHistoryBlock],
    *,
    stream: str = "vlm",
    rank: int = 32,
    alpha: float = 32.0,
    include_mlp: bool = True,
) -> list[nn.Parameter]:
    """Inject zero-init LoRA adapters into `vlm` (`PaliGemma-2B`) or `expert` (`ActionExpert-300M`)."""
    added: list[nn.Parameter] = []
    for block in layers:
        target = block.vlm if stream == "vlm" else block.expert
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(target, attr)
            if isinstance(mod, nn.Linear):
                wrapped = LoRALinear(mod, rank=rank, alpha=alpha)
                setattr(target, attr, wrapped)
                added.extend([wrapped.lora_a, wrapped.lora_b])
        if include_mlp:
            for attr in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(target.mlp, attr)
                if isinstance(mod, nn.Linear):
                    wrapped = LoRALinear(mod, rank=rank, alpha=alpha)
                    setattr(target.mlp, attr, wrapped)
                    added.extend([wrapped.lora_a, wrapped.lora_b])
    return added
