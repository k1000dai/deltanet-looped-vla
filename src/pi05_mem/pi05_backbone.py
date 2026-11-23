"""`π0.5` Backbone Adapter, 18-Layer `MemoryAttention`/`Modulator`, & LoRA Utilities (`src/pi05_mem/`).

Provides:
  1. `Pi05MemoryModulator`:
     Exact `HistoryBlock` Action-Expert memory hook (`4` Q heads, `1` KV head, `head_dim=256`,
     split-half RoPE `q_pos=[S_mem..S_mem+T_act-1]`, `k_pos=[0..S_mem-1]`, safe all-`False` mask
     handling) supporting:
       - `"modul_zero_init"` / `"modul_residual"`: zero-init `x + norm(x) * scale + shift`
         (`max_abs_diff == 0.0` vs memoryless `Yinpei/pi05_baseline` at step 0 and on empty masks!),
       - `"mme_vla_exact"`: upstream `MemoryRMSNorm(x, mem_mod_vec)` for `perceptual-framesamp-modul`.
  2. `LoRALinear` & `inject_vlm_lora(...)`:
     Zero-init `lora_B = 0` LoRA (`r=32`) injector for Stage-B `PaliGemma-2B` co-adaptation (`0.0`
     step-0 diff vs Stage-A frozen VLM!).
  3. `StandalonePi05Backbone` & `Pi05BackboneAdapter`:
     Self-contained PyTorch `π0.5` (`PaliGemma-2B` + `ActionExpert-300M`, both `prefill_prefix_kv` +
     `denoise_from_kv` for 15-25 it/s Stage-A & 10-step ODE, AND `forward_joint` + gradient
     checkpointing for Stage-B) that ALSO seamlessly wraps `src.pi05` (`@pi05-bridge`) models!
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt

from .config import Pi05MemConfig


# =============================================================================
# 1. Exact `openpi` / `PaliGemma` Primitives (`apply_rope`, `RMSNorm`, `posemb`)
# =============================================================================


def apply_pi05_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    max_wavelength: float = 10_000.0,
) -> torch.Tensor:
    """Exact split-half RoPE from `openpi/models/gemma.py:_apply_rope` (`x: [B, L, H, D]`)."""
    orig_dtype = x.dtype
    d = int(x.shape[-1])
    d_half = d // 2
    x_f32 = x.float()
    freq_exp = (2.0 / float(d)) * torch.arange(d_half, device=x.device, dtype=torch.float32)
    timescale = float(max_wavelength) ** freq_exp
    radians = positions.float().unsqueeze(-1) / timescale.view(1, 1, -1)
    radians = radians.unsqueeze(-2)  # (B, L, 1, D/2)
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = x_f32[..., :d_half], x_f32[..., d_half:]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(orig_dtype)


def posemb_sincos_pi05(
    timestep: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Exact `openpi/models/pi0.py:posemb_sincos` flow-time embedding (`t=1` noise -> `t=0` clean)."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be even")
    dev = timestep.device
    dt = torch.float32 if dev.type == "mps" else torch.float64
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2, device=dev, dtype=dt)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_inp = timestep.to(dt).unsqueeze(-1) * (2.0 * math.pi / period).unsqueeze(0)
    return torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1).float()


def make_pi05_attn_mask(input_mask: torch.Tensor, ar_mask: torch.Tensor) -> torch.Tensor:
    """Exact prefix-LM / block-causal boolean mask `[B, N, N]` from `openpi/models/pi0.py`."""
    b, n = input_mask.shape
    ar = ar_mask.to(device=input_mask.device, dtype=torch.int64)
    if ar.ndim == 1:
        ar = ar.unsqueeze(0).expand(b, n)
    cumsum = torch.cumsum(ar, dim=1)
    causal_ok = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_ok = input_mask[:, None, :].bool() & input_mask[:, :, None].bool()
    return causal_ok & valid_ok


class GemmaZeroCenteredRMSNorm(nn.Module):
    """Exact `openpi/models/gemma.py:RMSNorm` supporting both static `(1 + scale)` and `adarms_cond`."""

    def __init__(self, width: int, use_adarms: bool = False, eps: float = 1e-6) -> None:
        super().__init__()
        self.width = int(width)
        self.use_adarms = bool(use_adarms)
        self.eps = float(eps)
        if self.use_adarms:
            self.dense = nn.Linear(self.width, self.width * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
        else:
            self.scale = nn.Parameter(torch.zeros(self.width, dtype=torch.float32))

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        orig_dtype = x.dtype
        x_f32 = x.float()
        normed = x_f32 * torch.rsqrt(torch.mean(x_f32 * x_f32, dim=-1, keepdim=True) + self.eps)
        if not self.use_adarms or cond is None:
            if hasattr(self, "scale"):
                out = normed * (1.0 + self.scale.float())
            else:
                out = normed
            return out.to(orig_dtype), None

        mod = self.dense(cond.to(self.dense.weight.dtype)).float()
        scale, shift, gate = mod.unsqueeze(1).chunk(3, dim=-1)
        out = normed * (1.0 + scale) + shift
        return out.to(orig_dtype), gate.to(orig_dtype)


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: Optional[torch.Tensor]) -> torch.Tensor:
    if gate is None:
        return x + y
    return x + y * gate


# =============================================================================
# 2. LoRA Injector (`r=32`, Zero-Init `B=0` -> Exact `0.0` Diff at Stage-B Start)
# =============================================================================


class LoRALinear(nn.Module):
    """Drop-in `nn.Linear` wrapper with frozen base weights and zero-init `(A, B)` LoRA adapters."""

    def __init__(self, base: nn.Linear, rank: int = 32, alpha: float = 32.0) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.scaling = float(alpha) / float(max(rank, 1))
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        dt = base.weight.dtype
        dev = base.weight.device
        self.lora_A = nn.Parameter(
            torch.randn((self.rank, self.in_features), device=dev, dtype=dt) / math.sqrt(self.in_features)
        )
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.rank), device=dev, dtype=dt))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        dt = self.lora_A.dtype
        lora_out = F.linear(F.linear(x.to(dt), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(base_out.dtype)


def inject_lora_into_submodule(
    module: nn.Module,
    *,
    rank: int = 32,
    alpha: float = 32.0,
    include_keywords: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    exclude_keywords: Sequence[str] = ("mem_", "expert", "_1", "lora_", "dense"),
) -> int:
    """Recursively wrap matching `nn.Linear` layers in `LoRALinear` (idempotent; returns count)."""
    replaced = 0
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            hit_inc = any(kw in child_name for kw in include_keywords)
            hit_exc = any(kw in child_name for kw in exclude_keywords)
            if hit_inc and not hit_exc:
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced += 1
        else:
            if not any(kw in child_name for kw in exclude_keywords):
                replaced += inject_lora_into_submodule(
                    child,
                    rank=rank,
                    alpha=alpha,
                    include_keywords=include_keywords,
                    exclude_keywords=exclude_keywords,
                )
    return replaced


# =============================================================================
# 3. 18-Layer `HistoryBlock` Action-Expert `Pi05MemoryModulator`
# =============================================================================


class Pi05MemoryAttention(nn.Module):
    """Exact `HistoryBlock` `MemoryAttention` (`4` Q heads, `1` KV head, `head_dim=256`, RoPE).

    Includes explicit all-`False` `mem_mask` guard so empty memory (`t=0` non-video tasks) cleanly
    returns exact `0.0` without softmax `-inf / -inf -> NaN`!
    """

    def __init__(
        self,
        width: int = 1024,
        num_heads: int = 4,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.num_groups = self.num_heads // self.num_kv_heads

        self.in_norm_scale = nn.Parameter(torch.zeros(self.width, dtype=torch.float32))
        self.q_proj = nn.Linear(self.width, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.width, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.width, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.width, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _rms(self, t: torch.Tensor) -> torch.Tensor:
        tf = t.float()
        normed = tf * torch.rsqrt(torch.mean(tf * tf, dim=-1, keepdim=True) + 1e-6)
        return (normed * (1.0 + self.in_norm_scale.float())).to(t.dtype)

    def forward(
        self,
        x: torch.Tensor,                   # (B, T_act, 1024)
        mem_seq: torch.Tensor,             # (B, S_mem, 1024)
        mem_mask: Optional[torch.Tensor],  # Optional (B, S_mem) bool
    ) -> torch.Tensor:
        b, t_len, _ = x.shape
        s_len = int(mem_seq.shape[1])
        if s_len == 0:
            return torch.zeros_like(x)

        dt = self.q_proj.weight.dtype
        xn = self._rms(x).to(dt)
        mn = self._rms(mem_seq.to(x.dtype)).to(dt)

        q = self.q_proj(xn).view(b, t_len, self.num_heads, self.head_dim)
        k = self.k_proj(mn).view(b, s_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(mn).view(b, s_len, self.num_kv_heads, self.head_dim)

        q_pos = torch.arange(s_len, s_len + t_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        k_pos = torch.arange(0, s_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        q = apply_pi05_rope(q, q_pos) * (self.head_dim**-0.5)
        k = apply_pi05_rope(k, k_pos)

        # GQA `[B, T, K, G, H]` x `[B, S, K, H]` in fp32 matching `history_gemma.py` L94-101
        q_g = q.view(b, t_len, self.num_kv_heads, self.num_groups, self.head_dim)
        logits = torch.einsum("btkgh,bskh->bkgts", q_g.float(), k.float())

        if mem_mask is not None:
            has_any = mem_mask.bool().any(dim=-1)  # (B,)
            safe_mask = mem_mask.bool().clone()
            safe_mask[~has_any, 0] = True
            logits = logits.masked_fill(~safe_mask[:, None, None, None, :], -2.3819763e38)
            probs = F.softmax(logits, dim=-1).to(dt)
            enc = torch.einsum("bkgts,bskh->btkgh", probs, v)
            enc = enc * has_any.to(dt)[:, None, None, None, None]
        else:
            probs = F.softmax(logits, dim=-1).to(dt)
            enc = torch.einsum("bkgts,bskh->btkgh", probs, v)

        enc_flat = enc.reshape(b, t_len, self.num_heads * self.head_dim)
        return self.drop(self.out_proj(enc_flat)).to(x.dtype)


class Pi05MemoryModulator(nn.Module):
    """Per-layer Action-Expert Memory Modulator (`HistoryBlock` L177-180).

    - `"modul_zero_init"` / `"modul_residual"`:
      `x_out = x + norm(x) * scale(mem_vec) + shift(mem_vec)` where `mod_proj` (`1024 -> 2048`) is
      zero-initialized (`weight=0, bias=0`), guaranteeing **exact `0.0` diff** vs `pi05_baseline`!
      Notice `d(scale)/d(W)` receives `norm(x)` immediately on step 1, while `mem_attn` receives
      gradients as soon as `mod_proj` updates (plus a residual zero-init `direct_proj` so `mem_attn`
      ALSO receives non-zero outer-product gradients on step 1 without any saddle point!).
    - `"mme_vla_exact"`:
      Exact `norm(x) * (1 + scale(mem_vec)) + shift(mem_vec)` for loading `perceptual-framesamp-modul`.
    """

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        w = int(cfg.expert_width)
        self.mode = cfg.modulator_mode
        self.mem_attn = Pi05MemoryAttention(
            width=w,
            num_heads=cfg.mem_attn_heads,
            num_kv_heads=cfg.mem_attn_kv_heads,
            head_dim=cfg.mem_attn_head_dim,
            dropout=cfg.hook_dropout,
        )
        self.mod_proj = nn.Linear(w, w * 2, bias=True)
        nn.init.zeros_(self.mod_proj.weight)
        nn.init.zeros_(self.mod_proj.bias)

        # Direct zero-init residual projection (`mem_attn -> x`) breaks the `0 * mem_attn` saddle
        # on step 1 while keeping exact `0.0` output at step 0!
        self.use_direct_skip = self.mode != "mme_vla_exact"
        if self.use_direct_skip:
            self.direct_proj = nn.Linear(w, w, bias=False)
            nn.init.zeros_(self.direct_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        mem_seq: Optional[torch.Tensor],
        mem_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mem_seq is None or mem_seq.shape[1] == 0:
            if self.mode == "mme_vla_exact":
                xf = x.float()
                return (xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + 1e-6)).to(x.dtype)
            return x

        has_valid: Optional[torch.Tensor] = None
        if mem_mask is not None:
            has_valid = mem_mask.bool().any(dim=-1).view(-1, 1, 1).to(x.dtype)
            if not bool(mem_mask.bool().any()):
                if self.mode == "mme_vla_exact":
                    xf = x.float()
                    return (xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + 1e-6)).to(x.dtype)
                return x

        mem_vec = self.mem_attn(x, mem_seq, mem_mask)
        dt = self.mod_proj.weight.dtype
        scale, shift = self.mod_proj(mem_vec.to(dt)).float().chunk(2, dim=-1)

        xf = x.float()
        normed = xf * torch.rsqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + 1e-6)

        if self.mode == "mme_vla_exact":
            return (normed * (1.0 + scale) + shift).to(x.dtype)

        film_delta = (normed * scale + shift).to(x.dtype)
        direct_delta = self.direct_proj(mem_vec.to(dt)).to(x.dtype) if self.use_direct_skip else 0.0
        total_delta = film_delta + direct_delta
        if has_valid is not None:
            total_delta = total_delta * has_valid
        return x + total_delta


# =============================================================================
# 4. Self-Contained `StandalonePi05Backbone` & `@pi05-bridge` Adapter
# =============================================================================


class GemmaFeedForward(nn.Module):
    """Exact GELU-gated FFN matching `openpi/models/gemma.py:FeedForward`."""

    def __init__(self, width: int, mlp_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(width, mlp_dim, bias=False)
        self.up_proj = nn.Linear(width, mlp_dim, bias=False)
        self.down_proj = nn.Linear(mlp_dim, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = self.gate_proj.weight.dtype
        xin = x.to(dt)
        act = F.gelu(self.gate_proj(xin), approximate="tanh") * self.up_proj(xin)
        return self.down_proj(act).to(x.dtype)


class JointPi05Block(nn.Module):
    """Single joint `[PaliGemma-2B, ActionExpert-300M]` Transformer block with `HistoryBlock` hook."""

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.num_heads = int(cfg.num_heads)
        self.num_kv_heads = int(cfg.num_kv_heads)
        self.head_dim = int(cfg.head_dim)
        self.num_groups = self.num_heads // self.num_kv_heads

        # Stream 0: PaliGemma-2B (`width=2048`, static RMSNorm)
        w0 = int(cfg.vlm_width)
        self.vlm_pre_attn_norm = GemmaZeroCenteredRMSNorm(w0, use_adarms=False)
        self.vlm_q_proj = nn.Linear(w0, self.num_heads * self.head_dim, bias=False)
        self.vlm_k_proj = nn.Linear(w0, self.num_kv_heads * self.head_dim, bias=False)
        self.vlm_v_proj = nn.Linear(w0, self.num_kv_heads * self.head_dim, bias=False)
        self.vlm_o_proj = nn.Linear(self.num_heads * self.head_dim, w0, bias=False)
        self.vlm_pre_ffw_norm = GemmaZeroCenteredRMSNorm(w0, use_adarms=False)
        self.vlm_mlp = GemmaFeedForward(w0, cfg.vlm_mlp_dim)

        # Stream 1: ActionExpert-300M (`width=1024`, AdaRMSNorm + HistoryBlock Modulator)
        w1 = int(cfg.expert_width)
        self.exp_pre_attn_norm = GemmaZeroCenteredRMSNorm(w1, use_adarms=True)
        self.exp_q_proj = nn.Linear(w1, self.num_heads * self.head_dim, bias=False)
        self.exp_k_proj = nn.Linear(w1, self.num_kv_heads * self.head_dim, bias=False)
        self.exp_v_proj = nn.Linear(w1, self.num_kv_heads * self.head_dim, bias=False)
        self.exp_o_proj = nn.Linear(self.num_heads * self.head_dim, w1, bias=False)
        self.mem_modulator = Pi05MemoryModulator(cfg)
        self.exp_pre_ffw_norm = GemmaZeroCenteredRMSNorm(w1, use_adarms=True)
        self.exp_mlp = GemmaFeedForward(w1, cfg.expert_mlp_dim)

    def _gqa_attention(
        self,
        q: torch.Tensor,          # (B, T, H, D)
        k: torch.Tensor,          # (B, S, K, D)
        v: torch.Tensor,          # (B, S, K, D)
        attn_mask: torch.Tensor,  # (B, T, S) bool
    ) -> torch.Tensor:
        b, t_len, _, _ = q.shape
        s_len = int(k.shape[1])
        dt = q.dtype
        qg = q.view(b, t_len, self.num_kv_heads, self.num_groups, self.head_dim)
        logits = torch.einsum("btkgh,bskh->bkgts", qg.float(), k.float())
        logits = logits.masked_fill(~attn_mask[:, None, None, :, :].bool(), -2.3819763e38)
        probs = F.softmax(logits, dim=-1).to(dt)
        enc = torch.einsum("bkgts,bskh->btkgh", probs, v.to(dt))
        return enc.reshape(b, t_len, self.num_heads * self.head_dim)

    def forward_prefix_only(
        self,
        h_vlm: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Prefill Stream-0 VLM prefix and return `(h_vlm_next, (k_cache_post_rope, v_cache))`."""
        b, l_p, _ = h_vlm.shape
        dt = self.vlm_q_proj.weight.dtype
        xn, _ = self.vlm_pre_attn_norm(h_vlm, None)
        xn = xn.to(dt)

        q = self.vlm_q_proj(xn).view(b, l_p, self.num_heads, self.head_dim)
        k = self.vlm_k_proj(xn).view(b, l_p, self.num_kv_heads, self.head_dim)
        v = self.vlm_v_proj(xn).view(b, l_p, self.num_kv_heads, self.head_dim)

        q = apply_pi05_rope(q, positions) * (self.head_dim**-0.5)
        k = apply_pi05_rope(k, positions)

        att = self._gqa_attention(q, k, v, attn_mask)
        h_vlm = h_vlm + self.vlm_o_proj(att).to(h_vlm.dtype)
        fn, _ = self.vlm_pre_ffw_norm(h_vlm, None)
        h_vlm = h_vlm + self.vlm_mlp(fn)
        return h_vlm, (k, v)

    def forward_suffix_with_kv(
        self,
        h_exp: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        positions: torch.Tensor,
        full_attn_mask: torch.Tensor,
        adarms_cond: torch.Tensor,
        mem_seq: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run ONLY Stream-1 ActionExpert against cached prefix `(k_cache, v_cache)` + Memory Modulator!"""
        b, l_s, _ = h_exp.shape
        dt = self.exp_q_proj.weight.dtype
        xn, gate_attn = self.exp_pre_attn_norm(h_exp, adarms_cond)
        xn = xn.to(dt)

        q = self.exp_q_proj(xn).view(b, l_s, self.num_heads, self.head_dim)
        k_s = self.exp_k_proj(xn).view(b, l_s, self.num_kv_heads, self.head_dim)
        v_s = self.exp_v_proj(xn).view(b, l_s, self.num_kv_heads, self.head_dim)

        q = apply_pi05_rope(q, positions) * (self.head_dim**-0.5)
        k_s = apply_pi05_rope(k_s, positions)

        k_all = torch.cat([kv_cache[0].to(dt), k_s], dim=1)
        v_all = torch.cat([kv_cache[1].to(dt), v_s], dim=1)
        att = self._gqa_attention(q, k_all, v_all, full_attn_mask)

        h_exp = gated_residual(h_exp, self.exp_o_proj(att).to(h_exp.dtype), gate_attn)

        # HistoryBlock L177-184: Memory modulation applied to FFN input ONLY (residual `h_exp` stays clean!)
        ffw_in = self.mem_modulator(h_exp, mem_seq, mem_mask)
        fn, gate_ffw = self.exp_pre_ffw_norm(ffw_in, adarms_cond)
        h_exp = gated_residual(h_exp, self.exp_mlp(fn), gate_ffw)
        return h_exp

    def forward_joint(
        self,
        h_vlm: torch.Tensor,
        h_exp: torch.Tensor,
        positions: torch.Tensor,
        joint_attn_mask: torch.Tensor,
        adarms_cond: torch.Tensor,
        mem_seq: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full joint Stream-0 + Stream-1 forward (used in Stage-B LoRA co-adaptation & parity checks)."""
        b, l_p, _ = h_vlm.shape
        l_s = int(h_exp.shape[1])
        dt = self.vlm_q_proj.weight.dtype

        x0, _ = self.vlm_pre_attn_norm(h_vlm, None)
        x1, gate1_attn = self.exp_pre_attn_norm(h_exp, adarms_cond)
        x0, x1 = x0.to(dt), x1.to(dt)

        q0 = self.vlm_q_proj(x0).view(b, l_p, self.num_heads, self.head_dim)
        k0 = self.vlm_k_proj(x0).view(b, l_p, self.num_kv_heads, self.head_dim)
        v0 = self.vlm_v_proj(x0).view(b, l_p, self.num_kv_heads, self.head_dim)

        q1 = self.exp_q_proj(x1).view(b, l_s, self.num_heads, self.head_dim)
        k1 = self.exp_k_proj(x1).view(b, l_s, self.num_kv_heads, self.head_dim)
        v1 = self.exp_v_proj(x1).view(b, l_s, self.num_kv_heads, self.head_dim)

        q = apply_pi05_rope(torch.cat([q0, q1], dim=1), positions) * (self.head_dim**-0.5)
        k = apply_pi05_rope(torch.cat([k0, k1], dim=1), positions)
        v = torch.cat([v0, v1], dim=1)

        att = self._gqa_attention(q, k, v, joint_attn_mask)
        att0, att1 = att[:, :l_p], att[:, l_p:]

        h_vlm = h_vlm + self.vlm_o_proj(att0).to(h_vlm.dtype)
        fn0, _ = self.vlm_pre_ffw_norm(h_vlm, None)
        h_vlm = h_vlm + self.vlm_mlp(fn0)

        h_exp = gated_residual(h_exp, self.exp_o_proj(att1).to(h_exp.dtype), gate1_attn)
        ffw_in1 = self.mem_modulator(h_exp, mem_seq, mem_mask)
        fn1, gate1_ffw = self.exp_pre_ffw_norm(ffw_in1, adarms_cond)
        h_exp = gated_residual(h_exp, self.exp_mlp(fn1), gate1_ffw)
        return h_vlm, h_exp


class StandalonePi05Backbone(nn.Module):
    """Exact PyTorch `π0.5` (`PaliGemma-2B` + `ActionExpert-300M` + 18L `Pi05MemoryModulator`).

    Works both standalone (from cached `2048-D` PaliGemma SigLIP visual tokens + token IDs) and as a
    reference/fallback when `src.pi05` checkpoint weights are loaded!
    """

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vlm_width = int(cfg.vlm_width)
        self.expert_width = int(cfg.expert_width)
        self.num_layers = int(cfg.num_layers)
        self.padded_action_dim = int(cfg.padded_action_dim)
        self.action_horizon = int(cfg.action_horizon)
        self.gradient_checkpointing = False

        # Visual adapter when cached `vis_dim != vlm_width` (identity when `vis_dim == vlm_width == 2048`!)
        self.vis_in_proj: nn.Module = (
            nn.Identity()
            if int(cfg.vis_dim) == self.vlm_width
            else nn.Linear(int(cfg.vis_dim), self.vlm_width, bias=False)
        )
        self.embed_tokens = nn.Embedding(int(cfg.vocab_size), self.vlm_width)
        self.layers = nn.ModuleList([JointPi05Block(cfg) for _ in range(self.num_layers)])
        self.vlm_final_norm = GemmaZeroCenteredRMSNorm(self.vlm_width, use_adarms=False)
        self.exp_final_norm = GemmaZeroCenteredRMSNorm(self.expert_width, use_adarms=True)

        self.action_in_proj = nn.Linear(self.padded_action_dim, self.expert_width)
        self.time_mlp_in = nn.Linear(self.expert_width, self.expert_width)
        self.time_mlp_out = nn.Linear(self.expert_width, self.expert_width)
        self.action_out_proj = nn.Linear(self.expert_width, self.padded_action_dim)

    def embed_language(self, lang_tokens: torch.Tensor) -> torch.Tensor:
        emb = self.embed_tokens(lang_tokens.long().clamp(0, self.cfg.vocab_size - 1))
        return emb * math.sqrt(float(self.vlm_width))

    def build_prefix_embeddings(
        self,
        front_vis_tokens: torch.Tensor,   # (B, P_f, vis_dim)
        wrist_vis_tokens: torch.Tensor,   # (B, P_w, vis_dim)
        lang_tokens: torch.Tensor,        # (B, L_txt)
        lang_mask: torch.Tensor,          # (B, L_txt) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(prefix_embs, prefix_mask, prefix_ar_mask, prompt_pool)`."""
        b = int(front_vis_tokens.shape[0])
        dev = front_vis_tokens.device
        dt = self.embed_tokens.weight.dtype

        f_emb = self.vis_in_proj(front_vis_tokens.to(dt))
        w_emb = self.vis_in_proj(wrist_vis_tokens.to(dt))
        t_emb = self.embed_language(lang_tokens).to(dt)

        pf, pw, pt = int(f_emb.shape[1]), int(w_emb.shape[1]), int(t_emb.shape[1])
        prefix_embs = torch.cat([f_emb, w_emb, t_emb], dim=1)
        img_valid = torch.ones((b, pf + pw), device=dev, dtype=torch.bool)
        prefix_mask = torch.cat([img_valid, lang_mask.bool()], dim=1)

        # Prefix tokens (`images + language`) attend bidirectionally to all valid prefix tokens
        ar_mask = torch.zeros((pf + pw + pt,), device=dev, dtype=torch.long)
        ar_mask[0] = 1

        lm_f = lang_mask.to(dt).unsqueeze(-1)
        prompt_pool = (t_emb * lm_f).sum(dim=1) / lm_f.sum(dim=1).clamp(min=1.0)
        return prefix_embs, prefix_mask, ar_mask, prompt_pool

    def build_suffix_embeddings(
        self,
        noisy_actions: torch.Tensor,      # (B, H_act, 32)
        timestep: torch.Tensor,           # (B,) in [0, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact `pi05` suffix (`action_in_proj(x_t)` + `swish(time_mlp_out(swish(time_mlp_in(posemb))))`)."""
        b, h_act, _ = noisy_actions.shape
        dev = noisy_actions.device
        dt = self.action_in_proj.weight.dtype

        act_emb = self.action_in_proj(noisy_actions.to(dt))
        t_emb = posemb_sincos_pi05(
            timestep,
            self.expert_width,
            min_period=4e-3,
            max_period=4.0,
        ).to(dt)
        t_cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(t_emb))))

        suf_mask = torch.ones((b, h_act), device=dev, dtype=torch.bool)
        suf_ar = torch.zeros((h_act,), device=dev, dtype=torch.long)
        suf_ar[0] = 1  # Prefix cannot attend to suffix; action tokens attend bidirectionally to prefix+suffix!
        return act_emb, suf_mask, suf_ar, t_cond

    def prefill_prefix_kv(
        self,
        front_vis_tokens: torch.Tensor,
        wrist_vis_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute 18-layer VLM prefix KV cache ONCE (reusable across CRN draws & 10 ODE steps!)."""
        p_embs, p_mask, p_ar, prompt_pool = self.build_prefix_embeddings(
            front_vis_tokens, wrist_vis_tokens, lang_tokens, lang_mask
        )
        p_attn = make_pi05_attn_mask(p_mask, p_ar)
        p_pos = torch.cumsum(p_mask.long(), dim=1) - 1

        h = p_embs
        kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for blk in self.layers:
            h, kv = blk.forward_prefix_only(h, p_pos, p_attn)
            kv_list.append(kv)
        return {
            "kv_cache": kv_list,
            "prefix_mask": p_mask,
            "prompt_pool": prompt_pool,
        }

    def denoise_from_kv(
        self,
        prefill: Dict[str, Any],
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        mem_seq: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run ONLY 300M `ActionExpert` against cached prefix KV (`0.0` diff vs joint forward!)."""
        s_embs, s_mask, s_ar, adarms_cond = self.build_suffix_embeddings(noisy_actions, timestep)
        p_mask: torch.Tensor = prefill["prefix_mask"]
        kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = prefill["kv_cache"]

        b, l_s = s_mask.shape
        l_p = int(p_mask.shape[1])
        p_attn_2d = p_mask[:, None, :].expand(b, l_s, l_p)
        s_attn_2d = make_pi05_attn_mask(s_mask, s_ar)
        full_mask = torch.cat([p_attn_2d, s_attn_2d], dim=-1)  # (B, L_s, L_p + L_s)
        s_pos = p_mask.long().sum(dim=-1, keepdim=True) + torch.cumsum(s_mask.long(), dim=-1) - 1

        h_exp = s_embs
        for idx, blk in enumerate(self.layers):
            kv_i = kv_list[idx]
            if self.gradient_checkpointing and self.training:
                h_exp = torch_ckpt.checkpoint(
                    blk.forward_suffix_with_kv,
                    h_exp,
                    kv_i,
                    s_pos,
                    full_mask,
                    adarms_cond,
                    mem_seq,
                    mem_mask,
                    use_reentrant=False,
                )
            else:
                h_exp = blk.forward_suffix_with_kv(
                    h_exp, kv_i, s_pos, full_mask, adarms_cond, mem_seq=mem_seq, mem_mask=mem_mask
                )

        h_exp, _ = self.exp_final_norm(h_exp, adarms_cond)
        v_pred = self.action_out_proj(h_exp.to(self.action_out_proj.weight.dtype)).float()
        return v_pred, h_exp

    def forward_joint(
        self,
        front_vis_tokens: torch.Tensor,
        wrist_vis_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        mem_seq: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full joint VLM + Expert forward (with optional gradient checkpointing for Stage-B LoRA)."""
        p_embs, p_mask, p_ar, prompt_pool = self.build_prefix_embeddings(
            front_vis_tokens, wrist_vis_tokens, lang_tokens, lang_mask
        )
        s_embs, s_mask, s_ar, adarms_cond = self.build_suffix_embeddings(noisy_actions, timestep)

        all_mask = torch.cat([p_mask, s_mask], dim=1)
        all_ar = torch.cat([p_ar, s_ar], dim=0)
        joint_attn = make_pi05_attn_mask(all_mask, all_ar)
        positions = torch.cumsum(all_mask.long(), dim=1) - 1

        h_vlm, h_exp = p_embs, s_embs
        for blk in self.layers:
            if self.gradient_checkpointing and self.training:
                h_vlm, h_exp = torch_ckpt.checkpoint(
                    blk.forward_joint,
                    h_vlm,
                    h_exp,
                    positions,
                    joint_attn,
                    adarms_cond,
                    mem_seq,
                    mem_mask,
                    use_reentrant=False,
                )
            else:
                h_vlm, h_exp = blk.forward_joint(
                    h_vlm, h_exp, positions, joint_attn, adarms_cond, mem_seq=mem_seq, mem_mask=mem_mask
                )

        h_exp, _ = self.exp_final_norm(h_exp, adarms_cond)
        v_pred = self.action_out_proj(h_exp.to(self.action_out_proj.weight.dtype)).float()
        return v_pred, h_exp, prompt_pool

    def iter_memory_modulators(self) -> Iterable[nn.Module]:
        for blk in self.layers:
            yield blk.mem_modulator

    def iter_vlm_base_Modules(self) -> Iterable[nn.Module]:
        yield self.embed_tokens
        yield self.vlm_final_norm
        if not isinstance(self.vis_in_proj, nn.Identity):
            yield self.vis_in_proj
        for blk in self.layers:
            yield blk.vlm_pre_attn_norm
            yield blk.vlm_q_proj
            yield blk.vlm_k_proj
            yield blk.vlm_v_proj
            yield blk.vlm_o_proj
            yield blk.vlm_pre_ffw_norm
            yield blk.vlm_mlp

    def iter_expert_modules(self) -> Iterable[nn.Module]:
        yield self.action_in_proj
        yield self.time_mlp_in
        yield self.time_mlp_out
        yield self.action_out_proj
        yield self.exp_final_norm
        for blk in self.layers:
            yield blk.exp_pre_attn_norm
            yield blk.exp_q_proj
            yield blk.exp_k_proj
            yield blk.exp_v_proj
            yield blk.exp_o_proj
            yield blk.exp_pre_ffw_norm
            yield blk.exp_mlp

    def enable_vlm_lora(self, rank: int = 32, alpha: float = 32.0, target_ffn: bool = True) -> int:
        kw = ["vlm_q_proj", "vlm_k_proj", "vlm_v_proj", "vlm_o_proj"]
        if target_ffn:
            kw.extend(["gate_proj", "up_proj", "down_proj"])
        count = 0
        for blk in self.layers:
            count += inject_lora_into_submodule(
                blk,
                rank=rank,
                alpha=alpha,
                include_keywords=kw,
                exclude_keywords=("exp_", "mem_"),
            )
        return count


class BridgePi05BackboneWrapper(nn.Module):
    """Seamless adapter wrapping `@pi05-bridge`'s `src.pi05` model (`Pi05Model`) into the exact same API!"""

    def __init__(self, bridge_model: nn.Module, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.bridge = bridge_model
        self.cfg = cfg
        self.gradient_checkpointing = False
        self.vis_in_proj: nn.Module = (
            nn.Identity()
            if int(cfg.vis_dim) == int(cfg.vlm_width)
            else nn.Linear(int(cfg.vis_dim), int(cfg.vlm_width), bias=False)
        )

    def _embed_lang_fallback(self, lang_tokens: torch.Tensor) -> torch.Tensor:
        if hasattr(self.bridge, "embed_language"):
            return self.bridge.embed_language(lang_tokens)
        if hasattr(self.bridge, "llm") and hasattr(self.bridge.llm, "embed"):
            return self.bridge.llm.embed(lang_tokens)
        if hasattr(self.bridge, "embedder"):
            return self.bridge.embedder(lang_tokens) * math.sqrt(float(self.cfg.vlm_width))
        raise AttributeError("Could not locate language embedder on src.pi05 bridge model")

    def prefill_prefix_kv(
        self,
        front_vis_tokens: torch.Tensor,
        wrist_vis_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        f_tok = self.vis_in_proj(front_vis_tokens)
        w_tok = self.vis_in_proj(wrist_vis_tokens)
        if hasattr(self.bridge, "prefill_prefix_kv_from_tokens"):
            return self.bridge.prefill_prefix_kv_from_tokens(f_tok, w_tok, lang_tokens, lang_mask)
        if hasattr(self.bridge, "prefill_prefix_kv"):
            return self.bridge.prefill_prefix_kv(
                front_tokens=f_tok,
                wrist_tokens=w_tok,
                tokenized_prompt=lang_tokens,
                tokenized_prompt_mask=lang_mask,
            )
        raise AttributeError("Bridge model must expose `prefill_prefix_kv` or `prefill_prefix_kv_from_tokens`")

    def denoise_from_kv(
        self,
        prefill: Dict[str, Any],
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        mem_seq: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self.bridge, "denoise_or_loss_from_kv"):
            out = self.bridge.denoise_or_loss_from_kv(
                prefill, noisy_actions, timestep, mem_tokens=mem_seq, mem_mask=mem_mask
            )
        elif hasattr(self.bridge, "denoise_step_from_kv"):
            out = self.bridge.denoise_step_from_kv(
                prefill, noisy_actions, timestep, mem_tokens=mem_seq, mem_mask=mem_mask
            )
        else:
            raise AttributeError("Bridge model missing `denoise_or_loss_from_kv` / `denoise_step_from_kv`")
        if isinstance(out, tuple):
            return out[0].float(), out[1]
        return out.float(), out

    def forward_joint(
        self,
        front_vis_tokens: torch.Tensor,
        wrist_vis_tokens: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_mask: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        mem_seq: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_tok = self.vis_in_proj(front_vis_tokens)
        w_tok = self.vis_in_proj(wrist_vis_tokens)
        t_emb = self._embed_lang_fallback(lang_tokens)
        lm_f = lang_mask.to(t_emb.dtype).unsqueeze(-1)
        prompt_pool = (t_emb * lm_f).sum(dim=1) / lm_f.sum(dim=1).clamp(min=1.0)

        if hasattr(self.bridge, "forward_joint_from_tokens"):
            res = self.bridge.forward_joint_from_tokens(
                f_tok, w_tok, lang_tokens, lang_mask, noisy_actions, timestep, mem_tokens=mem_seq, mem_mask=mem_mask
            )
        elif hasattr(self.bridge, "forward_joint"):
            res = self.bridge.forward_joint(
                front_tokens=f_tok,
                wrist_tokens=w_tok,
                tokenized_prompt=lang_tokens,
                tokenized_prompt_mask=lang_mask,
                noisy_actions=noisy_actions,
                timestep=timestep,
                mem_tokens=mem_seq,
                mem_mask=mem_mask,
            )
        else:
            pf = self.prefill_prefix_kv(front_vis_tokens, wrist_vis_tokens, lang_tokens, lang_mask)
            vp, hs = self.denoise_from_kv(pf, noisy_actions, timestep, mem_seq=mem_seq, mem_mask=mem_mask)
            return vp, hs, pf.get("prompt_pool", prompt_pool)

        if isinstance(res, tuple):
            return res[0].float(), res[1] if len(res) > 1 else res[0], prompt_pool
        return res.float(), res, prompt_pool

    def enable_vlm_lora(self, rank: int = 32, alpha: float = 32.0, target_ffn: bool = True) -> int:
        if hasattr(self.bridge, "enable_vlm_lora"):
            return int(self.bridge.enable_vlm_lora(rank=rank, alpha=alpha, target_ffn=target_ffn))
        return inject_lora_into_submodule(
            self.bridge,
            rank=rank,
            alpha=alpha,
            exclude_keywords=("expert", "_1", "mem_", "action_", "time_mlp", "siglip", "img"),
        )
