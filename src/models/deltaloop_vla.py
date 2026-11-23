"""Unified Visuomotor Flow-Matching Policy Suite (`DeltaLoop-VLA` & Controlled Baselines).

Implements Cycle 2A of `RESEARCH_PLAN.md` (v2.1):
- **100% Identical Shared Observation & Action Expert Backbone** across all 7+1 models:
  * Current context fusion: `front_emb` (`8x8=64` + 2D SinCos pos emb), `wrist_emb` (`4x4=16` +
    2D SinCos pos emb), `state_norm` (`8` -> `d`), and masked `text_token_emb` (`64` tokens)
    fused via a shared 2-layer Multimodal Context Encoder into `E_t` `(B, 145, d)`.
  * 6-layer Flow-Matching Action Expert (`H_act = 20` action tokens, 10 Euler steps at inference)
    with **dual zero-init memory injection** (`RESEARCH_PLAN.md` §3.4):
    1. Zero-init additive `adaRMSNorm` modulation (`W_mod = 0`), AND
    2. Flamingo-style zero-init `tanh(gamma_ctx)` GQA/MHA Cross-Attention (`gamma_ctx = 0`).
    Guarantees exact mathematical equivalence to `m0_memoryless` at training step 0 while avoiding
    any spatial pooling bottleneck once `gamma_ctx` opens!

- **Controlled Architecture Suite (`ArchVLA`):**
  * `m0_memoryless`: Markov Flow-Matching baseline (no history memory, `L=1` / no loop).
  * `m1_rdvla_loop4`: Memoryless Weight-Tied Looped Core (`L=4`, no GDN memory — `RD-VLA` control).
  * `m2_framesamp_modul`: Perceptual Frame-Sampling buffer (`32` strided/uniform past frame slots +
    Modulator/Cross-Attention — `FrameSamp+Modul` equivalent).
  * `m3_gdn_b1_l1`: Standard Single-Pass GDN (`beta in (0, 1)`, `L=1`).
  * `m4_gdn_b2_l1`: Negative-Eigenvalue Householder Single-Pass GDN (`beta in (0, 2)`, `L=1`).
  * `m5_gdn2layer_b2_l1`: 2-Layer Stacked Neg-Eig GDN (`beta in (0, 2)`, `L=1`, strictly iso-state!).
  * `m6_deltaloop_b2`: **DeltaLoop-VLA (Ours: `beta in (0, 2)` GDN + Weight-Tied Dynamic Multi-Hop
    Readout `L=1..4` + Stage-2 CRN `M_CRN=8` Adaptive Halting)**.
  * `m6b_deltaloop_wb`: **DeltaLoop-VLA + Chunk Boundary Write-Back (`c_wb`, `g=0.0`)** ablation.

- **Strict Mathematical Invariants Enforced:**
  * **A1 (Frame-Wise `fp32` Log-Decay):** Perceiver Slot Resampler (`N_w = 4` slots/written frame,
    `write_stride = 4` frames = 2.5 Hz) with slot-wise temporal `CausalConv1D(w=4)` along the frame
    axis (`C1`) and `fp32` log-decay `g` active ONLY on slot `s=0` (`g[:, :, 1:, :] = 0.0`).
  * **A3 (Video-Demo Cutoff Rule):** `cutoff_excl = max(exec_start_idx, max(0, step_idx - delta_excl + 1))`
    (`delta_excl = 16`), preserving 100% of conditioning video demo frames even at `step_idx == exec_start_idx`
    while excluding near-duplicate recent execution frames.
  * **Dual Backend Parity (`< 1e-5`):** Uses Hopper Triton `fla` `chunk_gated_delta_rule` (`cu_seqlens`
    packed or batched) on Miyabi GH200 when available, with exact `fp32`-accumulating pure-PyTorch
    fallback for seamless evaluation on local RTX 5080 / CPU without Triton compilation.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from memory.gated_deltanet import chunk_gated_delta_rule as pytorch_chunk_gdn
    from memory.gated_deltanet import l2_normalize, recurrent_gated_delta_rule
except ImportError:
    from src.memory.gated_deltanet import chunk_gated_delta_rule as pytorch_chunk_gdn
    from src.memory.gated_deltanet import l2_normalize, recurrent_gated_delta_rule

try:
    from analysis.subgoal_grounding import (
        apply_subgrid_affine_to_tokens,
        sample_subgrid_crop_affine,
        spatial_softmax_2d,
    )
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from analysis.subgoal_grounding import (
        apply_subgrid_affine_to_tokens,
        sample_subgrid_crop_affine,
        spatial_softmax_2d,
    )

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gdn

    HAS_FLA = True
except Exception:
    fla_chunk_gdn = None
    HAS_FLA = False


ArchVLA = Literal[
    "m0_memoryless",
    "m1_rdvla_loop4",
    "m2_framesamp_modul",
    "m3_gdn_b1_l1",
    "m4_gdn_b2_l1",
    "m5_gdn2layer_b2_l1",
    "m6_deltaloop_b2",
    "m6b_deltaloop_wb",
    "m7_deltaloop_2l_b2",
]

ALL_VLA_ARCHS: Tuple[ArchVLA, ...] = (
    "m0_memoryless",
    "m1_rdvla_loop4",
    "m2_framesamp_modul",
    "m3_gdn_b1_l1",
    "m4_gdn_b2_l1",
    "m5_gdn2layer_b2_l1",
    "m6_deltaloop_b2",
    "m6b_deltaloop_wb",
    "m7_deltaloop_2l_b2",
)


@dataclass
class DeltaLoopVLAConfig:
    """Complete serializable configuration for `DeltaLoopVLAPolicy`."""

    arch: ArchVLA = "m6_deltaloop_b2"
    vis_dim: int = 768
    state_dim: int = 8
    action_dim: int = 8
    action_horizon: int = 20
    exec_horizon: int = 16
    exclusion_window: int = 16
    write_stride: int = 4

    # Shared Context & Action Expert dimensions
    d_model: int = 384
    attn_heads: int = 6
    d_ff: int = 1024
    ctx_layers: int = 2
    expert_layers: int = 6

    # Perceiver Slot Resampler & Coda dimensions
    num_write_slots: int = 4      # N_w slots per written frame
    num_coda_tokens: int = 4      # N_c Coda / latent query slots
    slot_conv_kernel: int = 4     # C1: per-slot temporal CausalConv1D width along frame axis
    framesamp_buffer_size: int = 32

    # Gated DeltaNet Memory dimensions (H * d_v * d_k = 8 * 64 * 64 = 32,768 fp32 elements = 128 KB)
    gdn_heads: int = 8
    gdn_d_k: int = 64
    gdn_d_v: int = 64
    min_halflife_frames: float = 8.0
    max_halflife_frames: float = 512.0
    readout_eps: float = 0.1      # C3: Damped RMSNorm constant

    # Loop & Halting dimensions
    max_loops: int = 4
    wb_trunc_chunks: int = 4      # B6: detach non-linear write-back grad every K_wb=4 chunks
    aux_dim: int = 49             # 1 (progress) + 8 (anchor state) + 8 (delta state) + 32 (visual anchor)

    # Cycle 3 (v2.3 §3.7 U1–U5) Shared Upgrades — ALL default to False / Cycle-2A-exact for strict=True compat!
    v3_mode: bool = False
    obs_horizon2: bool = False
    use_spatial_softmax: bool = False
    spatial_kp_front: int = 32
    spatial_kp_wrist: int = 16
    subgrid_aug: bool = False
    subgrid_scale_min: float = 0.90
    subgrid_max_shift: float = 0.04
    patch_dropout: float = 0.10
    feat_noise_std: float = 0.02
    use_gripper_head: bool = False
    gripper_bce_weight: float = 0.05
    use_grounding_head: bool = False
    subgoal_dim: int = 64
    grounding_loss_weight: float = 0.15
    adaptive_demo_stride_threshold: int = 256
    long_demo_stride: int = 8
    use_motion_beta_gate: bool = False
    v3_beta_bias: float = -2.5
    sample_clamp: float = 1.5
    ode_solver: str = "euler"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeltaLoopVLAConfig":
        valid = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def v3_preset(cls, arch: ArchVLA = "m6_deltaloop_b2", **overrides: Any) -> "DeltaLoopVLAConfig":
        """Canonical Cycle 3 (`RESEARCH_PLAN.md` v2.3 §3.7 `U1–U5`) configuration preset."""
        defaults: Dict[str, Any] = {
            "arch": arch,
            "v3_mode": True,
            "num_write_slots": 8,
            "gdn_heads": 8,
            "gdn_d_k": 128,
            "gdn_d_v": 128,
            "obs_horizon2": True,
            "use_spatial_softmax": True,
            "spatial_kp_front": 32,
            "spatial_kp_wrist": 16,
            "subgrid_aug": True,
            "subgrid_scale_min": 0.90,
            "subgrid_max_shift": 0.04,
            "patch_dropout": 0.10,
            "feat_noise_std": 0.02,
            "use_gripper_head": True,
            "gripper_bce_weight": 0.05,
            "use_grounding_head": True,
            "subgoal_dim": 64,
            "grounding_loss_weight": 0.15,
            "adaptive_demo_stride_threshold": 256,
            "long_demo_stride": 8,
            "use_motion_beta_gate": True,
            "v3_beta_bias": -2.5,
            "sample_clamp": 2.5,
            "ode_solver": "midpoint",
        }
        defaults.update(overrides)
        return cls(**defaults)


# =============================================================================
# Positional Embeddings & Core Building Blocks
# =============================================================================


def build_2d_sincos_pos_embed(grid_h: int, grid_w: int, embed_dim: int) -> torch.Tensor:
    """Create deterministic 2D sinusoidal positional embeddings `(grid_h * grid_w, embed_dim)`."""
    if embed_dim % 4 != 0:
        raise ValueError(f"embed_dim must be divisible by 4 for 2D SinCos, got {embed_dim}")
    quarter = embed_dim // 4
    omega = torch.exp(
        -math.log(10000.0) * torch.arange(quarter, dtype=torch.float32) / max(quarter - 1, 1)
    )
    ys = torch.arange(grid_h, dtype=torch.float32)
    xs = torch.arange(grid_w, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    yy = yy.reshape(-1, 1) * omega.unsqueeze(0)
    xx = xx.reshape(-1, 1) * omega.unsqueeze(0)
    pe = torch.cat([torch.sin(yy), torch.cos(yy), torch.sin(xx), torch.cos(xx)], dim=-1)
    return pe  # (grid_h * grid_w, embed_dim)


def build_1d_sincos_embed(positions: torch.Tensor, embed_dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Embed continuous or integer scalar positions `(...)` into `(..., embed_dim)` in fp32."""
    half = embed_dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=positions.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = positions.float().unsqueeze(-1) * freqs
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embed_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class SwiGLU(nn.Module):
    """Pre-norm friendly SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


def safe_masked_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multi-head SDPA `(B, H, L_q, D)` -> `(B, H, L_q, D)` safe against all-False mask rows!

    In PyTorch, passing an all-False row in `attn_mask` to `scaled_dot_product_attention` yields
    `NaN` (`0.0 * NaN == NaN`, breaking even zero-gated residual branches!). Here, any batch item
    whose `key_valid_mask` has zero True entries cleanly returns exact `0.0`.
    """
    if key_valid_mask is None:
        return F.scaled_dot_product_attention(q, k, v)

    # key_valid_mask: (B, L_kv) bool
    has_any = key_valid_mask.any(dim=-1)  # (B,)
    safe_mask = key_valid_mask.clone()
    safe_mask[~has_any, 0] = True
    attn_mask = safe_mask[:, None, None, :]  # (B, 1, 1, L_kv)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return out * has_any.to(out.dtype)[:, None, None, None]


class MultiHeadAttn(nn.Module):
    """Clean Pre-RMSNorm Multi-Head Self/Cross Attention with QK-RMSNorm & key-validity masks."""

    def __init__(self, d_model: int, num_heads: int, zero_init_out: bool = False) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        if zero_init_out:
            nn.init.zeros_(self.o_proj.weight)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        kv_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Lq, D = x_q.shape
        Lkv = x_kv.shape[1]
        q = self.q_norm(self.q_proj(x_q).view(B, Lq, self.num_heads, self.head_dim)).transpose(1, 2)
        k, v = self.kv_proj(x_kv).chunk(2, dim=-1)
        k = self.k_norm(k.view(B, Lkv, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, Lkv, self.num_heads, self.head_dim).transpose(1, 2)
        out = safe_masked_sdpa(q, k, v, key_valid_mask=kv_valid_mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, Lq, D))


# =============================================================================
# 1. Shared Observation Context Encoder (100% Identical Across All Models)
# =============================================================================


class ContextTransformerBlock(nn.Module):
    """Bidirectional multimodal context block with padding-aware attention."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttn(d_model, num_heads)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, kv_valid_mask=valid_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SpatialKeypointBranch(nn.Module):
    """Cycle 3 `U3`: Continuous 2D `SpatialSoftmax` keypoint bottleneck (`Front K=32`, `Wrist K=16`)."""

    def __init__(self, vis_dim: int, kp_front: int, kp_wrist: int, d_model: int) -> None:
        super().__init__()
        self.kp_front = int(kp_front)
        self.kp_wrist = int(kp_wrist)
        self.front_kp = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, self.kp_front))
        self.wrist_kp = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, self.kp_wrist))
        kp_in_dim = 2 * (self.kp_front + self.kp_wrist)
        self.kp_mlp = nn.Sequential(
            nn.Linear(kp_in_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, front_emb: torch.Tensor, wrist_emb: torch.Tensor) -> torch.Tensor:
        B, Nf, _ = front_emb.shape
        Nw = wrist_emb.shape[1]
        dt = self.front_kp[1].weight.dtype  # type: ignore[index]
        side_f = int(round(Nf**0.5))
        side_w = int(round(Nw**0.5))

        f_logits = self.front_kp(front_emb.to(dt)).transpose(1, 2)  # (B, K_f, Nf)
        w_logits = self.wrist_kp(wrist_emb.to(dt)).transpose(1, 2)  # (B, K_w, Nw)
        with torch.autocast(device_type=front_emb.device.type, enabled=False):
            xy_f, _ = spatial_softmax_2d(f_logits.float(), grid_hw=(side_f, side_f), align_corners=False)
            xy_w, _ = spatial_softmax_2d(w_logits.float(), grid_hw=(side_w, side_w), align_corners=False)
        kp_flat = torch.cat([xy_f.reshape(B, -1), xy_w.reshape(B, -1)], dim=-1).to(dt)
        return self.kp_mlp(kp_flat).unsqueeze(1)  # (B, 1, d_model)


class SharedObservationEncoder(nn.Module):
    """Fuses `front_emb (64)`, `wrist_emb (16)`, `state_norm (1)`, and `text_token_emb (64)` into `E_t`."""

    def __init__(self, cfg: DeltaLoopVLAConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.obs_horizon2 = bool(cfg.obs_horizon2)
        self.subgrid_aug = bool(cfg.subgrid_aug)
        self.subgrid_scale_min = float(cfg.subgrid_scale_min)
        self.subgrid_max_shift = float(cfg.subgrid_max_shift)
        self.patch_dropout = float(cfg.patch_dropout)
        self.feat_noise_std = float(cfg.feat_noise_std)

        self.front_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        self.wrist_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        self.text_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        state_in_dim = 3 * cfg.state_dim if self.obs_horizon2 else cfg.state_dim
        self.state_proj = nn.Sequential(
            nn.Linear(state_in_dim, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        self.front_vel_proj: Optional[nn.Linear] = None
        self.wrist_vel_proj: Optional[nn.Linear] = None
        if self.obs_horizon2:
            self.front_vel_proj = nn.Linear(cfg.vis_dim, d, bias=False)
            self.wrist_vel_proj = nn.Linear(cfg.vis_dim, d, bias=False)
            nn.init.normal_(self.front_vel_proj.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.wrist_vel_proj.weight, mean=0.0, std=0.01)

        self.kp_branch: Optional[SpatialKeypointBranch] = None
        self.kp_mod_emb: Optional[nn.Parameter] = None
        if cfg.use_spatial_softmax:
            self.kp_branch = SpatialKeypointBranch(
                cfg.vis_dim, cfg.spatial_kp_front, cfg.spatial_kp_wrist, d
            )
            self.kp_mod_emb = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.register_buffer("front_pos", build_2d_sincos_pos_embed(8, 8, d), persistent=False)
        self.register_buffer("wrist_pos", build_2d_sincos_pos_embed(4, 4, d), persistent=False)
        self.register_buffer(
            "text_pos",
            build_1d_sincos_embed(torch.arange(64, dtype=torch.float32), d),
            persistent=False,
        )
        # 4 modality embeddings: 0=front, 1=wrist, 2=state, 3=text
        self.mod_emb = nn.Embedding(4, d)
        nn.init.normal_(self.mod_emb.weight, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            [ContextTransformerBlock(d, cfg.attn_heads, cfg.d_ff) for _ in range(cfg.ctx_layers)]
        )
        self.out_norm = nn.RMSNorm(d)

    def forward(
        self,
        front_emb: torch.Tensor,       # (B, 64, 768)
        wrist_emb: torch.Tensor,       # (B, 16, 768)
        state_norm: torch.Tensor,      # (B, 8)
        text_token_emb: torch.Tensor,  # (B, 64, 768)
        text_attn_mask: torch.Tensor,  # (B, 64) bool
        *,
        prev_front_emb: Optional[torch.Tensor] = None,
        prev_wrist_emb: Optional[torch.Tensor] = None,
        prev_state_norm: Optional[torch.Tensor] = None,
        aug_theta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = front_emb.shape[0]
        dt = self.front_proj[1].weight.dtype  # type: ignore[index]
        me = self.mod_emb.weight.to(dt)

        if (not self.obs_horizon2) and (self.kp_branch is None) and (not self.subgrid_aug) and (aug_theta is None):
            f_tok = self.front_proj(front_emb.to(dt)) + self.front_pos.to(dt).unsqueeze(0) + me[0].view(1, 1, -1)
            w_tok = self.wrist_proj(wrist_emb.to(dt)) + self.wrist_pos.to(dt).unsqueeze(0) + me[1].view(1, 1, -1)
            s_tok = self.state_proj(state_norm.to(dt)).unsqueeze(1) + me[2].view(1, 1, -1)
            t_tok = self.text_proj(text_token_emb.to(dt)) + self.text_pos.to(dt).unsqueeze(0) + me[3].view(1, 1, -1)

            tokens = torch.cat([f_tok, w_tok, s_tok, t_tok], dim=1)  # (B, 145, d)
            obs_valid = torch.ones((B, 81), device=tokens.device, dtype=torch.bool)
            ctx_mask = torch.cat([obs_valid, text_attn_mask.bool()], dim=1)  # (B, 145)

            for blk in self.blocks:
                tokens = blk(tokens, ctx_mask)
            return self.out_norm(tokens), ctx_mask

        f_in = front_emb
        w_in = wrist_emb
        pf_in = prev_front_emb if prev_front_emb is not None else front_emb
        pw_in = prev_wrist_emb if prev_wrist_emb is not None else wrist_emb
        ps_in = prev_state_norm if prev_state_norm is not None else state_norm

        eff_theta = aug_theta
        if eff_theta is None and self.training and self.subgrid_aug:
            eff_theta = sample_subgrid_crop_affine(
                B,
                scale_range=(self.subgrid_scale_min, 1.0),
                max_extra_shift=self.subgrid_max_shift,
                device=front_emb.device,
            )

        if eff_theta is not None:
            side_f = int(round(f_in.shape[1] ** 0.5))
            with torch.autocast(device_type=front_emb.device.type, enabled=False):
                if pf_in is f_in:
                    f_in = apply_subgrid_affine_to_tokens(f_in, eff_theta, grid_hw=(side_f, side_f))
                    pf_in = f_in
                else:
                    f_in = apply_subgrid_affine_to_tokens(f_in, eff_theta, grid_hw=(side_f, side_f))
                    pf_in = apply_subgrid_affine_to_tokens(pf_in, eff_theta, grid_hw=(side_f, side_f))

        if self.training and self.subgrid_aug and self.feat_noise_std > 0.0:
            f_scale = f_in.detach().float().std(dim=-1, keepdim=True).clamp(min=1e-3).to(f_in.dtype)
            w_scale = w_in.detach().float().std(dim=-1, keepdim=True).clamp(min=1e-3).to(w_in.dtype)
            f_in = f_in + torch.randn_like(f_in) * (self.feat_noise_std * f_scale)
            w_in = w_in + torch.randn_like(w_in) * (self.feat_noise_std * w_scale)

        f_proj = self.front_proj(f_in.to(dt))
        if self.training and self.subgrid_aug and self.patch_dropout > 0.0:
            keep_mask = F.dropout(
                torch.ones((B, f_proj.shape[1], 1), device=f_proj.device, dtype=dt),
                p=self.patch_dropout,
                training=True,
            )
            f_proj = f_proj * keep_mask

        f_tok = f_proj + self.front_pos.to(dt).unsqueeze(0) + me[0].view(1, 1, -1)
        w_tok = self.wrist_proj(w_in.to(dt)) + self.wrist_pos.to(dt).unsqueeze(0) + me[1].view(1, 1, -1)

        if self.obs_horizon2:
            assert self.front_vel_proj is not None and self.wrist_vel_proj is not None
            df = (f_in.float() - pf_in.float()).to(dt)
            dw = (w_in.float() - pw_in.float()).to(dt)
            f_tok = f_tok + self.front_vel_proj(df)
            w_tok = w_tok + self.wrist_vel_proj(dw)

            ds_scaled = ((state_norm.float() - ps_in.float()) * 5.0).clamp(-3.0, 3.0)
            st_cat = torch.cat([ps_in.float(), state_norm.float(), ds_scaled], dim=-1).to(dt)
            s_tok = self.state_proj(st_cat).unsqueeze(1) + me[2].view(1, 1, -1)
        else:
            s_tok = self.state_proj(state_norm.to(dt)).unsqueeze(1) + me[2].view(1, 1, -1)

        t_tok = self.text_proj(text_token_emb.to(dt)) + self.text_pos.to(dt).unsqueeze(0) + me[3].view(1, 1, -1)

        tok_parts = [f_tok, w_tok]
        if self.kp_branch is not None and self.kp_mod_emb is not None:
            kp_tok = self.kp_branch(f_in, w_in) + self.kp_mod_emb.to(dt)
            tok_parts.append(kp_tok)
        tok_parts.extend([s_tok, t_tok])

        tokens = torch.cat(tok_parts, dim=1)
        obs_len = tokens.shape[1] - text_attn_mask.shape[1]
        obs_valid = torch.ones((B, obs_len), device=tokens.device, dtype=torch.bool)
        ctx_mask = torch.cat([obs_valid, text_attn_mask.bool()], dim=1)

        for blk in self.blocks:
            tokens = blk(tokens, ctx_mask)
        return self.out_norm(tokens), ctx_mask


# =============================================================================
# 2. Shared Flow-Matching Action Expert (`adaRMSNorm` + Flamingo `tanh(gamma)`)
# =============================================================================


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm modulated by flow-time + zero-init memory condition vector."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, elementwise_affine=False)
        self.mod = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.mod(F.silu(cond)).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ActionExpertBlock(nn.Module):
    """Pi0.5 / MME-VLA-compatible Flow Action Expert Block (`RESEARCH_PLAN.md` §3.4).

    Dual zero-init memory integration:
      1. Additive `adaRMSNorm` condition `psi_tilde = psi(tau) + W_mod * MeanPool(C_t)` (`W_mod = 0`).
      2. Flamingo-style `x <- x + tanh(gamma_ctx) * MemCrossAttn(RMSNorm(x), C_t)` (`gamma_ctx = 0`)
         immediately before `pre_ffw_norm`, preserving full spatial token granularity without
         changing self-attention softmax denominators at step 0!
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, v3_mode: bool = False) -> None:
        super().__init__()
        self.v3_mode = bool(v3_mode)
        self.pre_sa_norm = AdaRMSNorm(d_model)
        self.self_attn = MultiHeadAttn(d_model, num_heads)

        self.pre_ca_norm = nn.RMSNorm(d_model)
        self.ctx_attn = MultiHeadAttn(d_model, num_heads)

        # Memory Path 1: Zero-init additive modulator onto AdaRMSNorm condition
        self.mem_mod_proj = nn.Linear(d_model, d_model, bias=True)
        nn.init.zeros_(self.mem_mod_proj.weight)
        nn.init.zeros_(self.mem_mod_proj.bias)

        # Memory Path 2: Flamingo-style zero-init Cross-Attention
        # In v3_mode, zero_init_out=True + gate = (1 + tanh(gamma_ctx)) preserves exact 0.0 step-0
        # zero-init while giving o_proj full 1.0x outer-product gradients on step 1 (avoiding the
        # scalar gamma=0 * random_W saddle point).
        self.pre_mem_norm = nn.RMSNorm(d_model)
        self.mem_cross_attn = MultiHeadAttn(d_model, num_heads, zero_init_out=self.v3_mode)
        self.gamma_ctx = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.pre_ffw_norm = AdaRMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,                        # (B, H_act, d)
        t_cond: torch.Tensor,                   # (B, d)
        ctx_tokens: torch.Tensor,               # (B, N_ctx, d)
        ctx_mask: torch.Tensor,                 # (B, N_ctx) bool
        mem_tokens: Optional[torch.Tensor] = None,  # (B, N_c, d)
        mem_mask: Optional[torch.Tensor] = None,    # (B, N_c) bool
    ) -> torch.Tensor:
        if mem_tokens is not None:
            if mem_mask is not None:
                m_float = mem_mask.to(mem_tokens.dtype).unsqueeze(-1)
                denom = m_float.sum(dim=1).clamp(min=1.0)
                mem_pool = (mem_tokens * m_float).sum(dim=1) / denom
                has_mem = mem_mask.any(dim=-1, keepdim=True).to(mem_tokens.dtype)
                mem_delta = self.mem_mod_proj(mem_pool) * has_mem
            else:
                mem_pool = mem_tokens.mean(dim=1)
                mem_delta = self.mem_mod_proj(mem_pool)
            cond = t_cond + mem_delta
        else:
            cond = t_cond

        h_sa = self.pre_sa_norm(x, cond)
        x = x + self.self_attn(h_sa, h_sa)

        x = x + self.ctx_attn(self.pre_ca_norm(x), ctx_tokens, kv_valid_mask=ctx_mask)

        if mem_tokens is not None:
            gate = (
                (1.0 + torch.tanh(self.gamma_ctx)).to(x.dtype)
                if self.v3_mode
                else torch.tanh(self.gamma_ctx).to(x.dtype)
            )
            mem_out = self.mem_cross_attn(self.pre_mem_norm(x), mem_tokens, kv_valid_mask=mem_mask)
            x = x + gate * mem_out

        x = x + self.ffn(self.pre_ffw_norm(x, cond))
        return x


class FlowActionExpert(nn.Module):
    """6-Block Flow-Matching Velocity Network `v_theta(x_tau, tau | E_t, C_t)`."""

    def __init__(self, cfg: DeltaLoopVLAConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.action_horizon = cfg.action_horizon
        self.exec_horizon = cfg.exec_horizon
        self.action_dim = cfg.action_dim

        self.act_in = nn.Linear(cfg.action_dim, d)
        self.register_buffer(
            "act_pos",
            build_1d_sincos_embed(torch.arange(cfg.action_horizon, dtype=torch.float32), d),
            persistent=False,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        self.blocks = nn.ModuleList(
            [ActionExpertBlock(d, cfg.attn_heads, cfg.d_ff, v3_mode=cfg.v3_mode) for _ in range(cfg.expert_layers)]
        )

        self.final_mem_mod = nn.Linear(d, d, bias=True)
        nn.init.zeros_(self.final_mem_mod.weight)
        nn.init.zeros_(self.final_mem_mod.bias)
        self.final_norm = AdaRMSNorm(d)
        self.act_out = nn.Linear(d, cfg.action_dim)
        nn.init.normal_(self.act_out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.act_out.bias)

        self.gripper_head: Optional[nn.Linear] = None
        if cfg.use_gripper_head:
            self.gripper_head = nn.Linear(d, 1)
            nn.init.normal_(self.gripper_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.gripper_head.bias)

    def forward(
        self,
        x_tau: torch.Tensor,                        # (B, H_act, 8)
        tau: torch.Tensor,                          # (B,) in [0, 1]
        ctx_tokens: torch.Tensor,                   # (B, N_ctx, d)
        ctx_mask: torch.Tensor,                     # (B, N_ctx) bool
        mem_tokens: Optional[torch.Tensor] = None,  # (B, N_c, d)
        mem_mask: Optional[torch.Tensor] = None,    # (B, N_c) bool
        *,
        return_gripper_logits: bool = False,
    ) -> Any:
        dt = self.act_in.weight.dtype
        x = self.act_in(x_tau.to(dt)) + self.act_pos.to(dt).unsqueeze(0)
        t_emb = build_1d_sincos_embed(tau * 1000.0, self.d_model, max_period=10000.0).to(dt)
        t_cond = self.time_mlp(t_emb)

        mem_t = mem_tokens.to(dt) if mem_tokens is not None else None
        for blk in self.blocks:
            x = blk(x, t_cond, ctx_tokens, ctx_mask, mem_tokens=mem_t, mem_mask=mem_mask)

        if mem_t is not None:
            if mem_mask is not None:
                m_f = mem_mask.to(dt).unsqueeze(-1)
                mp = (mem_t * m_f).sum(dim=1) / m_f.sum(dim=1).clamp(min=1.0)
                has_m = mem_mask.any(dim=-1, keepdim=True).to(dt)
                fin_cond = t_cond + self.final_mem_mod(mp) * has_m
            else:
                fin_cond = t_cond + self.final_mem_mod(mem_t.mean(dim=1))
        else:
            fin_cond = t_cond

        x = self.final_norm(x, fin_cond)
        v_out = self.act_out(x).float()
        if return_gripper_logits:
            grip_logits = (
                self.gripper_head(x[:, : self.exec_horizon, :]).squeeze(-1).float()
                if self.gripper_head is not None
                else None
            )
            return v_out, grip_logits
        return v_out


# =============================================================================
# 3. Perceiver Slot Resampler + Per-Slot Frame-Axis CausalConv1D (`C1`)
# =============================================================================


class PerceiverSlotResampler(nn.Module):
    """Compresses each written history frame `(front 64 + wrist 16 + state 1)` into `N_w` slots.

    Includes:
    - 2D SinCos spatial positional embeddings on `8x8` front and `4x4` wrist tokens
    - `is_demo` conditioning vs execution embedding + frame step time embedding
    - Slot-wise `CausalConv1D(w=4)` strictly along the **frame time axis `m`** independently for
      each slot index `s in 0..N_w-1` (`RESEARCH_PLAN.md` §3.1 `C1`), capturing container swap
      velocities and object pass dynamics without inducing artificial intra-frame slot ordering!
    """

    def __init__(self, cfg: DeltaLoopVLAConfig, num_slots: Optional[int] = None) -> None:
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.num_slots = cfg.num_write_slots if num_slots is None else num_slots
        self.conv_width = cfg.slot_conv_kernel

        self.front_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        self.wrist_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        self.state_proj = nn.Sequential(nn.Linear(cfg.state_dim, d), nn.SiLU(), nn.Linear(d, d))

        self.kp_branch: Optional[SpatialKeypointBranch] = None
        self.kp_view_emb: Optional[nn.Parameter] = None
        if cfg.use_spatial_softmax:
            self.kp_branch = SpatialKeypointBranch(
                cfg.vis_dim, cfg.spatial_kp_front, cfg.spatial_kp_wrist, d
            )
            self.kp_view_emb = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        self.register_buffer("front_pos", build_2d_sincos_pos_embed(8, 8, d), persistent=False)
        self.register_buffer("wrist_pos", build_2d_sincos_pos_embed(4, 4, d), persistent=False)
        self.view_emb = nn.Embedding(3, d)  # 0=front, 1=wrist, 2=state
        self.demo_emb = nn.Embedding(2, d)  # 0=execution frame, 1=conditioning video demo frame
        self.time_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.view_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.demo_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.time_proj.weight, mean=0.0, std=0.02)

        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, d) * 0.02)
        self.q_norm = nn.RMSNorm(d)
        self.kv_norm = nn.RMSNorm(d)
        self.cross_attn = MultiHeadAttn(d, cfg.attn_heads)
        self.ffn_norm = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d * 2)

        # Depthwise causal 1D conv along frame axis per slot (initialized as identity on current frame!)
        if self.conv_width > 1:
            self.temporal_conv = nn.Conv1d(
                d,
                d,
                kernel_size=self.conv_width,
                groups=d,
                bias=True,
            )
            nn.init.zeros_(self.temporal_conv.weight)
            # Last kernel tap corresponds to current frame m -> identity initialization
            self.temporal_conv.weight.data[:, 0, -1] = 1.0
            nn.init.zeros_(self.temporal_conv.bias)
        else:
            self.temporal_conv = None

        self.out_norm = nn.RMSNorm(d)

    def encode_frames_raw(
        self,
        front_emb: torch.Tensor,   # (N, 64, 768)
        wrist_emb: torch.Tensor,   # (N, 16, 768)
        state_norm: torch.Tensor,  # (N, 8)
        step_idx: torch.Tensor,    # (N,) int64/float
        is_demo: torch.Tensor,     # (N,) bool/long
    ) -> torch.Tensor:
        """Encode `N` frames independently into `(N, num_slots, d_model)` before frame-axis conv."""
        N = front_emb.shape[0]
        if N == 0:
            return torch.empty(
                (0, self.num_slots, self.d_model),
                device=front_emb.device,
                dtype=self.slot_queries.dtype,
            )
        dt = self.front_proj[1].weight.dtype  # type: ignore[index]
        ve = self.view_emb.weight.to(dt)

        f_tok = self.front_proj(front_emb.to(dt)) + self.front_pos.to(dt).unsqueeze(0) + ve[0].view(1, 1, -1)
        w_tok = self.wrist_proj(wrist_emb.to(dt)) + self.wrist_pos.to(dt).unsqueeze(0) + ve[1].view(1, 1, -1)
        s_tok = self.state_proj(state_norm.to(dt)).unsqueeze(1) + ve[2].view(1, 1, -1)
        if self.kp_branch is not None and self.kp_view_emb is not None:
            kp_tok = self.kp_branch(front_emb, wrist_emb) + self.kp_view_emb.to(dt)
            kv = self.kv_norm(torch.cat([f_tok, w_tok, kp_tok, s_tok], dim=1))  # (N, 82, d)
        else:
            kv = self.kv_norm(torch.cat([f_tok, w_tok, s_tok], dim=1))  # (N, 81, d)

        t_pe = self.time_proj(build_1d_sincos_embed(step_idx.float() / 4.0, self.d_model).to(dt))
        d_pe = self.demo_emb(is_demo.long().clamp(0, 1)).to(dt)
        frame_bias = (t_pe + d_pe).unsqueeze(1)  # (N, 1, d)

        q = self.q_norm(self.slot_queries.to(dt).unsqueeze(0).expand(N, -1, -1) + frame_bias)
        slots = q + self.cross_attn(q, kv)
        slots = slots + self.ffn(self.ffn_norm(slots))
        return slots  # (N, num_slots, d)

    def apply_episode_slot_conv(self, raw_slots: torch.Tensor) -> torch.Tensor:
        """Apply strictly causal `CausalConv1D` along frame axis `W` for `(W, N_w, d)` or `(B, W, N_w, d)`."""
        squeeze_b = False
        if raw_slots.ndim == 3:
            raw_slots = raw_slots.unsqueeze(0)
            squeeze_b = True
        B, W, Nw, D = raw_slots.shape
        if W == 0:
            out = raw_slots
            return out.squeeze(0) if squeeze_b else out

        dt = self.slot_queries.dtype
        x = raw_slots.to(dt)
        if self.temporal_conv is not None and self.conv_width > 1:
            # Reshape so each (b, slot s) is an independent 1D time series of length W!
            x_bs = x.permute(0, 2, 3, 1).reshape(B * Nw, D, W)  # (B * Nw, D, W)
            x_pad = F.pad(x_bs, (self.conv_width - 1, 0))
            y_bs = F.silu(self.temporal_conv(x_pad))
            x = y_bs.reshape(B, Nw, D, W).permute(0, 3, 1, 2).contiguous()

        out = self.out_norm(x)
        return out.squeeze(0) if squeeze_b else out


# =============================================================================
# 4. Dual-Backend (`fla` Hopper Triton + Exact Pure-PyTorch) GDN Layer
# =============================================================================


def l2_normalize_fp32(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize tensor along `dim` and KEEP in `torch.float32` (exact Householder unit norm!)."""
    x_f32 = x.float()
    norm = torch.sqrt(torch.sum(x_f32 * x_f32, dim=dim, keepdim=True) + eps * eps)
    return x_f32 / norm


def run_dual_backend_gdn(
    k: torch.Tensor,                              # (1, Total_T, H, K) or (B, T, H, K)
    v: torch.Tensor,                              # (1, Total_T, H, V) or (B, T, H, V)
    g_fp32: torch.Tensor,                         # (..., H) float32 <= 0
    beta_fp32: torch.Tensor,                      # (..., H) float32 in [0, 2]
    q: Optional[torch.Tensor] = None,             # Optional (..., H, K) for seq readout
    initial_state_fp32: Optional[torch.Tensor] = None,  # (N_seq, H, V, K) float32
    initial_occ_fp32: Optional[torch.Tensor] = None,    # (N_seq, H, K) float32
    cu_seqlens: Optional[torch.Tensor] = None,    # (N_seq + 1,) int32/int64 when B == 1
    track_occ: bool = True,
    force_pytorch: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """Wrapper disabling `torch.autocast` so GDN state updates execute strictly in `float32`."""
    with torch.autocast(device_type=k.device.type, enabled=False):
        return _run_dual_backend_gdn_impl(
            k=k,
            v=v,
            g_fp32=g_fp32,
            beta_fp32=beta_fp32,
            q=q,
            initial_state_fp32=initial_state_fp32,
            initial_occ_fp32=initial_occ_fp32,
            cu_seqlens=cu_seqlens,
            track_occ=track_occ,
            force_pytorch=force_pytorch,
        )


def _run_dual_backend_gdn_impl(
    k: torch.Tensor,                              # (1, Total_T, H, K) or (B, T, H, K)
    v: torch.Tensor,                              # (1, Total_T, H, V) or (B, T, H, V)
    g_fp32: torch.Tensor,                         # (..., H) float32 <= 0
    beta_fp32: torch.Tensor,                      # (..., H) float32 in [0, 2]
    q: Optional[torch.Tensor] = None,             # Optional (..., H, K) for seq readout
    initial_state_fp32: Optional[torch.Tensor] = None,  # (N_seq, H, V, K) float32
    initial_occ_fp32: Optional[torch.Tensor] = None,    # (N_seq, H, K) float32
    cu_seqlens: Optional[torch.Tensor] = None,    # (N_seq + 1,) int32/int64 when B == 1
    track_occ: bool = True,
    force_pytorch: bool = False,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
    """Execute Gated Delta Rule returning `(seq_out_curr, S_final_fp32, s_occ_final_fp32)`.

    Canonical state layout returned is ALWAYS `V-first`:
      - `S_final_fp32`: `(N_seq, H, V, K)` in `float32`
      - `s_occ_final_fp32`: `(N_seq, H, K)` in `float32`
    Executes in `float32` on both `fla.ops.gated_delta_rule.chunk_gated_delta_rule` (padding `V`
    to a multiple of 32) and the pure-PyTorch WY/recurrent fallback so negative-eigenvalue
    Householder reflections (`beta in (0, 2)`) remain strictly orthogonal and cross-hardware parity
    between Miyabi GH200 (`fla`) and local RTX 5080 / CPU (`PyTorch`) holds to `< 1e-5`!
    """
    orig_dtype = v.dtype
    k_f32 = k.float().contiguous()
    v_f32 = v.float().contiguous()
    q_f32 = q.float().contiguous() if q is not None else None
    use_fla = HAS_FLA and k.is_cuda and (not force_pytorch)

    if use_fla:
        B_in, T_in, H, K = k_f32.shape
        V = v_f32.shape[-1]
        N_seq = (cu_seqlens.numel() - 1) if cu_seqlens is not None else B_in
        if T_in == 0 or N_seq == 0:
            S0 = (
                initial_state_fp32.float()
                if initial_state_fp32 is not None
                else torch.zeros((N_seq, H, V, K), device=k.device, dtype=torch.float32)
            )
            occ0 = (
                initial_occ_fp32.float()
                if initial_occ_fp32 is not None
                else (torch.zeros((N_seq, H, K), device=k.device, dtype=torch.float32) if track_occ else None)
            )
            o_empty = torch.zeros_like(v) if q is not None else None
            return o_empty, S0, occ0

        V_need = V + (1 if track_occ else 0)
        V_pad = ((V_need + 31) // 32) * 32
        pad_extra = V_pad - V_need

        parts = [v_f32]
        if track_occ:
            parts.append(torch.ones((B_in, T_in, H, 1), device=v.device, dtype=torch.float32))
        if pad_extra > 0:
            parts.append(torch.zeros((B_in, T_in, H, pad_extra), device=v.device, dtype=torch.float32))
        v_aug = torch.cat(parts, dim=-1).contiguous()

        q_in = q_f32 if q_f32 is not None else torch.zeros_like(k_f32)

        h0_fla = None
        if initial_state_fp32 is not None or (track_occ and initial_occ_fp32 is not None):
            s_init = (
                initial_state_fp32.float()
                if initial_state_fp32 is not None
                else torch.zeros((N_seq, H, V, K), device=k.device, dtype=torch.float32)
            )
            s_parts = [s_init]
            if track_occ:
                occ_init = (
                    initial_occ_fp32.float()
                    if initial_occ_fp32 is not None
                    else torch.zeros((N_seq, H, K), device=k.device, dtype=torch.float32)
                )
                s_parts.append(occ_init.unsqueeze(-2))
            if pad_extra > 0:
                s_parts.append(torch.zeros((N_seq, H, pad_extra, K), device=k.device, dtype=torch.float32))
            s_aug_v_first = torch.cat(s_parts, dim=-2)  # (N_seq, H, V_pad, K)
            # fla default `state_v_first=False` expects `[N_seq, H, K, V_pad]`
            h0_fla = s_aug_v_first.transpose(-1, -2).contiguous()

        cu_i32 = cu_seqlens.to(dtype=torch.int32).contiguous() if cu_seqlens is not None else None
        if K > 64 and V_pad > 64:
            # Hopper dynamic SMEM (`228 KB`) cap in `chunk_bwd_kernel_dqkwg` for `fp32` `K=128`:
            # Each `V`-row of `S_t` evolves independently under `(k, g, beta)`, so `V_chunk <= 64`
            # (multiples of 32) runs natively in `fla` with `< 2e-6` parity vs IEEE fp32 PyTorch.
            o_chunks: List[torch.Tensor] = []
            h_chunks: List[torch.Tensor] = []
            g_c = g_fp32.float().contiguous()
            b_c = beta_fp32.float().contiguous()
            for v0 in range(0, V_pad, 64):
                v1 = min(v0 + 64, V_pad)
                v_sl = v_aug[..., v0:v1].contiguous()
                h0_sl = h0_fla[..., v0:v1].contiguous() if h0_fla is not None else None
                o_sl, h_sl = fla_chunk_gdn(
                    q=q_in,
                    k=k_f32,
                    v=v_sl,
                    g=g_c,
                    beta=b_c,
                    scale=1.0,
                    initial_state=h0_sl,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                    use_beta_sigmoid_in_kernel=False,
                    allow_neg_eigval=False,
                    state_v_first=False,
                    cu_seqlens=cu_i32,
                    chunk_size=32,
                )
                o_chunks.append(o_sl)
                h_chunks.append(h_sl)
            o_aug = torch.cat(o_chunks, dim=-1)
            h_final_kv = torch.cat(h_chunks, dim=-1)
        else:
            o_aug, h_final_kv = fla_chunk_gdn(
                q=q_in,
                k=k_f32,
                v=v_aug,
                g=g_fp32.float().contiguous(),
                beta=beta_fp32.float().contiguous(),
                scale=1.0,
                initial_state=h0_fla,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
                use_beta_sigmoid_in_kernel=False,
                allow_neg_eigval=False,
                state_v_first=False,
                cu_seqlens=cu_i32,
                chunk_size=32,
            )
        # Convert `[N_seq, H, K, V_pad]` -> `[N_seq, H, V_pad, K]` in fp32
        s_all = h_final_kv.float().transpose(-1, -2).contiguous()
        S_final = s_all[:, :, :V, :]
        occ_final = s_all[:, :, V, :] if track_occ else None
        seq_out = o_aug[..., :V].to(orig_dtype) if q is not None else None
        return seq_out, S_final, occ_final

    # Pure-PyTorch exact IEEE fp32 fallback (handles both batched `(B, T, H, K)` and packed `cu_seqlens`)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if k.is_cuda else False
    if k.is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if cu_seqlens is not None:
            assert k_f32.shape[0] == 1, "Packed cu_seqlens expects batch dim 1"
            N_seq = cu_seqlens.numel() - 1
            H, K = k_f32.shape[2], k_f32.shape[3]
            V = v_f32.shape[-1]
            if N_seq == 0 or k_f32.shape[1] == 0:
                S0 = (
                    initial_state_fp32.float()
                    if initial_state_fp32 is not None
                    else torch.zeros((N_seq, H, V, K), device=k.device, dtype=torch.float32)
                )
                occ0 = (
                    initial_occ_fp32.float()
                    if initial_occ_fp32 is not None
                    else (torch.zeros((N_seq, H, K), device=k.device, dtype=torch.float32) if track_occ else None)
                )
                return (torch.zeros_like(v) if q is not None else None), S0, occ0

            lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
            max_len = max(lens)
            k_pad = torch.zeros((N_seq, max_len, H, K), device=k.device, dtype=torch.float32)
            v_pad = torch.zeros((N_seq, max_len, H, V), device=v.device, dtype=torch.float32)
            g_pad = torch.zeros((N_seq, max_len, H), device=g_fp32.device, dtype=torch.float32)
            b_pad = torch.zeros((N_seq, max_len, H), device=beta_fp32.device, dtype=torch.float32)
            q_pad = torch.zeros((N_seq, max_len, H, K), device=k.device, dtype=torch.float32) if q_f32 is not None else None

            offsets = cu_seqlens.tolist()
            for idx_s in range(N_seq):
                a, b = offsets[idx_s], offsets[idx_s + 1]
                ln = b - a
                if ln > 0:
                    k_pad[idx_s, :ln] = k_f32[0, a:b]
                    v_pad[idx_s, :ln] = v_f32[0, a:b]
                    g_pad[idx_s, :ln] = g_fp32[0, a:b].float()
                    b_pad[idx_s, :ln] = beta_fp32[0, a:b].float()
                    if q_pad is not None and q_f32 is not None:
                        q_pad[idx_s, :ln] = q_f32[0, a:b]

            o_b, s_b, occ_b, _ = pytorch_chunk_gdn(
                k=k_pad,
                v=v_pad,
                g=g_pad,
                beta=b_pad,
                q=q_pad,
                initial_state=initial_state_fp32,
                initial_norm_state=initial_occ_fp32,
                read_from="S_curr",
                chunk_size=64,
                track_norm_state=track_occ,
                return_penultimate_state=False,
            )
            o_packed = None
            if q is not None and o_b is not None:
                segs = [o_b[i, : lens[i]] for i in range(N_seq)]
                o_packed = torch.cat(segs, dim=0).unsqueeze(0).to(orig_dtype)
            return (
                o_packed,
                s_b.float(),
                occ_b.float() if occ_b is not None else None,
            )

        # Equal-length `(B, T, H, K)` path
        if k_f32.shape[1] == 0:
            B_in, _, H, K = k_f32.shape
            V = v_f32.shape[-1]
            S0 = (
                initial_state_fp32.float()
                if initial_state_fp32 is not None
                else torch.zeros((B_in, H, V, K), device=k.device, dtype=torch.float32)
            )
            occ0 = (
                initial_occ_fp32.float()
                if initial_occ_fp32 is not None
                else (torch.zeros((B_in, H, K), device=k.device, dtype=torch.float32) if track_occ else None)
            )
            return (torch.zeros_like(v) if q is not None else None), S0, occ0

        if k_f32.shape[1] <= 16:
            o_b, s_b, occ_b, _ = recurrent_gated_delta_rule(
                k=k_f32,
                v=v_f32,
                g=g_fp32.float(),
                beta=beta_fp32.float(),
                q=q_f32,
                initial_state=initial_state_fp32,
                initial_norm_state=initial_occ_fp32,
                read_from="S_curr",
                track_norm_state=track_occ,
                return_penultimate_state=False,
            )
        else:
            o_b, s_b, occ_b, _ = pytorch_chunk_gdn(
                k=k_f32,
                v=v_f32,
                g=g_fp32.float(),
                beta=beta_fp32.float(),
                q=q_f32,
                initial_state=initial_state_fp32,
                initial_norm_state=initial_occ_fp32,
                read_from="S_curr",
                chunk_size=64,
                track_norm_state=track_occ,
                return_penultimate_state=False,
            )
        return (
            o_b.to(orig_dtype) if o_b is not None else None,
            s_b.float(),
            occ_b.float() if occ_b is not None else None,
        )
    finally:
        if k.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32


class SlotGDNMemoryWriter(nn.Module):
    """Frame-Slot Gated DeltaNet Temporal Writer enforcing A1 (`g` active ONLY on slot `s=0`)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_k: int = 64,
        d_v: int = 64,
        beta_max: float = 2.0,
        min_halflife_frames: float = 8.0,
        max_halflife_frames: float = 512.0,
        use_seq_readout: bool = False,
        use_motion_beta_gate: bool = False,
        v3_beta_bias: float = -2.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.beta_max = float(beta_max)
        self.use_seq_readout = use_seq_readout

        self.k_proj = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * d_v, bias=False)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=True)
        self.a_proj = nn.Linear(d_model, num_heads, bias=False)

        hl = torch.logspace(
            math.log10(min_halflife_frames),
            math.log10(max_halflife_frames),
            steps=num_heads,
            dtype=torch.float32,
        )
        target_neg_g = math.log(2.0) / hl
        dt_bias = torch.log(torch.expm1(target_neg_g).clamp(min=1e-6))
        self.A_log = nn.Parameter(torch.zeros(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(dt_bias)

        # Sparse initial write bias (B3 / U5): prevents early memory saturation & static-wait overwrites
        nn.init.zeros_(self.beta_proj.weight)
        self.motion_beta_proj: Optional[nn.Linear] = None
        if use_motion_beta_gate:
            init_b = float(v3_beta_bias) if beta_max > 1.0 else (float(v3_beta_bias) + 0.5)
            nn.init.constant_(self.beta_proj.bias, init_b)
            self.motion_beta_proj = nn.Linear(1, num_heads, bias=False)
            # U5 empirical SigLIP patch-motion RMS calibration (`robomme_siglip_v1`):
            # static/wait frames `rms <= 0.20 -> beta = 2*sigmoid(-2.5 + 3.5*0.15) = 0.24 < 0.5` (contraction),
            # swap/pass frames `rms ~ 0.80 -> beta = 2*sigmoid(-2.5 + 3.5*0.80) = 1.15..1.31` (reflection).
            nn.init.constant_(self.motion_beta_proj.weight, 3.5)
        else:
            nn.init.constant_(self.beta_proj.bias, -1.5 if beta_max > 1.0 else -1.0)

        if use_seq_readout:
            self.q_proj = nn.Linear(d_model, num_heads * d_k, bias=False)
            self.g_out_proj = nn.Linear(d_model, num_heads * d_v, bias=True)
            self.o_proj = nn.Linear(num_heads * d_v, d_model, bias=False)

    @property
    def state_numel(self) -> int:
        return self.num_heads * self.d_v * self.d_k

    def project_frame_slots(
        self,
        frame_slots: torch.Tensor,                      # (..., W, N_w, d_model)
        motion_feat: Optional[torch.Tensor] = None,     # Optional (..., W, 1) visual motion RMS
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Project `(..., W, N_w, d)` into `(..., W * N_w, ...)` with strict A1 `g[..., 1:, :] = 0.0`!"""
        *lead, W, Nw, D = frame_slots.shape
        dt = self.k_proj.weight.dtype
        z = frame_slots.to(dt)

        k = l2_normalize_fp32(self.k_proj(z).view(*lead, W * Nw, self.num_heads, self.d_k), dim=-1)
        v = self.v_proj(z).view(*lead, W * Nw, self.num_heads, self.d_v).float()

        # A1: Frame-wise fp32 log-decay computed on slot s=0 ONLY; strictly 0.0 on slots s >= 1!
        z_s0 = z[..., 0, :]  # (..., W, D)
        raw_a = self.a_proj(z_s0).float() + self.dt_bias.float()
        g_s0 = (-torch.exp(self.A_log.float()) * F.softplus(raw_a)).clamp(min=-15.0, max=-1e-7)
        g_slots = torch.zeros((*lead, W, Nw, self.num_heads), device=z.device, dtype=torch.float32)
        g_slots[..., 0, :] = g_s0
        g_flat = g_slots.reshape(*lead, W * Nw, self.num_heads)

        raw_beta = self.beta_proj(z).float()
        if self.motion_beta_proj is not None:
            if motion_feat is not None:
                m_in = motion_feat.to(dtype=self.motion_beta_proj.weight.dtype).view(*lead, W, 1, 1)
            else:
                m_in = torch.zeros(
                    (*lead, W, 1, 1), device=z.device, dtype=self.motion_beta_proj.weight.dtype
                )
            raw_beta = raw_beta + self.motion_beta_proj(m_in).float()

        beta_flat = (self.beta_max * torch.sigmoid(raw_beta)).reshape(
            *lead, W * Nw, self.num_heads
        )
        q = None
        if self.use_seq_readout:
            q = l2_normalize_fp32(self.q_proj(z).view(*lead, W * Nw, self.num_heads, self.d_k), dim=-1)
        return k, v, g_flat, beta_flat, q

    def project_writeback_slot(
        self,
        wb_slot: torch.Tensor,  # (B, 1, d_model)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project a chunk-boundary write-back slot `c_wb` with `g == 0.0` (zero extra frame decay!)."""
        B, L_wb, _ = wb_slot.shape
        dt = self.k_proj.weight.dtype
        z = wb_slot.to(dt)
        k = l2_normalize_fp32(self.k_proj(z).view(B, L_wb, self.num_heads, self.d_k), dim=-1)
        v = self.v_proj(z).view(B, L_wb, self.num_heads, self.d_v).float()
        g = torch.zeros((B, L_wb, self.num_heads), device=z.device, dtype=torch.float32)
        beta = self.beta_max * torch.sigmoid(self.beta_proj(z).float())
        return k, v, g, beta

    def format_seq_readout(self, z_flat: torch.Tensor, raw_o: torch.Tensor) -> torch.Tensor:
        """Apply SiLU/sigmoid output gate to `raw_o: (..., T, H, d_v)` -> `(..., T, d_model)`."""
        dt = self.k_proj.weight.dtype
        gate = F.silu(self.g_out_proj(z_flat.to(dt))).view(*raw_o.shape)
        flat = (gate * raw_o.to(dt)).reshape(*raw_o.shape[:-2], self.num_heads * self.d_v)
        return self.o_proj(flat)


# =============================================================================
# 5. Prelude, Multi-Slot Damped Readout (`C3`), Weight-Tied Core, Coda, Halting
# =============================================================================


class DampedSlotMemoryReadout(nn.Module):
    """Multi-slot dynamic readout from `S_past` with Damped RMSNorm (`RESEARCH_PLAN.md` §3.3 Eq. 114)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_k: int = 64,
        d_v: int = 64,
        eps_read: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.eps_read = float(eps_read)

        self.q_proj = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.g_proj = nn.Linear(d_model, num_heads * d_v, bias=True)
        self.o_proj = nn.Linear(num_heads * d_v, d_model, bias=False)
        # Small init on o_proj keeps early loop updates smooth while preserving immediate grad flow to S!
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=0.02)

    def forward(
        self,
        h_slots: torch.Tensor,                  # (B, N_c, d_model)
        S_past: torch.Tensor,                   # (B, H, d_v, d_k) float32
        s_occ_past: Optional[torch.Tensor] = None,  # (B, H, d_k) float32
    ) -> Dict[str, torch.Tensor]:
        B, Nc, _ = h_slots.shape
        dt = self.q_proj.weight.dtype
        h_in = h_slots.to(dt)

        q = l2_normalize_fp32(self.q_proj(h_in).view(B, Nc, self.num_heads, self.d_k), dim=-1)
        # S_past: (B, H, V, K), q: (B, Nc, H, K) -> raw: (B, Nc, H, V) strictly in fp32 (no autocast!)
        with torch.autocast(device_type=h_slots.device.type, enabled=False):
            raw_f32 = torch.einsum("bhvk,bnhk->bnhv", S_past.float(), q.float())
            rms = torch.sqrt(torch.mean(raw_f32 * raw_f32, dim=-1, keepdim=True) + self.eps_read * self.eps_read)
            r_damped = (raw_f32 / rms).to(dt)

        gate = F.silu(self.g_proj(h_in)).view(B, Nc, self.num_heads, self.d_v)
        m = self.o_proj((gate * r_damped).reshape(B, Nc, self.num_heads * self.d_v))

        # Diagnostic norms & occupancy coverage averaged across the N_c query slots -> (B, H)
        r_norm = torch.linalg.vector_norm(raw_f32, ord=2, dim=-1).mean(dim=1)  # (B, H)
        if s_occ_past is not None:
            cov = torch.sum(q.float() * s_occ_past.float().unsqueeze(1), dim=-1).mean(dim=1)
        else:
            cov = torch.zeros_like(r_norm)

        return {
            "m": m,
            "q": q,
            "r_norm": r_norm,
            "coverage": cov,
        }


class VLAPrelude(nn.Module):
    """Extracts initial `N_c = 4` latent reasoning slots `h_t^{(0)}` from current context `E_t`."""

    def __init__(self, cfg: DeltaLoopVLAConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.num_coda = cfg.num_coda_tokens
        self.init_queries = nn.Parameter(torch.randn(cfg.num_coda_tokens, d) * 0.02)
        self.q_norm = nn.RMSNorm(d)
        self.cross_attn = MultiHeadAttn(d, cfg.attn_heads)
        self.ffn_norm = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d * 2)

    def forward(self, ctx_tokens: torch.Tensor, ctx_mask: torch.Tensor) -> torch.Tensor:
        B = ctx_tokens.shape[0]
        dt = ctx_tokens.dtype
        q = self.q_norm(self.init_queries.to(dt).unsqueeze(0).expand(B, -1, -1))
        h = q + self.cross_attn(q, ctx_tokens, kv_valid_mask=ctx_mask)
        h = h + self.ffn(self.ffn_norm(h))
        return h  # (B, N_c, d)


class VLARecurrentCore(nn.Module):
    """Weight-tied depth-recurrent reasoning block over `(h_t^{(l)}, m_t^{(l)}, E_t, e_loop(l))`."""

    def __init__(self, cfg: DeltaLoopVLAConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.max_loops = max(cfg.max_loops, 8)
        self.loop_emb = nn.Embedding(self.max_loops, d)
        nn.init.normal_(self.loop_emb.weight, mean=0.0, std=0.02)

        self.sa_norm = nn.RMSNorm(d)
        self.self_attn = MultiHeadAttn(d, cfg.attn_heads)

        self.ca_norm = nn.RMSNorm(d)
        self.ctx_attn = MultiHeadAttn(d, cfg.attn_heads)

        self.h_norm = nn.RMSNorm(d)
        self.m_norm = nn.RMSNorm(d)
        self.fuse_proj = nn.Linear(2 * d, d, bias=True)
        self.update_gate = nn.Linear(2 * d, d, bias=True)
        nn.init.constant_(self.update_gate.bias, 0.0)

        self.ffn_norm = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d * 2)

    def get_loop_emb(self, l_idx: int, dtype: torch.dtype) -> torch.Tensor:
        idx = min(l_idx, self.max_loops - 1)
        return self.loop_emb.weight[idx].to(dtype).view(1, 1, -1)

    def forward(
        self,
        h: torch.Tensor,           # (B, N_c, d)
        m: torch.Tensor,           # (B, N_c, d)
        ctx_tokens: torch.Tensor,  # (B, N_ctx, d)
        ctx_mask: torch.Tensor,    # (B, N_ctx) bool
        l_idx: int,
    ) -> torch.Tensor:
        e_l = self.get_loop_emb(l_idx, h.dtype)
        h_step = h + e_l
        h_sa = h + self.self_attn(self.sa_norm(h_step), self.sa_norm(h_step))
        h_ctx = h_sa + self.ctx_attn(self.ca_norm(h_sa), ctx_tokens, kv_valid_mask=ctx_mask)

        cat = torch.cat([self.h_norm(h_ctx), self.m_norm(m)], dim=-1)
        cand = self.fuse_proj(cat)
        gate = torch.sigmoid(self.update_gate(cat))
        h_mid = h_ctx + gate * cand
        return h_mid + self.ffn(self.ffn_norm(h_mid))


class VLACoda(nn.Module):
    """Maps latent slots `h_t^{(l+1)}` -> `C_t^{(l)} in R^{B x N_c x d}`."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))


class CRNHaltingGate(nn.Module):
    """Stage-II Variance-Reduced (`M_CRN=8`) Marginal-Gain Adaptive Halting Gate (`RESEARCH_PLAN.md` §3.5)."""

    def __init__(self, d_model: int, num_heads: int = 8, max_loops: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_loops = max_loops

        # Pooled h^{(l)}, pooled m^{(l)}, ||dh||_2, ||dc||_2, per-head r_norm (H), per-head coverage (H), loop one-hot
        in_dim = 2 * d_model + 2 + 2 * num_heads + max_loops
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        nn.init.constant_(self.net[-1].bias, -0.5)
        self.register_buffer("calibrated_delta", torch.tensor(0.005, dtype=torch.float32))

    def step_hazard(
        self,
        h_l: torch.Tensor,         # (B, N_c, d)
        m_l: torch.Tensor,         # (B, N_c, d)
        dh_norm: torch.Tensor,     # (B, 1)
        dc_norm: torch.Tensor,     # (B, 1)
        r_norm: torch.Tensor,      # (B, H)
        coverage: torch.Tensor,    # (B, H)
        l_idx: int,
        L_max: int,
    ) -> torch.Tensor:
        """Compute hazard `lambda_t^{(l)}` in `(0, 1]` at a single loop step `l_idx`."""
        B = h_l.shape[0]
        if l_idx >= L_max - 1:
            return torch.ones(B, device=h_l.device, dtype=torch.float32)
        oh = F.one_hot(
            torch.full((B,), min(l_idx, self.max_loops - 1), device=h_l.device, dtype=torch.long),
            num_classes=self.max_loops,
        ).float()
        feat = torch.cat(
            [
                h_l.float().mean(dim=1),
                m_l.float().mean(dim=1),
                dh_norm.float(),
                dc_norm.float(),
                r_norm.float(),
                coverage.float(),
                oh,
            ],
            dim=-1,
        )
        dt = self.net[0].weight.dtype  # type: ignore[union-attr]
        return torch.sigmoid(self.net(feat.to(dt)).float().squeeze(-1))

    def forward_all(
        self,
        h_list: List[torch.Tensor],
        m_list: List[torch.Tensor],
        dh_norms: List[torch.Tensor],
        dc_norms: List[torch.Tensor],
        r_norms: List[torch.Tensor],
        coverages: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        L_max = len(h_list)
        B = h_list[0].shape[0]
        device = h_list[0].device

        hazards = [
            self.step_hazard(
                h_list[l], m_list[l], dh_norms[l], dc_norms[l], r_norms[l], coverages[l], l, L_max
            )
            for l in range(L_max)
        ]
        lam = torch.stack(hazards, dim=1)  # (B, L_max)
        one_minus = (1.0 - lam).clamp(min=1e-7, max=1.0)
        surv = torch.cat(
            [torch.ones((B, 1), device=device, dtype=torch.float32), torch.cumprod(one_minus[:, :-1], dim=1)],
            dim=1,
        )
        p_tilde = lam * surv
        p_tilde = p_tilde / p_tilde.sum(dim=1, keepdim=True).clamp(min=1e-7)
        depths = torch.arange(1, L_max + 1, device=device, dtype=torch.float32)
        expected_loops = torch.sum(p_tilde * depths.unsqueeze(0), dim=1)
        return {
            "lambda": lam,
            "p_tilde": p_tilde,
            "expected_loops": expected_loops,
        }

    def compute_stage2_loss(
        self,
        halt_out: Dict[str, torch.Tensor],
        crn_losses_detached: torch.Tensor,  # (B, L_max) averaged over M_CRN=8 draws
        delta_margin: Optional[float] = None,
        ponder_cost: float = 0.003,
    ) -> torch.Tensor:
        """Combined Winner's-Curse-free marginal-gain BCE (`RESEARCH_PLAN.md` Eq. 163) + expected CRN loss."""
        lam = halt_out["lambda"]          # (B, L_max)
        p_tilde = halt_out["p_tilde"]     # (B, L_max)
        L_max = lam.shape[1]
        margin = float(self.calibrated_delta.item()) if delta_margin is None else float(delta_margin)

        loss_final = crn_losses_detached[:, -1:].detach()
        # w_t^{(l)} = 1 if depth l is worse than L_max-1 by > margin -> keep looping (lambda -> 0)!
        w_keep = ((crn_losses_detached[:, :-1].detach() - loss_final) > margin).float()
        lam_clamped = lam[:, :-1].clamp(min=1e-6, max=1.0 - 1e-6)
        bce = -(w_keep * torch.log(1.0 - lam_clamped) + (1.0 - w_keep) * torch.log(lam_clamped)).mean()

        exp_fm = torch.sum(p_tilde * crn_losses_detached.detach(), dim=1).mean()
        step_pen = ponder_cost * halt_out["expected_loops"].mean()
        return bce + exp_fm + step_pen


# =============================================================================
# 6. Unified `DeltaLoopVLAPolicy`
# =============================================================================


class DeltaLoopVLAPolicy(nn.Module):
    """Unified Visuomotor Flow-Matching Policy implementing `m0..m6` + `m6b_deltaloop_wb`."""

    def __init__(self, cfg: DeltaLoopVLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.arch: ArchVLA = cfg.arch
        d = cfg.d_model

        # 100% Identical Shared Observation & Action Expert Backbone
        self.obs_encoder = SharedObservationEncoder(cfg)
        self.action_expert = FlowActionExpert(cfg)

        # Architecture-specific Memory & Loop modules
        self.slot_resampler: Optional[PerceiverSlotResampler] = None
        self.gdn: Optional[SlotGDNMemoryWriter] = None
        self.gdn1: Optional[SlotGDNMemoryWriter] = None
        self.gdn2: Optional[SlotGDNMemoryWriter] = None
        self.mid_norm: Optional[nn.RMSNorm] = None
        self.mid_ffn: Optional[SwiGLU] = None
        self.readout: Optional[DampedSlotMemoryReadout] = None
        self.readout1: Optional[DampedSlotMemoryReadout] = None
        self.readout2: Optional[DampedSlotMemoryReadout] = None
        self.prelude: Optional[VLAPrelude] = None
        self.core: Optional[VLARecurrentCore] = None
        self.coda: Optional[VLACoda] = None
        self.halting_gate: Optional[CRNHaltingGate] = None
        self.wb_proj: Optional[nn.Linear] = None
        self.framesamp_refine: Optional[ContextTransformerBlock] = None
        self.aux_head: Optional[nn.Linear] = None

        if self.arch == "m0_memoryless":
            pass

        elif self.arch == "m1_rdvla_loop4":
            # Memoryless Looped Action Core (L=4, no GDN memory — RD-VLA equivalent!)
            self.prelude = VLAPrelude(cfg)
            self.core = VLARecurrentCore(cfg)
            self.coda = VLACoda(d)
            self.aux_head = nn.Linear(d, cfg.aux_dim)

        elif self.arch == "m2_framesamp_modul":
            # Perceptual Frame-Sampling buffer (32 past frame slots + Modulator/Cross-Attn)
            self.slot_resampler = PerceiverSlotResampler(
                cfg, num_slots=1
            )  # 1 perceptual slot per sampled frame -> 32 buffer tokens
            # Disable temporal conv on non-uniform linspace buffer slots
            self.slot_resampler.temporal_conv = None
            self.framesamp_refine = ContextTransformerBlock(d, cfg.attn_heads, d * 2)
            self.coda = VLACoda(d)
            self.aux_head = nn.Linear(d, cfg.aux_dim)

        elif self.arch in ("m5_gdn2layer_b2_l1", "m7_deltaloop_2l_b2"):
            # Strictly Iso-State 2-Layer Stacked Neg-Eig GDN (+ Multi-Hop Dynamic Readout on `m7`):
            # 2 * (H * (d_v // 2) * d_k) == 1-Layer (H * d_v * d_k)!
            d_v_half = cfg.gdn_d_v // 2
            self.slot_resampler = PerceiverSlotResampler(cfg)
            self.gdn1 = SlotGDNMemoryWriter(
                d_model=d,
                num_heads=cfg.gdn_heads,
                d_k=cfg.gdn_d_k,
                d_v=d_v_half,
                beta_max=2.0,
                min_halflife_frames=cfg.min_halflife_frames,
                max_halflife_frames=cfg.max_halflife_frames,
                use_seq_readout=True,
                use_motion_beta_gate=cfg.use_motion_beta_gate,
                v3_beta_bias=cfg.v3_beta_bias,
            )
            self.mid_norm = nn.RMSNorm(d)
            self.mid_ffn = SwiGLU(d, d * 2)
            self.gdn2 = SlotGDNMemoryWriter(
                d_model=d,
                num_heads=cfg.gdn_heads,
                d_k=cfg.gdn_d_k,
                d_v=d_v_half,
                beta_max=2.0,
                min_halflife_frames=cfg.min_halflife_frames,
                max_halflife_frames=cfg.max_halflife_frames,
                use_seq_readout=False,
                use_motion_beta_gate=cfg.use_motion_beta_gate,
                v3_beta_bias=cfg.v3_beta_bias,
            )
            self.readout1 = DampedSlotMemoryReadout(
                d, cfg.gdn_heads, cfg.gdn_d_k, d_v_half, eps_read=cfg.readout_eps
            )
            self.readout2 = DampedSlotMemoryReadout(
                d, cfg.gdn_heads, cfg.gdn_d_k, d_v_half, eps_read=cfg.readout_eps
            )
            self.prelude = VLAPrelude(cfg)
            self.core = VLARecurrentCore(cfg)
            self.coda = VLACoda(d)
            self.aux_head = nn.Linear(d, cfg.aux_dim)
            if self.arch == "m7_deltaloop_2l_b2":
                self.halting_gate = CRNHaltingGate(
                    d_model=d, num_heads=cfg.gdn_heads, max_loops=cfg.max_loops
                )

        elif self.arch in ("m3_gdn_b1_l1", "m4_gdn_b2_l1", "m6_deltaloop_b2", "m6b_deltaloop_wb"):
            beta_max = 1.0 if self.arch == "m3_gdn_b1_l1" else 2.0
            self.slot_resampler = PerceiverSlotResampler(cfg)
            self.gdn = SlotGDNMemoryWriter(
                d_model=d,
                num_heads=cfg.gdn_heads,
                d_k=cfg.gdn_d_k,
                d_v=cfg.gdn_d_v,
                beta_max=beta_max,
                min_halflife_frames=cfg.min_halflife_frames,
                max_halflife_frames=cfg.max_halflife_frames,
                use_seq_readout=False,
                use_motion_beta_gate=cfg.use_motion_beta_gate,
                v3_beta_bias=cfg.v3_beta_bias,
            )
            self.readout = DampedSlotMemoryReadout(
                d, cfg.gdn_heads, cfg.gdn_d_k, cfg.gdn_d_v, eps_read=cfg.readout_eps
            )
            self.prelude = VLAPrelude(cfg)
            self.core = VLARecurrentCore(cfg)
            self.coda = VLACoda(d)
            self.aux_head = nn.Linear(d, cfg.aux_dim)

            if self.arch in ("m6_deltaloop_b2", "m6b_deltaloop_wb"):
                self.halting_gate = CRNHaltingGate(
                    d_model=d, num_heads=cfg.gdn_heads, max_loops=cfg.max_loops
                )
            if self.arch == "m6b_deltaloop_wb":
                self.wb_proj = nn.Linear(d, d, bias=True)
                nn.init.zeros_(self.wb_proj.weight)
                nn.init.zeros_(self.wb_proj.bias)
        else:
            raise ValueError(f"Unknown VLA architecture: {self.arch!r}")

        # Cycle 3 `U4`: Shared training-only 2D grounding `<x, y>` + subgoal language alignment heads
        # Instantiated identically across ALL architectures (`m0..m6b`) ONLY when `use_grounding_head=True`!
        self.grounding_xy_head: Optional[nn.Sequential] = None
        self.grounding_sg_head: Optional[nn.Sequential] = None
        if cfg.use_grounding_head:
            self.grounding_xy_head = nn.Sequential(
                nn.RMSNorm(d),
                nn.Linear(d, d),
                nn.SiLU(),
                nn.Linear(d, 2),
            )
            self.grounding_sg_head = nn.Sequential(
                nn.RMSNorm(d),
                nn.Linear(d, d),
                nn.SiLU(),
                nn.Linear(d, cfg.subgoal_dim),
            )

        # Cycle 3 `U1`: Non-persistent `(H_act, 8)` velocity loss weights (`RESEARCH_PLAN.md` Eq. 106)
        if cfg.v3_mode:
            w_hd = torch.zeros((cfg.action_horizon, cfg.action_dim), dtype=torch.float32)
            for h_i in range(cfg.action_horizon):
                if h_i < cfg.exec_horizon:
                    w_hd[h_i, :7] = 1.0 * (0.97**h_i)
                    w_hd[h_i, 7:] = 2.5
                else:
                    w_hd[h_i, :7] = 0.20
                    w_hd[h_i, 7:] = 0.40
            w_hd = w_hd / w_hd.sum().clamp(min=1e-8)
            self.register_buffer("_u1_loss_weights", w_hd, persistent=False)
        else:
            self._u1_loss_weights = None

        # Fixed random projection buffer for visual-anchor auxiliary target (not trained!)
        gen = torch.Generator()
        gen.manual_seed(20261111)
        vis_anchor_mat = torch.randn((cfg.vis_dim, 32), generator=gen, dtype=torch.float32) / math.sqrt(cfg.vis_dim)
        self.register_buffer("vis_anchor_proj", vis_anchor_mat, persistent=True)

    # -------------------------------------------------------------------------
    # Introspection & Parameter Grouping (for Separate Gradient Clipping!)
    # -------------------------------------------------------------------------

    @property
    def uses_history(self) -> bool:
        return self.arch not in ("m0_memoryless", "m1_rdvla_loop4")

    @property
    def uses_gdn(self) -> bool:
        return self.arch in (
            "m3_gdn_b1_l1",
            "m4_gdn_b2_l1",
            "m5_gdn2layer_b2_l1",
            "m6_deltaloop_b2",
            "m6b_deltaloop_wb",
            "m7_deltaloop_2l_b2",
        )

    @property
    def is_multi_depth(self) -> bool:
        return self.arch in (
            "m1_rdvla_loop4",
            "m6_deltaloop_b2",
            "m6b_deltaloop_wb",
            "m7_deltaloop_2l_b2",
        )

    @property
    def default_train_loops(self) -> int:
        return self.cfg.max_loops if self.is_multi_depth else 1

    def state_size_bytes(self) -> int:
        """Persistent memory state size in bytes per environment."""
        if self.arch in ("m0_memoryless", "m1_rdvla_loop4"):
            return 0
        if self.arch == "m2_framesamp_modul":
            # 32 bf16 tokens of width d_model
            return self.cfg.framesamp_buffer_size * self.cfg.d_model * 2
        if self.arch in ("m5_gdn2layer_b2_l1", "m7_deltaloop_2l_b2"):
            assert self.gdn1 is not None and self.gdn2 is not None
            occ_elems = 2 * self.cfg.gdn_heads * self.cfg.gdn_d_k
            return (self.gdn1.state_numel + self.gdn2.state_numel + occ_elems) * 4
        assert self.gdn is not None
        occ_elems = self.cfg.gdn_heads * self.cfg.gdn_d_k
        return (self.gdn.state_numel + occ_elems) * 4

    def parameter_breakdown(self) -> Dict[str, int]:
        backbone = sum(p.numel() for p in self.obs_encoder.parameters()) + sum(
            p.numel() for p in self.action_expert.parameters()
        )
        halt = sum(p.numel() for p in self.halting_gate.parameters()) if self.halting_gate is not None else 0
        aux = sum(p.numel() for p in self.aux_head.parameters()) if self.aux_head is not None else 0
        ground = 0
        if self.grounding_xy_head is not None:
            ground += sum(p.numel() for p in self.grounding_xy_head.parameters())
        if self.grounding_sg_head is not None:
            ground += sum(p.numel() for p in self.grounding_sg_head.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "shared_backbone": backbone,
            "memory_and_loop": total - backbone - halt - aux - ground,
            "halting_gate": halt,
            "aux_head": aux + ground,
        }

    def split_write_vs_other_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Partition trainable parameters for separate gradient norm clipping!"""
        write_prefixes = ("slot_resampler.", "gdn.", "gdn1.", "gdn2.", "mid_norm.", "mid_ffn.", "framesamp_refine.", "wb_proj.")
        write_params: List[nn.Parameter] = []
        other_params: List[nn.Parameter] = []
        for name, p in self.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if name.startswith(write_prefixes):
                write_params.append(p)
            else:
                other_params.append(p)
        return write_params, other_params

    # -------------------------------------------------------------------------
    # A3 Cutoff, U5 Adaptive Demo Stride & Frame-Sampling Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_a3_cutoff_excl(step_idx: int, exec_start_idx: int, exclusion_window: int = 16) -> int:
        """Exact A3 Video-Demo Cutoff Rule (`cutoff_excl` for eligible history `[0 : cutoff_excl)`).

        - At `step_idx == exec_start_idx`: returns `exec_start_idx` so ALL conditioning video demo
          frames `0 .. exec_start_idx - 1` remain in memory!
        - At later steps: returns `max(exec_start_idx, max(0, step_idx - exclusion_window + 1))`,
          excluding the last `exclusion_window` near-duplicate execution frames.
        """
        return max(int(exec_start_idx), max(0, int(step_idx) - int(exclusion_window) + 1))

    @staticmethod
    def compute_written_frame_indices(
        cutoff_excl: int,
        exec_start_idx: int,
        cfg: DeltaLoopVLAConfig,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Return `(written_frame_indices_lt_cutoff, num_demo_strided)` under Cycle 2A or Cycle 3 `U5` striding.

        - When `cfg.v3_mode == False`: uniform `0, write_stride, 2*write_stride, ... < cutoff_excl`.
        - When `cfg.v3_mode == True` (`U5` adaptive demo striding):
            * Demo segment `[0, min(cutoff_excl, exec_start_idx))`: stride `long_demo_stride` (`8`)
              if `exec_start_idx > adaptive_demo_stride_threshold` (`256`) else `write_stride` (`4`).
            * Execution segment `[exec_start_idx, cutoff_excl)`: anchored at `exec_start_idx` with
              stride `write_stride` (`4`), i.e., `exec_start_idx, exec_start_idx + 4, ... < cutoff_excl`.
        """
        cut = max(0, int(cutoff_excl))
        ex = max(0, int(exec_start_idx))
        ws = int(cfg.write_stride)
        if not cfg.v3_mode:
            if cut <= 0:
                return torch.zeros((0,), dtype=torch.long, device=device), 0
            idx = torch.arange(0, cut, ws, dtype=torch.long, device=device)
            n_demo = (min(cut, ex) + ws - 1) // ws
            return idx, int(n_demo)

        ds = int(cfg.long_demo_stride) if ex > int(cfg.adaptive_demo_stride_threshold) else ws
        demo_lim = min(cut, ex)
        demo_idx = (
            torch.arange(0, demo_lim, ds, dtype=torch.long, device=device)
            if demo_lim > 0
            else torch.zeros((0,), dtype=torch.long, device=device)
        )
        n_demo = int(demo_idx.numel())
        if cut <= ex:
            return demo_idx, n_demo
        exec_idx = torch.arange(ex, cut, ws, dtype=torch.long, device=device)
        return torch.cat([demo_idx, exec_idx], dim=0), n_demo

    @staticmethod
    def compute_visual_motion_rms(front_seq: torch.Tensor) -> torch.Tensor:
        """Compute causal per-written-frame visual motion RMS `(W, 1)` in `float32` (`U5`)."""
        W = int(front_seq.shape[0])
        if W == 0:
            return torch.empty((0, 1), device=front_seq.device, dtype=torch.float32)
        f32 = front_seq.float()
        m0 = torch.sqrt(torch.mean(f32[0:1] * f32[0:1], dim=(-2, -1)) + 1e-8)
        if W == 1:
            return m0.view(1, 1)
        diffs = f32[1:] - f32[:-1]
        m_rest = torch.sqrt(torch.mean(diffs * diffs, dim=(-2, -1)) + 1e-8)
        return torch.cat([m0, m_rest], dim=0).view(W, 1)

    @staticmethod
    def select_framesamp_slot_indices(
        num_eligible_strided: int,
        num_demo_strided: int,
        budget: int = 32,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, int]:
        """Select up to `budget` indices into the strided history `[0 .. num_eligible_strided - 1]`."""
        if num_eligible_strided <= 0:
            return torch.zeros((0,), dtype=torch.long, device=device), 0
        if num_eligible_strided <= budget:
            return torch.arange(num_eligible_strided, dtype=torch.long, device=device), num_eligible_strided

        if num_demo_strided > 0:
            demo_b = min(budget // 2, num_demo_strided)
            demo_idx = torch.linspace(0, num_demo_strided - 1, steps=demo_b, device=device).long()
            exec_avail = num_eligible_strided - num_demo_strided
            rem_b = budget - demo_b
            if exec_avail <= 0:
                idx = torch.linspace(0, num_demo_strided - 1, steps=budget, device=device).long()
            elif exec_avail <= rem_b:
                exec_idx = torch.arange(num_demo_strided, num_eligible_strided, dtype=torch.long, device=device)
                idx = torch.cat([demo_idx, exec_idx], dim=0)
            else:
                exec_idx = torch.linspace(
                    num_demo_strided, num_eligible_strided - 1, steps=rem_b, device=device
                ).long()
                idx = torch.cat([demo_idx, exec_idx], dim=0)
            return idx, int(idx.numel())

        idx = torch.linspace(0, num_eligible_strided - 1, steps=budget, device=device).long()
        return idx, budget

    # -------------------------------------------------------------------------
    # Packed Episode-Prefix Memory Encoding (Fast GH200 Training + CPU Parity)
    # -------------------------------------------------------------------------

    @staticmethod
    def _scan_episode_prefix_states(
        k_all: torch.Tensor,                  # (1, Total_Tok, H, K)
        v_all: torch.Tensor,                  # (1, Total_Tok, H, V_dim)
        g_all: torch.Tensor,                  # (1, Total_Tok, H)
        b_all: torch.Tensor,                  # (1, Total_Tok, H)
        ep_tok_offset: Dict[int, int],
        counts_list: List[int],
        ep_ids_list: List[int],
        Nw: int,
        V_dim: int,
        force_pytorch: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Deduplicate `(episode, wc)` prefixes and run a single warm-cache `cu_seqlens` GDN pass (`initial_state=None`)."""
        B = len(counts_list)
        device = k_all.device
        H = int(k_all.shape[2])
        K = int(k_all.shape[3])

        S_out = torch.zeros((B, H, V_dim, K), device=device, dtype=torch.float32)
        occ_out = torch.zeros((B, H, K), device=device, dtype=torch.float32)

        # Deduplicate identical `(e_id, wc)` pairs within the batch while keeping `initial_state=None`
        # so `fla_chunk_gdn` hits 100% pre-compiled Hopper Triton kernels (`IS_VARLEN=True, USE_INITIAL_STATE=False`).
        uniq_key_to_idx: Dict[Tuple[int, int], int] = {}
        uniq_idx_ranges: List[torch.Tensor] = []
        uniq_seg_lens: List[int] = []
        pos_b_indices: List[int] = []
        pos_u_indices: List[int] = []

        for b_i, wc in enumerate(counts_list):
            if wc <= 0:
                continue
            e_id = ep_ids_list[b_i]
            if e_id not in ep_tok_offset:
                continue
            key = (e_id, wc)
            u_idx = uniq_key_to_idx.get(key)
            if u_idx is None:
                u_idx = len(uniq_seg_lens)
                uniq_key_to_idx[key] = u_idx
                off = ep_tok_offset[e_id]
                ln = wc * Nw
                uniq_idx_ranges.append(torch.arange(off, off + ln, dtype=torch.long))
                uniq_seg_lens.append(ln)
            pos_b_indices.append(b_i)
            pos_u_indices.append(u_idx)

        if not uniq_seg_lens:
            return S_out, occ_out

        gather_idx = torch.cat(uniq_idx_ranges, dim=0).to(device=device, non_blocking=True)
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(uniq_seg_lens), dim=0).tolist()),
            device=device,
            dtype=torch.int32,
        )
        _, S_uniq, occ_uniq = run_dual_backend_gdn(
            k=k_all[:, gather_idx],
            v=v_all[:, gather_idx],
            g_fp32=g_all[:, gather_idx],
            beta_fp32=b_all[:, gather_idx],
            cu_seqlens=cu_seqlens,
            track_occ=True,
            force_pytorch=force_pytorch,
        )
        assert occ_uniq is not None
        u_tensor = torch.tensor(pos_u_indices, device=device, dtype=torch.long)
        b_tensor = torch.tensor(pos_b_indices, device=device, dtype=torch.long)
        S_out = S_out.index_copy(0, b_tensor, S_uniq.index_select(0, u_tensor))
        occ_out = occ_out.index_copy(0, b_tensor, occ_uniq.index_select(0, u_tensor))
        return S_out, occ_out

    def compute_packed_gdn_states(
        self,
        episode_slot_seqs: List[torch.Tensor],        # len B_ep, each `(W_e, N_w, d_model)`
        sample_ep_ids: torch.Tensor,                  # (B,) int64 in `0 .. B_ep - 1`
        sample_written_counts: torch.Tensor,          # (B,) int64 number of written frames `< cutoff_excl`
        *,
        episode_motion_seqs: Optional[List[torch.Tensor]] = None,  # Optional len B_ep, each `(W_e, 1)`
        force_pytorch: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute exact causal `S_past` for `B` chunks from `B_ep` pre-encoded episodes via monotonic segment scan."""
        B = int(sample_ep_ids.shape[0])
        device = sample_ep_ids.device
        Nw = self.cfg.num_write_slots
        H = self.cfg.gdn_heads
        K = self.cfg.gdn_d_k

        counts_list = sample_written_counts.tolist()
        ep_ids_list = sample_ep_ids.tolist()

        if self.arch in ("m5_gdn2layer_b2_l1", "m7_deltaloop_2l_b2"):
            assert self.gdn1 is not None and self.gdn2 is not None and self.mid_norm is not None and self.mid_ffn is not None
            Vh = self.cfg.gdn_d_v // 2

            nonempty_z = []
            nonempty_m = []
            ep_tok_lens = []
            ep_tok_offset: Dict[int, int] = {}
            tok_cursor = 0
            for e_i, z_ep in enumerate(episode_slot_seqs):
                W_e = int(z_ep.shape[0])
                if W_e > 0:
                    nonempty_z.append(z_ep)
                    if episode_motion_seqs is not None:
                        nonempty_m.append(episode_motion_seqs[e_i])
                    ln = W_e * Nw
                    ep_tok_lens.append(ln)
                    ep_tok_offset[e_i] = tok_cursor
                    tok_cursor += ln

            if not nonempty_z:
                z_s = torch.zeros((B, H, Vh, K), device=device, dtype=torch.float32)
                z_o = torch.zeros((B, H, K), device=device, dtype=torch.float32)
                return {"S1": z_s, "occ1": z_o, "S2": z_s.clone(), "occ2": z_o.clone()}

            z_all = torch.cat(nonempty_z, dim=0)  # (Total_W, Nw, d)
            m_all = torch.cat(nonempty_m, dim=0).unsqueeze(0) if nonempty_m else None
            k1_all, v1_all, g1_all, b1_all, q1_all = self.gdn1.project_frame_slots(
                z_all.unsqueeze(0), motion_feat=m_all
            )
            assert q1_all is not None
            cu_ep = torch.tensor(
                [0] + list(torch.cumsum(torch.tensor(ep_tok_lens), dim=0).tolist()),
                device=device,
                dtype=torch.int32,
            )
            o1_cat, _, _ = run_dual_backend_gdn(
                k=k1_all,
                v=v1_all,
                g_fp32=g1_all,
                beta_fp32=b1_all,
                q=q1_all,
                cu_seqlens=cu_ep,
                track_occ=False,
                force_pytorch=force_pytorch,
            )
            assert o1_cat is not None
            z_cat = z_all.reshape(tok_cursor, -1)
            seq1_cat = self.gdn1.format_seq_readout(z_cat, o1_cat[0])
            z_mid_cat = z_cat.to(seq1_cat.dtype) + seq1_cat
            z_mid_cat = z_mid_cat + self.mid_ffn(self.mid_norm(z_mid_cat))
            k2_all, v2_all, g2_all, b2_all, _ = self.gdn2.project_frame_slots(
                z_mid_cat.view(1, tok_cursor // Nw, Nw, -1), motion_feat=m_all
            )

            S1_out, occ1_out = self._scan_episode_prefix_states(
                k1_all, v1_all, g1_all, b1_all, ep_tok_offset, counts_list, ep_ids_list, Nw, Vh, force_pytorch
            )
            S2_out, occ2_out = self._scan_episode_prefix_states(
                k2_all, v2_all, g2_all, b2_all, ep_tok_offset, counts_list, ep_ids_list, Nw, Vh, force_pytorch
            )
            return {"S1": S1_out, "occ1": occ1_out, "S2": S2_out, "occ2": occ2_out}

        # Single-layer GDN (`m3`, `m4`, `m6`) — single batched projection + monotonic per-episode scan!
        assert self.gdn is not None
        V = self.cfg.gdn_d_v

        nonempty_z = []
        nonempty_m_1l = []
        ep_tok_offset_1l: Dict[int, int] = {}
        tok_cursor_1l = 0
        for e_i, z_ep in enumerate(episode_slot_seqs):
            W_e = int(z_ep.shape[0])
            if W_e > 0:
                nonempty_z.append(z_ep)
                if episode_motion_seqs is not None:
                    nonempty_m_1l.append(episode_motion_seqs[e_i])
                ep_tok_offset_1l[e_i] = tok_cursor_1l
                tok_cursor_1l += W_e * Nw

        if not nonempty_z:
            return {
                "S": torch.zeros((B, H, V, K), device=device, dtype=torch.float32),
                "occ": torch.zeros((B, H, K), device=device, dtype=torch.float32),
            }

        z_all = torch.cat(nonempty_z, dim=0)
        m_all_1l = torch.cat(nonempty_m_1l, dim=0).unsqueeze(0) if nonempty_m_1l else None
        k_all, v_all, g_all, b_all, _ = self.gdn.project_frame_slots(
            z_all.unsqueeze(0), motion_feat=m_all_1l
        )
        S_out, occ_out = self._scan_episode_prefix_states(
            k_all, v_all, g_all, b_all, ep_tok_offset_1l, counts_list, ep_ids_list, Nw, V, force_pytorch
        )
        return {"S": S_out, "occ": occ_out}

    # -------------------------------------------------------------------------
    # Readout Loop -> Coda Tokens `C_t^{(0..L-1)}`
    # -------------------------------------------------------------------------

    def run_coda_readout(
        self,
        ctx_tokens: torch.Tensor,                    # (B, 145, d)
        ctx_mask: torch.Tensor,                      # (B, 145) bool
        mem_state: Optional[Dict[str, torch.Tensor]] = None,
        *,
        num_loops: Optional[int] = None,
        halt_mode: Literal["fixed", "learned_gate", "coda_conv"] = "fixed",
        halt_threshold: float = 0.5,
        coda_conv_tau: float = 0.08,
    ) -> Dict[str, Any]:
        """Produce Coda tokens `(C_t, mem_mask)` and per-depth lists for training or adaptive eval."""
        B = ctx_tokens.shape[0]
        device = ctx_tokens.device
        dt = ctx_tokens.dtype

        if self.arch == "m0_memoryless":
            return {
                "coda": None,
                "coda_per_depth": [None],
                "mem_mask": None,
                "loop_depths": torch.ones((B,), device=device, dtype=torch.float32),
            }

        if self.arch == "m2_framesamp_modul":
            assert self.framesamp_refine is not None and self.coda is not None and mem_state is not None
            buf_tokens = mem_state["buf_tokens"].to(dt)  # (B, 32, d)
            buf_mask = mem_state["buf_mask"]             # (B, 32) bool
            refined = self.framesamp_refine(buf_tokens, buf_mask)
            c_out = self.coda(refined)
            # Zero out masked slots cleanly
            c_out = c_out * buf_mask.to(c_out.dtype).unsqueeze(-1)
            return {
                "coda": c_out,
                "coda_per_depth": [c_out],
                "mem_mask": buf_mask,
                "loop_depths": torch.ones((B,), device=device, dtype=torch.float32),
            }

        assert self.prelude is not None and self.core is not None and self.coda is not None
        h = self.prelude(ctx_tokens, ctx_mask)  # (B, N_c, d)

        if self.arch == "m5_gdn2layer_b2_l1":
            assert self.readout1 is not None and self.readout2 is not None and mem_state is not None
            e0 = self.core.get_loop_emb(0, dt)
            ro1 = self.readout1(h + e0, mem_state["S1"], mem_state.get("occ1"))
            ro2 = self.readout2(h + e0, mem_state["S2"], mem_state.get("occ2"))
            m0 = 0.5 * (ro1["m"] + ro2["m"])
            h1 = self.core(h, m0, ctx_tokens, ctx_mask, l_idx=0)
            c0 = self.coda(h1)
            return {
                "coda": c0,
                "coda_per_depth": [c0],
                "mem_mask": None,
                "loop_depths": torch.ones((B,), device=device, dtype=torch.float32),
            }

        L_max = (
            (num_loops if num_loops is not None else self.cfg.max_loops)
            if self.is_multi_depth
            else 1
        )
        H_gdn = self.cfg.gdn_heads

        coda_list: List[torch.Tensor] = []
        h_list: List[torch.Tensor] = []
        m_list: List[torch.Tensor] = []
        dh_norms: List[torch.Tensor] = []
        dc_norms: List[torch.Tensor] = []
        r_norms: List[torch.Tensor] = []
        coverages: List[torch.Tensor] = []

        # Fast true-early-exit path for single-env online rollout (`B == 1`)!
        can_early_break = (
            (not self.training)
            and B == 1
            and L_max > 1
            and halt_mode in ("learned_gate", "coda_conv")
            and (halt_mode == "coda_conv" or self.halting_gate is not None)
        )
        surv_b1 = torch.ones((1,), device=device, dtype=torch.float32)
        cdf_b1 = torch.zeros((1,), device=device, dtype=torch.float32)

        prev_c: Optional[torch.Tensor] = None
        for l_idx in range(L_max):
            e_l = self.core.get_loop_emb(l_idx, dt)
            if self.arch == "m1_rdvla_loop4":
                m_l = torch.zeros_like(h)
                r_norm = torch.zeros((B, H_gdn), device=device, dtype=torch.float32)
                cov = torch.zeros((B, H_gdn), device=device, dtype=torch.float32)
            elif self.arch == "m7_deltaloop_2l_b2":
                assert self.readout1 is not None and self.readout2 is not None and mem_state is not None
                ro1 = self.readout1(h + e_l, mem_state["S1"], mem_state.get("occ1"))
                ro2 = self.readout2(h + e_l, mem_state["S2"], mem_state.get("occ2"))
                m_l = 0.5 * (ro1["m"] + ro2["m"])
                r_norm = 0.5 * (ro1["r_norm"] + ro2["r_norm"])
                cov = 0.5 * (ro1["coverage"] + ro2["coverage"])
            else:
                assert self.readout is not None and mem_state is not None
                ro = self.readout(h + e_l, mem_state["S"], mem_state.get("occ"))
                m_l = ro["m"]
                r_norm = ro["r_norm"]
                cov = ro["coverage"]

            h_next = self.core(h, m_l, ctx_tokens, ctx_mask, l_idx=l_idx)
            c_next = self.coda(h_next)

            dh = torch.linalg.vector_norm((h_next - h).float(), ord=2, dim=-1).mean(dim=-1, keepdim=True)
            if prev_c is None:
                dc = torch.linalg.vector_norm(c_next.float(), ord=2, dim=-1).mean(dim=-1, keepdim=True)
                rel_dc = torch.ones((B,), device=device, dtype=torch.float32)
            else:
                dc = torch.linalg.vector_norm((c_next - prev_c).float(), ord=2, dim=-1).mean(dim=-1, keepdim=True)
                prev_norm = torch.linalg.vector_norm(prev_c.float(), ord=2, dim=-1).mean(dim=-1)
                rel_dc = dc.squeeze(-1) / (prev_norm + 1e-5)

            coda_list.append(c_next)
            h_list.append(h_next)
            m_list.append(m_l)
            dh_norms.append(dh)
            dc_norms.append(dc)
            r_norms.append(r_norm)
            coverages.append(cov)

            h = h_next
            prev_c = c_next

            if can_early_break:
                if halt_mode == "learned_gate" and self.halting_gate is not None:
                    lam_l = self.halting_gate.step_hazard(
                        h_next, m_l, dh, dc, r_norm, cov, l_idx, L_max
                    )
                    p_l = lam_l * surv_b1
                    cdf_b1 = cdf_b1 + p_l
                    surv_b1 = surv_b1 * (1.0 - lam_l).clamp(min=1e-7)
                    if float(cdf_b1.item()) >= halt_threshold or l_idx == L_max - 1:
                        return {
                            "coda": c_next,
                            "coda_per_depth": coda_list,
                            "mem_mask": None,
                            "loop_depths": torch.full((1,), float(l_idx + 1), device=device),
                        }
                elif halt_mode == "coda_conv" and l_idx >= 1:
                    if float(rel_dc.item()) < coda_conv_tau or l_idx == L_max - 1:
                        return {
                            "coda": c_next,
                            "coda_per_depth": coda_list,
                            "mem_mask": None,
                            "loop_depths": torch.full((1,), float(l_idx + 1), device=device),
                        }

        halt_out = None
        chosen_coda = coda_list[-1]
        loop_depths = torch.full((B,), float(L_max), device=device, dtype=torch.float32)

        if self.halting_gate is not None:
            halt_out = self.halting_gate.forward_all(
                h_list, m_list, dh_norms, dc_norms, r_norms, coverages
            )
            if halt_mode == "learned_gate" and L_max > 1:
                p_tilde = halt_out["p_tilde"]  # (B, L_max)
                cdf = torch.cumsum(p_tilde, dim=1)
                hit = cdf >= halt_threshold
                hit[:, -1] = True
                chosen_idx = torch.argmax(hit.int(), dim=1)  # (B,)
                c_stack = torch.stack(coda_list, dim=1)  # (B, L_max, N_c, d)
                gather_idx = chosen_idx.view(B, 1, 1, 1).expand(B, 1, c_stack.shape[2], c_stack.shape[3])
                chosen_coda = torch.gather(c_stack, dim=1, index=gather_idx).squeeze(1)
                loop_depths = (chosen_idx + 1).float()

        if halt_mode == "coda_conv" and L_max > 1:
            c_stack = torch.stack(coda_list, dim=1)  # (B, L_max, N_c, d)
            hit_list = [torch.zeros((B,), device=device, dtype=torch.bool)]
            for l_i in range(1, L_max):
                diff_n = torch.linalg.vector_norm(
                    (c_stack[:, l_i] - c_stack[:, l_i - 1]).float(), ord=2, dim=-1
                ).mean(dim=-1)
                base_n = torch.linalg.vector_norm(
                    c_stack[:, l_i - 1].float(), ord=2, dim=-1
                ).mean(dim=-1)
                hit_list.append((diff_n / (base_n + 1e-5)) < coda_conv_tau)
            hit_mat = torch.stack(hit_list, dim=1)
            hit_mat[:, -1] = True
            chosen_idx = torch.argmax(hit_mat.int(), dim=1)
            gather_idx = chosen_idx.view(B, 1, 1, 1).expand(B, 1, c_stack.shape[2], c_stack.shape[3])
            chosen_coda = torch.gather(c_stack, dim=1, index=gather_idx).squeeze(1)
            loop_depths = (chosen_idx + 1).float()

        return {
            "coda": chosen_coda,
            "coda_per_depth": coda_list,
            "mem_mask": None,
            "loop_depths": loop_depths,
            "halt": halt_out,
        }

    # -------------------------------------------------------------------------
    # Cycle 3 `U4` Grounding Head & CRN Multi-Depth Flow + Aux Loss
    # -------------------------------------------------------------------------

    def _pool_grounding_feature(
        self,
        ctx_tokens: torch.Tensor,
        coda_tokens: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse visual context pool with depth-`l` Coda pool (`0` on `m0_memoryless` for exact fairness)."""
        vis_n = min(80, int(ctx_tokens.shape[1]))
        ctx_pool = ctx_tokens[:, :vis_n, :].float().mean(dim=1)  # (B, d)
        if coda_tokens is None:
            return ctx_pool
        if mem_mask is not None:
            mf = mem_mask.float().unsqueeze(-1)
            c_pool = (coda_tokens.float() * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
        else:
            # Handle masked zero rows cleanly (e.g. when `m2_framesamp_modul` passes pre-zeroed coda without mask)
            nz = (coda_tokens.float().abs().sum(dim=-1, keepdim=True) > 1e-12).float()
            denom = nz.sum(dim=1).clamp(min=1.0)
            c_pool = (coda_tokens.float() * nz).sum(dim=1) / denom
        return ctx_pool + c_pool

    def predict_grounding_xy(
        self,
        ctx_tokens: torch.Tensor,
        coda_tokens: Optional[torch.Tensor] = None,
        *,
        ctx_mask: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Predict canonical `(x_norm, y_norm)` in `[-1, 1]^2` (`None` on Cycle 2A models without `U4`)."""
        _ = ctx_mask
        if self.grounding_xy_head is None:
            return None
        dt = self.grounding_xy_head[1].weight.dtype  # type: ignore[index]
        feat = self._pool_grounding_feature(ctx_tokens, coda_tokens, mem_mask=mem_mask).to(dt)
        return 1.25 * torch.tanh(self.grounding_xy_head(feat).float())

    def compute_crn_flow_and_aux_loss(
        self,
        ctx_tokens: torch.Tensor,                        # (B, N_ctx, d)
        ctx_mask: torch.Tensor,                          # (B, N_ctx) bool
        coda_per_depth: List[Optional[torch.Tensor]],    # len L, each `(B, N_c, d)` or `None`
        mem_mask: Optional[torch.Tensor],                # Optional `(B, N_c)` bool
        target_actions: torch.Tensor,                    # (B, H_act, 8) float32
        aux_targets: Optional[torch.Tensor] = None,      # (B, 49) float32
        eta_aux: float = 0.1,
        tau_override: Optional[torch.Tensor] = None,     # Optional `(B,)`
        noise_override: Optional[torch.Tensor] = None,   # Optional `(B, H_act, 8)`
        grounding_xy_target: Optional[torch.Tensor] = None,   # Optional `(B, 2)` float32 in `[-1, 1]^2`
        grounding_xy_mask: Optional[torch.Tensor] = None,     # Optional `(B,)` bool
        subgoal_emb_target: Optional[torch.Tensor] = None,    # Optional `(B, 64)` unit float32
        subgoal_emb_mask: Optional[torch.Tensor] = None,      # Optional `(B,)` bool
    ) -> Dict[str, torch.Tensor]:
        """Compute CRN-shared Flow-Matching velocity MSE across depths `l = 0 .. L-1` + `eta_aux * L_aux`."""
        B = target_actions.shape[0]
        device = target_actions.device
        x1 = target_actions.float()

        if tau_override is not None:
            tau = tau_override.float()
        elif self.cfg.v3_mode:
            # Cycle 3 `U1`: openpi / SmolVLA noise-weighted schedule `tau = 1 - (0.999*u + 0.001), u ~ Beta(1.5, 1.0)`
            # Exact inverse-CDF of Beta(1.5, 1.0): `u = r^(1 / 1.5)` for `r ~ Uniform(0, 1)`!
            u_beta = torch.rand((B,), device=device, dtype=torch.float32).clamp(min=1e-7).pow(1.0 / 1.5)
            tau = 1.0 - (0.999 * u_beta + 0.001)
        else:
            tau = torch.rand((B,), device=device, dtype=torch.float32)

        eps = (
            noise_override.float()
            if noise_override is not None
            else torch.randn_like(x1)
        )
        t_view = tau.view(B, 1, 1)
        x_tau = (1.0 - t_view) * eps + t_view * x1
        u_target = x1 - eps  # (B, H_act, 8)

        use_grip = self.action_expert.gripper_head is not None and self.cfg.gripper_bce_weight > 0.0
        grip_target = (x1[:, : self.cfg.exec_horizon, 7] >= 0.0).float() if use_grip else None
        w_u1 = self._u1_loss_weights

        L = len(coda_per_depth)
        grip_loss = torch.zeros((), device=device, dtype=torch.float32)
        if L == 1:
            if use_grip:
                v_pred, grip_logits = self.action_expert(
                    x_tau,
                    tau,
                    ctx_tokens,
                    ctx_mask,
                    mem_tokens=coda_per_depth[0],
                    mem_mask=mem_mask,
                    return_gripper_logits=True,
                )
            else:
                v_pred = self.action_expert(
                    x_tau, tau, ctx_tokens, ctx_mask, mem_tokens=coda_per_depth[0], mem_mask=mem_mask
                )
                grip_logits = None

            sq_err = (v_pred - u_target) ** 2
            if w_u1 is not None:
                per_sample_l0 = torch.sum(sq_err * w_u1.unsqueeze(0), dim=(1, 2))
            else:
                per_sample_l0 = torch.mean(sq_err, dim=(1, 2))  # (B,)

            if grip_logits is not None and grip_target is not None:
                g_bce_ps = F.binary_cross_entropy_with_logits(
                    grip_logits, grip_target, reduction="none"
                ).mean(dim=-1)
                grip_loss = g_bce_ps.mean()
                per_sample_l0 = per_sample_l0 + self.cfg.gripper_bce_weight * g_bce_ps

            fm_loss = per_sample_l0.mean()
            per_depth_sample_loss = per_sample_l0.unsqueeze(1)  # (B, 1)
        else:
            # Fuse all L depths into ONE batched FlowActionExpert forward pass with CRN!
            c_cat = torch.cat([c for c in coda_per_depth if c is not None], dim=0)  # (L * B, N_c, d)
            x_rep = x_tau.repeat(L, 1, 1)
            tau_rep = tau.repeat(L)
            ctx_rep = ctx_tokens.repeat(L, 1, 1)
            mask_rep = ctx_mask.repeat(L, 1)
            mm_rep = mem_mask.repeat(L, 1) if mem_mask is not None else None

            if use_grip:
                v_flat, grip_flat = self.action_expert(
                    x_rep,
                    tau_rep,
                    ctx_rep,
                    mask_rep,
                    mem_tokens=c_cat,
                    mem_mask=mm_rep,
                    return_gripper_logits=True,
                )
            else:
                v_flat = self.action_expert(
                    x_rep, tau_rep, ctx_rep, mask_rep, mem_tokens=c_cat, mem_mask=mm_rep
                )
                grip_flat = None

            v_all = v_flat.view(L, B, self.cfg.action_horizon, self.cfg.action_dim)
            sq_all = (v_all - u_target.unsqueeze(0)) ** 2
            if w_u1 is not None:
                per_depth_sample_loss = torch.sum(
                    sq_all * w_u1.view(1, 1, self.cfg.action_horizon, self.cfg.action_dim), dim=(2, 3)
                ).transpose(0, 1)  # (B, L)
            else:
                per_depth_sample_loss = torch.mean(sq_all, dim=(2, 3)).transpose(0, 1)  # (B, L)

            if grip_flat is not None and grip_target is not None:
                g_all = grip_flat.view(L, B, self.cfg.exec_horizon)
                g_bce = F.binary_cross_entropy_with_logits(
                    g_all, grip_target.unsqueeze(0).expand(L, -1, -1), reduction="none"
                ).mean(dim=-1).transpose(0, 1)  # (B, L)
                grip_loss = g_bce.mean()
                per_depth_sample_loss = per_depth_sample_loss + self.cfg.gripper_bce_weight * g_bce

            fm_loss = per_depth_sample_loss.mean()

        aux_loss = torch.zeros((), device=device, dtype=torch.float32)
        if self.aux_head is not None and aux_targets is not None and coda_per_depth[0] is not None:
            aux_losses = []
            for c_l in coda_per_depth:
                assert c_l is not None
                if mem_mask is not None:
                    mf = mem_mask.to(c_l.dtype).unsqueeze(-1)
                    c_pool = (c_l * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                else:
                    c_pool = c_l.mean(dim=1)
                pred_aux = self.aux_head(c_pool).float()
                aux_losses.append(F.mse_loss(pred_aux, aux_targets.float()))
            aux_loss = torch.stack(aux_losses).mean()

        ground_loss = torch.zeros((), device=device, dtype=torch.float32)
        if (
            eta_aux > 0.0
            and self.grounding_xy_head is not None
            and (grounding_xy_target is not None or subgoal_emb_target is not None)
        ):
            dt_g = self.grounding_xy_head[1].weight.dtype  # type: ignore[index]
            g_losses = []
            xy_tgt = grounding_xy_target.float() if grounding_xy_target is not None else None
            xy_m = (
                grounding_xy_mask.float()
                if grounding_xy_mask is not None
                else (torch.ones((B,), device=device, dtype=torch.float32) if xy_tgt is not None else None)
            )
            sg_tgt = subgoal_emb_target.float() if subgoal_emb_target is not None else None
            sg_m = (
                subgoal_emb_mask.float()
                if subgoal_emb_mask is not None
                else (torch.ones((B,), device=device, dtype=torch.float32) if sg_tgt is not None else None)
            )
            for c_l in coda_per_depth:
                feat_l = self._pool_grounding_feature(ctx_tokens, c_l, mem_mask=mem_mask).to(dt_g)
                l_val = torch.zeros((), device=device, dtype=torch.float32)
                if xy_tgt is not None and xy_m is not None:
                    pred_xy = 1.25 * torch.tanh(self.grounding_xy_head(feat_l).float())
                    sl1 = F.smooth_l1_loss(pred_xy, xy_tgt, reduction="none", beta=0.05).mean(dim=-1)
                    if bool((xy_m > 0.5).any()):
                        l_val = l_val + (sl1 * xy_m).sum() / xy_m.sum().clamp(min=1.0)
                    else:
                        l_val = l_val + 0.0 * pred_xy.sum()
                if sg_tgt is not None and sg_m is not None and self.grounding_sg_head is not None:
                    pred_sg = self.grounding_sg_head(feat_l).float()
                    cos_l = 0.5 * (1.0 - F.cosine_similarity(pred_sg, sg_tgt, dim=-1, eps=1e-6))
                    if bool((sg_m > 0.5).any()):
                        l_val = l_val + (cos_l * sg_m).sum() / sg_m.sum().clamp(min=1.0)
                    else:
                        l_val = l_val + 0.0 * pred_sg.sum()
                g_losses.append(l_val)
            ground_loss = torch.stack(g_losses).mean()

        total_loss = fm_loss + eta_aux * aux_loss + self.cfg.grounding_loss_weight * ground_loss
        return {
            "loss": total_loss,
            "fm_loss": fm_loss,
            "aux_loss": aux_loss,
            "ground_loss": ground_loss,
            "grip_loss": grip_loss,
            "per_depth_fm_loss": per_depth_sample_loss,  # (B, L)
        }

    # -------------------------------------------------------------------------
    # Flow-Matching ODE Sampler (Inference: Euler / 2nd-Order Midpoint)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sample_action_chunk(
        self,
        ctx_tokens: torch.Tensor,                   # (B, N_ctx, d)
        ctx_mask: torch.Tensor,                     # (B, N_ctx) bool
        mem_tokens: Optional[torch.Tensor] = None,  # (B, N_c, d)
        mem_mask: Optional[torch.Tensor] = None,    # (B, N_c) bool
        *,
        ode_steps: int = 10,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Integrate `dx/dtau = v_theta(x, tau | E_t, C_t)` from `tau=0` to `tau=1`."""
        B = ctx_tokens.shape[0]
        device = ctx_tokens.device
        x = torch.randn(
            (B, self.cfg.action_horizon, self.cfg.action_dim),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        solver = str(getattr(self.cfg, "ode_solver", "euler")).lower()
        clamp_lim = float(getattr(self.cfg, "sample_clamp", 1.5))

        if solver == "midpoint" and int(ode_steps) >= 2:
            # 5 Midpoint steps == exact 10 NFE (identical chunk latency to 10 Euler steps, O(dt^3) local accuracy!)
            n_sub = max(1, int(ode_steps) // 2)
            dt_step = 1.0 / float(n_sub)
            for s_i in range(n_sub):
                t0 = torch.full((B,), s_i * dt_step, device=device, dtype=torch.float32)
                t_mid = torch.full((B,), (s_i + 0.5) * dt_step, device=device, dtype=torch.float32)
                v1 = self.action_expert(
                    x, t0, ctx_tokens, ctx_mask, mem_tokens=mem_tokens, mem_mask=mem_mask
                )
                x_mid = x + 0.5 * dt_step * v1
                v2 = self.action_expert(
                    x_mid, t_mid, ctx_tokens, ctx_mask, mem_tokens=mem_tokens, mem_mask=mem_mask
                )
                x = x + dt_step * v2
            return x.clamp(-clamp_lim, clamp_lim)

        dt_step = 1.0 / max(1, int(ode_steps))
        for s_i in range(max(1, int(ode_steps))):
            tau = torch.full((B,), s_i * dt_step, device=device, dtype=torch.float32)
            v = self.action_expert(
                x, tau, ctx_tokens, ctx_mask, mem_tokens=mem_tokens, mem_mask=mem_mask
            )
            x = x + dt_step * v
        return x.clamp(-clamp_lim, clamp_lim)
