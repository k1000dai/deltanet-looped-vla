"""Pretrained `SmolVLA-DeltaLoop` Visuomotor Flow-Matching Policy Suite (`RESEARCH_PLAN.md` v2.3 §3.7 `U6`).

Self-contained PyTorch implementation of `lerobot/smolvla_base` (`SmolVLM2-500M-Video-Instruct`
truncated to 16 VLM layers + pretrained `98.25M` 16-layer interleaved Flow-Matching Action Expert)
augmented with our controlled memory & dynamic multi-hop readout suite (`docs/PRETRAINED_VLA_DESIGN.md`):

- **Zero `lerobot` or `AutoProcessor` Runtime Dependency:**
  Uses ONLY `torch`, `transformers` (`AutoConfig`, `AutoModel`, `AutoTokenizer`,
  `SmolVLMForConditionalGeneration`), `safetensors.torch`, and verified memory/loop modules from
  `src/models/deltaloop_vla.py` (`100%` offline-compatible on both Miyabi GH200 and local RTX 5080!).
- **Exact Bit-for-Bit Step-0 Pretrained Equivalence (`0.0` Diff):**
  * Custom split-half `apply_rope(..., max_wavelength=10_000)` on both VLM and Action Expert layers.
  * Exact `embed_prefix` (`sqrt(960)` image & language scaling, `state` `att_mask=1`) and
    `embed_suffix` (`att_masks = [1] * chunk_size`, `min_period=4e-3, max_period=4.0`,
    `t = 1.0` pure noise -> `t = 0.0` clean action, `u_t = eps - actions`).
  * Immutable `dict[int, dict[str, Tensor]]` prefix KV cache (`prefill_vlm_kv_from_tokens`) reused
    out-of-place across all `L=1..4` multi-hop depths, `M_CRN=8` Stage-II halting draws, and `10`
    Euler ODE steps (`0.0` diff vs joint VLM+Expert forward!).
  * Dual zero-init `SmolVLAExpertMemoryHook` on all 8 odd VLM-KV cross-attention expert layers
    (`1, 3, 5, 7, 9, 11, 13, 15`) with `mem_o_proj = 0` + `(1 + tanh(gamma_ctx))` + zero-init FiLM.

- **Controlled 5-Model Pretrained Suite (`ArchSmolVLA`):**
  * `smolvla_m0_base`:       Pretrained Markov `SmolVLA` baseline (`0 B` state, `L=1`).
  * `smolvla_m1_rdvla_loop4`: Memoryless Looped Core (`0 B` state, `L=4` — `RD-VLA` control).
  * `smolvla_m2_framesamp`:  `32`-slot Perceiver Frame-Sampling buffer (`24 KB`, `L=1`).
  * `smolvla_m5_gdn2l_l1`:   Iso-state `512 KB` 2-Layer Motion-Gated Neg-Eig GDN (`L=1`).
  * `smolvla_m7_deltaloop`:  **Full `SmolVLA-DeltaLoop` (Ours: `512 KB` 2-Layer Motion-Gated Neg-Eig
                             GDN + Weight-Tied Multi-Hop Readout `L=1..4` + Stage-II `M_CRN=8` Gate)**.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from analysis.subgoal_grounding import (
        apply_subgrid_affine_to_tokens,
        sample_subgrid_crop_affine,
    )
except ImportError:
    import sys as _sys

    _repo_root = str(Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    from analysis.subgoal_grounding import (
        apply_subgrid_affine_to_tokens,
        sample_subgrid_crop_affine,
    )

try:
    from robomme_eval.interface import DELTA_JOINT_MASK, MME_VLA_NORM_STATS
except ImportError:
    from src.robomme_eval.interface import DELTA_JOINT_MASK, MME_VLA_NORM_STATS

from .deltaloop_vla import (
    CRNHaltingGate,
    ContextTransformerBlock,
    DampedSlotMemoryReadout,
    DeltaLoopVLAConfig,
    DeltaLoopVLAPolicy,
    PerceiverSlotResampler,
    SharedObservationEncoder,
    SlotGDNMemoryWriter,
    SwiGLU,
    VLACoda,
    VLAPrelude,
    VLARecurrentCore,
    run_dual_backend_gdn,
    safe_masked_sdpa,
)

SMOLVLM2_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_MIYABI_SMOLVLA_SAFETENSORS = (
    "/work/gj26/b20098/hub/models--lerobot--smolvla_base/"
    "snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
)

ArchSmolVLA = Literal[
    "smolvla_m0_base",
    "smolvla_m1_rdvla_loop4",
    "smolvla_m2_framesamp",
    "smolvla_m5_gdn2l_l1",
    "smolvla_m7_deltaloop",
]

ALL_SMOLVLA_ARCHS: Tuple[ArchSmolVLA, ...] = (
    "smolvla_m0_base",
    "smolvla_m1_rdvla_loop4",
    "smolvla_m2_framesamp",
    "smolvla_m5_gdn2l_l1",
    "smolvla_m7_deltaloop",
)


# =============================================================================
# 1. Exact `SmolVLA` Mathematical Primitives (`apply_rope`, Masks, `MEAN_STD`)
# =============================================================================


def apply_rope(x: torch.Tensor, positions: torch.Tensor, max_wavelength: int = 10_000) -> torch.Tensor:
    """Exact split-half RoPE from `lerobot.policies.smolvla.smolvlm_with_expert` (`max_wavelength=10_000`)."""
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x_f32 = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = float(max_wavelength) ** freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x_f32.split(d_half, dim=-1)
    res = torch.empty_like(x_f32)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)


def get_intermediate_size(hidden_dim: int, ffn_dim_multiplier: int = 4, multiple_of: int = 256) -> int:
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Exact float64 sinusoidal flow-time embedding from `modeling_smolvla.py`."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError(f"Expected 1D time `(B,)`, got shape {tuple(time.shape)}")

    dev = torch.device(device)
    # MPS lacks float64; CPU and CUDA both support float64 natively
    f_dtype = torch.float32 if dev.type == "mps" else torch.float64
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=f_dtype, device=dev)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = (1.0 / period) * (2.0 * math.pi)
    sin_input = scaling_factor[None, :] * time.to(device=dev, dtype=f_dtype)[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks: torch.Tensor, att_masks: torch.Tensor) -> torch.Tensor:
    """Exact block-causal / prefix-LM 2D attention mask from `modeling_smolvla.py`."""
    if att_masks.ndim != 2 or pad_masks.ndim != 2:
        raise ValueError(f"Expected 2D masks, got pad={pad_masks.ndim}, att={att_masks.ndim}")
    cumsum = torch.cumsum(att_masks.to(dtype=torch.int64), dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :].bool() & pad_masks[:, :, None].bool()
    return att_2d & pad_2d


def pad_vector(vector: torch.Tensor, new_dim: int) -> torch.Tensor:
    """Zero-pad trailing feature dimension to `new_dim` (`32` in `smolvla_base`)."""
    cur = int(vector.shape[-1])
    if cur == new_dim:
        return vector
    if cur > new_dim:
        return vector[..., :new_dim]
    out = torch.zeros((*vector.shape[:-1], new_dim), dtype=vector.dtype, device=vector.device)
    out[..., :cur] = vector
    return out


def resize_with_pad_smolvla(
    img: torch.Tensor, width: int = 512, height: int = 512, pad_value: float = 0.0
) -> torch.Tensor:
    """Exact `modeling_smolvla.resize_with_pad` on `(B, C, H, W)` float `[0, 1]` images."""
    if img.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W), got {tuple(img.shape)}")
    cur_h, cur_w = int(img.shape[2]), int(img.shape[3])
    if (cur_h, cur_w) == (height, width):
        return img
    ratio = max(cur_w / float(width), cur_h / float(height))
    resized_h = int(cur_h / ratio)
    resized_w = int(cur_w / ratio)
    resized = F.interpolate(img, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    pad_h = max(0, int(height - resized_h))
    pad_w = max(0, int(width - resized_w))
    if pad_h == 0 and pad_w == 0:
        return resized
    return F.pad(resized, (pad_w, 0, pad_h, 0), value=pad_value)


def preprocess_smolvla_rgb(
    images: np.ndarray | torch.Tensor,
    *,
    image_size: Tuple[int, int] = (512, 512),
    device: Optional[torch.device | str] = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convert `(H, W, 3)` or `(N, H, W, 3)` `uint8`/`float` RGB into `[-1, 1]` `(N, 3, 512, 512)`."""
    x = torch.from_numpy(np.ascontiguousarray(images)) if isinstance(images, np.ndarray) else images
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D RGB array, got shape {tuple(x.shape)}")
    if x.shape[-1] == 3 and x.shape[1] != 3:
        x = x.permute(0, 3, 1, 2)
    if device is not None:
        x = x.to(device=device, non_blocking=True)
    if x.dtype == torch.uint8:
        x = x.to(torch.float32) / 255.0
    else:
        x = x.to(torch.float32)
        if x.max() > 1.0 + 1e-4:
            x = x / 255.0
    h_tgt, w_tgt = image_size
    x = resize_with_pad_smolvla(x, width=w_tgt, height=h_tgt, pad_value=0.0)
    x = x * 2.0 - 1.0
    return x.to(dtype=out_dtype)


def pool_vlm_tokens_8x8_to_4x4(tokens: torch.Tensor) -> torch.Tensor:
    """Pool `(..., 64, D)` (`8x8`) VLM connector tokens to `(..., 16, D)` (`4x4`) wrist grid."""
    if tokens.shape[-2] == 16:
        return tokens
    if tokens.shape[-2] != 64:
        raise ValueError(f"Expected 64 or 16 spatial tokens, got {tokens.shape[-2]}")
    *lead, _, d = tokens.shape
    grid = tokens.reshape(*lead, 4, 2, 4, 2, d)
    return grid.float().mean(dim=(-4, -2)).to(tokens.dtype).reshape(*lead, 16, d)


def format_smolvla_prompt(prompt: str) -> str:
    """Append trailing newline expected by `lerobot/smolvla_base` tokenizer processor."""
    return str(prompt).rstrip("\n") + "\n"


class MeanStdDeltaNormalizer:
    """`MEAN_STD` state & `(20, 8)` delta-arm/abs-gripper action normalizer for `SmolVLA`."""

    def __init__(
        self,
        stats: Optional[Dict[str, Any]] = None,
        *,
        binarize_gripper_on_decode: bool = True,
    ) -> None:
        src = MME_VLA_NORM_STATS if stats is None else stats
        st = src["state"]
        ac = src.get("actions", src.get("action"))
        if ac is None:
            raise KeyError("Expected 'actions' in stats dict")

        self.state_mean = np.asarray(st["mean"], dtype=np.float32).reshape(8)
        self.state_std = np.maximum(np.asarray(st["std"], dtype=np.float32).reshape(8), 1e-4)

        a_mean = np.asarray(ac["mean"], dtype=np.float32)
        a_std = np.maximum(np.asarray(ac["std"], dtype=np.float32), 1e-4)
        if a_mean.ndim == 1:
            a_mean = a_mean.reshape(1, 8)
            a_std = a_std.reshape(1, 8)
        elif a_mean.ndim == 2 and a_mean.shape[1] == 8:
            pass
        else:
            raise ValueError(f"Unexpected action mean shape: {a_mean.shape}")
        self.action_mean = a_mean.astype(np.float32)
        self.action_std = a_std.astype(np.float32)
        self.delta_mask = np.asarray(DELTA_JOINT_MASK, dtype=bool)
        self.binarize_gripper_on_decode = bool(binarize_gripper_on_decode)

    def to_stats_dict(self) -> Dict[str, Any]:
        return {
            "state": {
                "mean": self.state_mean.tolist(),
                "std": self.state_std.tolist(),
            },
            "actions": {
                "mean": self.action_mean.tolist(),
                "std": self.action_std.tolist(),
            },
        }

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        st = np.asarray(state, dtype=np.float32)
        return ((st - self.state_mean) / (self.state_std + 1e-6)).astype(np.float32)

    def _slice_action_stats(self, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.action_mean.shape[0] == 1:
            return self.action_mean, self.action_std
        if horizon <= self.action_mean.shape[0]:
            return self.action_mean[:horizon], self.action_std[:horizon]
        pad_n = horizon - self.action_mean.shape[0]
        m = np.concatenate([self.action_mean, np.repeat(self.action_mean[-1:], pad_n, axis=0)], axis=0)
        s = np.concatenate([self.action_std, np.repeat(self.action_std[-1:], pad_n, axis=0)], axis=0)
        return m, s

    def encode_actions(self, raw_actions: np.ndarray, raw_state_at_step: np.ndarray) -> np.ndarray:
        act = np.asarray(raw_actions, dtype=np.float32).copy()
        st = np.asarray(raw_state_at_step, dtype=np.float32).reshape(8)
        act[..., self.delta_mask] -= st[self.delta_mask]
        m, s = self._slice_action_stats(act.shape[-2] if act.ndim >= 2 else 1)
        if act.ndim == 1:
            m, s = m[0], s[0]
        return ((act - m) / (s + 1e-6)).astype(np.float32)

    def decode_actions(self, norm_actions: np.ndarray, raw_state_at_step: np.ndarray) -> np.ndarray:
        z = np.asarray(norm_actions, dtype=np.float32)[..., :8]
        st = np.asarray(raw_state_at_step, dtype=np.float32).reshape(8)
        m, s = self._slice_action_stats(z.shape[-2] if z.ndim >= 2 else 1)
        if z.ndim == 1:
            m, s = m[0], s[0]
        act = z * (s + 1e-6) + m
        act[..., self.delta_mask] += st[self.delta_mask]
        if self.binarize_gripper_on_decode:
            act[..., 7] = np.where(act[..., 7] >= 0.0, 1.0, -1.0)
        return act.astype(np.float32)


# =============================================================================
# 2. Configuration & Zero-Init Action-Expert Memory Hook
# =============================================================================


@dataclass
class SmolVLADeltaLoopConfig:
    """Complete serializable configuration for `SmolVLADeltaLoopPolicy` (`U6`)."""

    arch: ArchSmolVLA = "smolvla_m7_deltaloop"
    vlm_model_name: str = SMOLVLM2_MODEL_ID
    num_vlm_layers: int = 16
    num_expert_layers: int = 16
    self_attn_every_n_layers: int = 2
    expert_width_multiplier: float = 0.75

    max_state_dim: int = 32
    max_action_dim: int = 32
    state_dim: int = 8
    action_dim: int = 8
    chunk_size: int = 20
    exec_horizon: int = 16
    exclusion_window: int = 16
    write_stride: int = 4
    long_demo_stride: int = 8
    adaptive_demo_stride_threshold: int = 256

    tokenizer_max_length: int = 48
    image_size: int = 512
    num_ode_steps: int = 10
    min_period: float = 4e-3
    max_period: float = 4.0
    train_state_proj: bool = True
    freeze_vlm: bool = True

    # Memory, Perceiver Resampler, GDN (`512 KB` iso-state), & Multi-Hop Loop parameters
    d_mem: int = 384
    mem_heads: int = 6
    mem_ff: int = 1024
    ctx_layers: int = 1
    num_write_slots: int = 8
    num_coda_tokens: int = 4
    slot_conv_kernel: int = 4
    framesamp_buffer_size: int = 32

    gdn_heads: int = 8
    gdn_d_k: int = 128
    gdn_d_v: int = 128
    min_halflife_frames: float = 8.0
    max_halflife_frames: float = 512.0
    readout_eps: float = 0.1
    use_motion_beta_gate: bool = True
    v3_beta_bias: float = -2.5

    max_loops: int = 4
    anchor_hop0_coda: bool = False
    aux_dim: int = 49
    subgoal_dim: int = 64

    # Shared U1–U4 perception & action-loss upgrades
    obs_horizon2: bool = True
    use_spatial_softmax: bool = True
    spatial_kp_front: int = 32
    spatial_kp_wrist: int = 16
    subgrid_aug: bool = True
    subgrid_scale_min: float = 0.90
    subgrid_max_shift: float = 0.04
    patch_dropout: float = 0.05
    feat_noise_std: float = 0.015

    use_gripper_head: bool = True
    gripper_bce_weight: float = 0.05
    use_grounding_head: bool = True
    grounding_loss_weight: float = 0.15
    use_u1_loss_weights: bool = True
    hook_dropout: float = 0.05
    sample_clamp: float = 6.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SmolVLADeltaLoopConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_mem_helper_config(self) -> DeltaLoopVLAConfig:
        """Build matching `DeltaLoopVLAConfig(vis_dim=960)` for shared memory/loop submodules."""
        mapped_arch = {
            "smolvla_m0_base": "m0_memoryless",
            "smolvla_m1_rdvla_loop4": "m1_rdvla_loop4",
            "smolvla_m2_framesamp": "m2_framesamp_modul",
            "smolvla_m5_gdn2l_l1": "m5_gdn2layer_b2_l1",
            "smolvla_m7_deltaloop": "m7_deltaloop_2l_b2",
        }[self.arch]
        return DeltaLoopVLAConfig(
            arch=mapped_arch,  # type: ignore[arg-type]
            vis_dim=960,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_horizon=self.chunk_size,
            exec_horizon=self.exec_horizon,
            exclusion_window=self.exclusion_window,
            write_stride=self.write_stride,
            d_model=self.d_mem,
            attn_heads=self.mem_heads,
            d_ff=self.mem_ff,
            ctx_layers=self.ctx_layers,
            expert_layers=1,
            num_write_slots=self.num_write_slots,
            num_coda_tokens=self.num_coda_tokens,
            slot_conv_kernel=self.slot_conv_kernel,
            framesamp_buffer_size=self.framesamp_buffer_size,
            gdn_heads=self.gdn_heads,
            gdn_d_k=self.gdn_d_k,
            gdn_d_v=self.gdn_d_v,
            min_halflife_frames=self.min_halflife_frames,
            max_halflife_frames=self.max_halflife_frames,
            readout_eps=self.readout_eps,
            max_loops=self.max_loops,
            aux_dim=self.aux_dim,
            v3_mode=True,
            obs_horizon2=self.obs_horizon2,
            use_spatial_softmax=self.use_spatial_softmax,
            spatial_kp_front=self.spatial_kp_front,
            spatial_kp_wrist=self.spatial_kp_wrist,
            # Sub-grid augmentation is applied once at the top level so VLM prefix & E_t stay synced!
            subgrid_aug=False,
            subgrid_scale_min=self.subgrid_scale_min,
            subgrid_max_shift=self.subgrid_max_shift,
            patch_dropout=0.0,
            feat_noise_std=0.0,
            use_gripper_head=self.use_gripper_head,
            gripper_bce_weight=self.gripper_bce_weight,
            use_grounding_head=self.use_grounding_head,
            grounding_loss_weight=self.grounding_loss_weight,
            subgoal_dim=self.subgoal_dim,
            adaptive_demo_stride_threshold=self.adaptive_demo_stride_threshold,
            long_demo_stride=self.long_demo_stride,
            use_motion_beta_gate=self.use_motion_beta_gate,
            v3_beta_bias=self.v3_beta_bias,
            sample_clamp=self.sample_clamp,
        )


class SmolVLAExpertMemoryHook(nn.Module):
    """Zero-init dual memory injection into an odd `SmolVLA` Action Expert layer (`d_exp=720, d_mem=384`).

    Guarantees exact `0.0` output delta at initialization (preserving bit-for-bit `smolvla_m0_base`
    equivalence at Step 0 even on all-`False` mask rows via `safe_masked_sdpa`), while providing
    immediate `1.0x` outer-product gradients on Step 1 via `mem_o_proj = 0` + `(1 + tanh(gamma_ctx))`.
    """

    def __init__(self, d_exp: int = 720, d_mem: int = 384, num_heads: int = 6, dropout: float = 0.05) -> None:
        super().__init__()
        if d_exp % num_heads != 0:
            raise ValueError(f"d_exp={d_exp} must be divisible by num_heads={num_heads}")
        self.d_exp = d_exp
        self.d_mem = d_mem
        self.num_heads = num_heads
        self.head_dim = d_exp // num_heads
        self.dropout = float(dropout)

        # Path 1: Zero-init Cross-Attention from action tokens `(B, 20, 720)` to Coda `C_t (B, N_c, 384)`
        self.mem_q_norm = nn.RMSNorm(d_exp)
        self.mem_kv_norm = nn.RMSNorm(d_mem)
        self.mem_q_proj = nn.Linear(d_exp, d_exp, bias=False)
        self.mem_kv_proj = nn.Linear(d_mem, 2 * d_exp, bias=False)
        self.mem_o_proj = nn.Linear(d_exp, d_exp, bias=False)
        nn.init.zeros_(self.mem_o_proj.weight)
        self.gamma_ctx = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        # Path 2: Zero-init FiLM scale/shift + flow-time awareness before pretrained Expert SwiGLU MLP
        self.film_proj = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(d_mem + d_exp, 2 * d_exp, bias=True),
        )
        nn.init.zeros_(self.film_proj[-1].weight)  # type: ignore[arg-type]
        nn.init.zeros_(self.film_proj[-1].bias)    # type: ignore[arg-type]

    def forward(
        self,
        u_ell: torch.Tensor,                      # (B, H_act, 720) post-VLM-attn residual
        post_attn_norm: nn.Module,                # pretrained expert_layer.post_attention_layernorm
        mlp: nn.Module,                           # pretrained expert_layer.mlp
        t_emb: torch.Tensor,                      # (B, 720) sinusoidal flow-time embedding
        mem_tokens: Optional[torch.Tensor] = None,# (B, N_c, 384) Coda slots C_t^{(l)}
        mem_mask: Optional[torch.Tensor] = None,  # Optional (B, N_c) bool
    ) -> torch.Tensor:
        norm_dt = post_attn_norm.weight.dtype  # type: ignore[attr-defined]
        if mem_tokens is None:
            h_norm = post_attn_norm(u_ell.to(norm_dt))
            return u_ell + mlp(h_norm).to(u_ell.dtype)

        q_dt = self.mem_q_proj.weight.dtype
        B, T_act, D = u_ell.shape
        u_in = u_ell.to(q_dt)
        mem_t = mem_tokens.to(q_dt)

        # 1. Safe Gated Memory Cross-Attention (zero NaN even when `mem_mask` is all-False!)
        q = self.mem_q_proj(self.mem_q_norm(u_in)).view(B, T_act, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.mem_kv_proj(self.mem_kv_norm(mem_t)).view(B, -1, 2, self.num_heads, self.head_dim)
        k = kv[:, :, 0].transpose(1, 2)
        v = kv[:, :, 1].transpose(1, 2)
        ca = safe_masked_sdpa(q, k, v, key_valid_mask=mem_mask)
        ca_out = self.mem_o_proj(ca.transpose(1, 2).reshape(B, T_act, D))
        if self.training and self.dropout > 0.0:
            ca_out = F.dropout(ca_out, p=self.dropout, training=True)

        gate = (1.0 + torch.tanh(self.gamma_ctx)).to(q_dt)
        if mem_mask is not None:
            gate = gate * mem_mask.any(dim=-1).view(B, 1, 1).to(q_dt)
        u_mod = u_ell + (gate * ca_out).to(u_ell.dtype)

        # 2. FiLM-Modulated Pretrained Expert MLP
        if mem_mask is not None:
            mf = mem_mask.to(q_dt).unsqueeze(-1)
            m_pool = (mem_t * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
            has_m = mem_mask.any(dim=-1, keepdim=True).to(norm_dt)
        else:
            m_pool = mem_t.mean(dim=1)
            has_m = torch.ones((B, 1), device=u_ell.device, dtype=norm_dt)

        film_in = torch.cat([m_pool, t_emb.to(q_dt)], dim=-1)
        scale, shift = self.film_proj(film_in).to(norm_dt).chunk(2, dim=-1)
        h_norm = post_attn_norm(u_mod.to(norm_dt))
        h_mod = h_norm * (1.0 + (scale * has_m).unsqueeze(1)) + (shift * has_m).unsqueeze(1)
        return u_mod + mlp(h_mod).to(u_mod.dtype)


# =============================================================================
# 3. `SmolVLMWithExpertBackbone` (Exact Key Layout Matching `model.safetensors`)
# =============================================================================


class SmolVLMWithExpertBackbone(nn.Module):
    """Truncated 16-layer `SmolVLM2` + 16-layer Interleaved `lm_expert` without `AutoProcessor`."""

    def __init__(self, cfg: SmolVLADeltaLoopConfig, *, attach_memory_hooks: bool = False) -> None:
        super().__init__()
        from transformers import AutoConfig, AutoModel, SmolVLMForConditionalGeneration

        hf_cfg = _load_hf_config_offline_first(cfg.vlm_model_name)
        self.vlm = SmolVLMForConditionalGeneration(config=hf_cfg)
        if cfg.num_vlm_layers > 0:
            self.vlm.model.text_model.layers = self.vlm.model.text_model.layers[: cfg.num_vlm_layers]
        # Drop unused 47.31M `lm_head` to save 94.6 MB VRAM per worker!
        self.vlm.lm_head = nn.Identity()

        self.num_vlm_layers = len(self.vlm.model.text_model.layers)
        self.config = hf_cfg

        exp_cfg = copy.deepcopy(hf_cfg.text_config)
        vlm_hidden = int(exp_cfg.hidden_size)
        exp_hidden = int(vlm_hidden * cfg.expert_width_multiplier)
        exp_cfg.hidden_size = exp_hidden
        exp_cfg.intermediate_size = get_intermediate_size(exp_hidden)
        exp_cfg.num_hidden_layers = (
            cfg.num_expert_layers if cfg.num_expert_layers > 0 else self.num_vlm_layers
        )
        self.lm_expert = AutoModel.from_config(exp_cfg)
        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = int(cfg.self_attn_every_n_layers)

        vlm_kv_dim = int(hf_cfg.text_config.num_key_value_heads * hf_cfg.text_config.head_dim)
        exp_kv_dim = int(exp_cfg.num_key_value_heads * exp_cfg.head_dim)
        for layer_idx in range(self.num_expert_layers):
            if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                continue
            self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                vlm_kv_dim, exp_kv_dim, bias=bool(exp_cfg.attention_bias)
            )
            self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                vlm_kv_dim, exp_kv_dim, bias=bool(exp_cfg.attention_bias)
            )
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = int(hf_cfg.text_config.num_attention_heads)
        self.num_key_value_heads = int(hf_cfg.text_config.num_key_value_heads)
        self.head_dim = int(hf_cfg.text_config.head_dim)
        self.vlm_hidden_size = vlm_hidden
        self.expert_hidden_size = exp_hidden
        self.freeze_vlm = bool(cfg.freeze_vlm)

        # Optional zero-init memory hooks on odd cross-attention layers `1, 3, ..., 15`
        self.memory_hooks: Optional[nn.ModuleDict] = None
        if attach_memory_hooks:
            hooks = {}
            for layer_idx in range(self.num_expert_layers):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                hooks[str(layer_idx)] = SmolVLAExpertMemoryHook(
                    d_exp=exp_hidden, d_mem=cfg.d_mem, num_heads=cfg.mem_heads, dropout=cfg.hook_dropout
                )
            self.memory_hooks = nn.ModuleDict(hooks)

        if self.freeze_vlm:
            self.vlm.eval()
            self.vlm.requires_grad_(False)

    def train(self, mode: bool = True) -> "SmolVLMWithExpertBackbone":
        super().train(mode)
        if self.freeze_vlm:
            self.vlm.eval()
        return self

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        vis_model = self.vlm.model.vision_model
        hs = vis_model(pixel_values=image.to(dtype=vis_model.dtype), patch_attention_mask=None).last_hidden_state
        return self.vlm.model.connector(hs)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.vlm.model.text_model.get_input_embeddings()(tokens)

    def eager_attention_forward(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """Exact float32-upcast GQA eager attention matching `smolvlm_with_expert.py` L516–561."""
        num_att_heads = self.num_attention_heads
        num_kv_heads = self.num_key_value_heads
        num_kv_groups = num_att_heads // num_kv_heads
        seq_len = key_states.shape[1]

        k_exp = key_states[:, :, :, None, :].expand(
            batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim
        ).reshape(batch_size, seq_len, num_att_heads, head_dim)
        v_exp = value_states[:, :, :, None, :].expand(
            batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim
        ).reshape(batch_size, seq_len, num_att_heads, head_dim)

        q_f32 = query_states.to(dtype=torch.float32).transpose(1, 2)
        k_f32 = k_exp.to(dtype=torch.float32).transpose(1, 2)

        att_weights = torch.matmul(q_f32, k_f32.transpose(2, 3)) * (head_dim**-0.5)
        big_neg = torch.finfo(att_weights.dtype).min
        masked = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = F.softmax(masked, dim=-1).to(dtype=v_exp.dtype)

        att_out = torch.matmul(probs, v_exp.permute(0, 2, 1, 3))
        return att_out.permute(0, 2, 1, 3).reshape(batch_size, -1, num_att_heads * head_dim)

    def prefill_prefix_kv(
        self,
        prefix_embs: torch.Tensor,          # (B, L_prefix, 960)
        prefix_att_2d_masks: torch.Tensor,  # (B, L_prefix, L_prefix) bool
        prefix_position_ids: torch.Tensor,  # (B, L_prefix) int64
        *,
        split_state_grad: bool = True,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Run 16 VLM text layers once and return immutable post-RoPE `past_key_values`.

        When `prefix_embs[:, :-1]` is frozen `(img + lang, att_mask=0)` and only `prefix_embs[:, -1:]`
        (`state_proj`, `att_mask=1`) requires gradients, tokens `0..L_p-2` never attend to token `L_p-1`
        (`prefix_att_2d_masks[:, :-1, -1] == False`). Running `0..L_p-2` under `torch.no_grad()` and only
        the 1-token state stream under autograd is mathematically identical (`0.0` diff on both forward KV
        and `state_proj` gradients) while cutting VLM activation memory & backward FLOPs by `176x`!
        """
        B, L_p, _ = prefix_embs.shape
        vlm_layers = self.vlm.model.text_model.layers
        past_kv: Dict[int, Dict[str, torch.Tensor]] = {}

        can_split = (
            split_state_grad
            and torch.is_grad_enabled()
            and self.freeze_vlm
            and L_p > 1
            and prefix_embs.requires_grad
            and (not bool(prefix_att_2d_masks[:, :-1, -1].any()))
        )
        if can_split:
            h_vlm = prefix_embs[:, :-1].detach()
            h_st = prefix_embs[:, -1:]
            pos_vlm = prefix_position_ids[:, :-1]
            pos_st = prefix_position_ids[:, -1:]
            mask_vlm = prefix_att_2d_masks[:, :-1, :-1]
            mask_st = prefix_att_2d_masks[:, -1:, :]

            for layer_idx in range(self.num_vlm_layers):
                layer = vlm_layers[layer_idx]
                norm_dt = layer.input_layernorm.weight.dtype
                proj_dt = layer.self_attn.q_proj.weight.dtype
                o_dt = layer.self_attn.o_proj.weight.dtype
                post_dt = layer.post_attention_layernorm.weight.dtype

                with torch.no_grad():
                    hn_0 = layer.input_layernorm(h_vlm.to(norm_dt)).to(proj_dt)
                    s_0 = (B, L_p - 1, -1, self.head_dim)
                    q_0 = apply_rope(layer.self_attn.q_proj(hn_0).view(s_0), pos_vlm)
                    k_0 = apply_rope(layer.self_attn.k_proj(hn_0).view(s_0), pos_vlm)
                    v_0 = layer.self_attn.v_proj(hn_0).view(s_0)
                    att_0 = self.eager_attention_forward(mask_vlm, B, self.head_dim, q_0, k_0, v_0)
                    u_0 = h_vlm.to(o_dt) + layer.self_attn.o_proj(att_0.to(o_dt))
                    h_vlm = u_0 + layer.mlp(layer.post_attention_layernorm(u_0.to(post_dt))).to(o_dt)

                hn_1 = layer.input_layernorm(h_st.to(norm_dt)).to(proj_dt)
                s_1 = (B, 1, -1, self.head_dim)
                q_1 = apply_rope(layer.self_attn.q_proj(hn_1).view(s_1), pos_st)
                k_1 = apply_rope(layer.self_attn.k_proj(hn_1).view(s_1), pos_st)
                v_1 = layer.self_attn.v_proj(hn_1).view(s_1)

                k_full = torch.cat([k_0, k_1], dim=1)
                v_full = torch.cat([v_0, v_1], dim=1)
                past_kv[layer_idx] = {"key_states": k_full, "value_states": v_full}

                att_1 = self.eager_attention_forward(mask_st, B, self.head_dim, q_1, k_full, v_full)
                u_1 = h_st.to(o_dt) + layer.self_attn.o_proj(att_1.to(o_dt))
                h_st = u_1 + layer.mlp(layer.post_attention_layernorm(u_1.to(post_dt))).to(o_dt)

            return past_kv

        hidden_states = prefix_embs
        for layer_idx in range(self.num_vlm_layers):
            layer = vlm_layers[layer_idx]
            norm_dt = layer.input_layernorm.weight.dtype
            proj_dt = layer.self_attn.q_proj.weight.dtype
            o_dt = layer.self_attn.o_proj.weight.dtype
            post_dt = layer.post_attention_layernorm.weight.dtype

            h_norm = layer.input_layernorm(hidden_states.to(norm_dt)).to(proj_dt)
            h_shape = (B, L_p, -1, self.head_dim)
            q = apply_rope(layer.self_attn.q_proj(h_norm).view(h_shape), prefix_position_ids)
            k = apply_rope(layer.self_attn.k_proj(h_norm).view(h_shape), prefix_position_ids)
            v = layer.self_attn.v_proj(h_norm).view(h_shape)

            past_kv[layer_idx] = {"key_states": k, "value_states": v}

            att_out = self.eager_attention_forward(prefix_att_2d_masks, B, self.head_dim, q, k, v)
            u = hidden_states.to(o_dt) + layer.self_attn.o_proj(att_out.to(o_dt))
            hidden_states = u + layer.mlp(layer.post_attention_layernorm(u.to(post_dt))).to(o_dt)

        return past_kv

    def forward_expert_with_kv(
        self,
        suffix_embs: torch.Tensor,                      # (B_suf, chunk_size, 720)
        full_att_2d_masks: torch.Tensor,                # (B_suf, chunk_size, L_prefix + chunk_size)
        suffix_position_ids: torch.Tensor,              # (B_suf, chunk_size)
        past_key_values: Dict[int, Dict[str, torch.Tensor]],
        *,
        t_emb: Optional[torch.Tensor] = None,           # (B_suf, 720) sinusoidal time emb
        mem_tokens: Optional[torch.Tensor] = None,      # Optional (B_suf, N_c, 384)
        mem_mask: Optional[torch.Tensor] = None,        # Optional (B_suf, N_c) bool
    ) -> torch.Tensor:
        """Run 16 interleaved Action Expert layers reusing immutable `past_key_values` out-of-place!

        Supports `B_suf == K_rep * B_kv` (`K_rep >= 1`) so all `L=4` multi-hop depths or `M_CRN=8`
        CRN draws share a single VLM prefix prefill without duplicating VLM computation!
        """
        B_suf, L_suf, _ = suffix_embs.shape
        hidden = suffix_embs
        zero_based_pos = suffix_position_ids - torch.min(suffix_position_ids, dim=1, keepdim=True).values

        for layer_idx in range(self.num_expert_layers):
            layer = self.lm_expert.layers[layer_idx]
            kv_entry = past_key_values[layer_idx]
            k_pref = kv_entry["key_states"]
            v_pref = kv_entry["value_states"]
            B_kv, L_pref = int(k_pref.shape[0]), int(k_pref.shape[1])
            if B_suf != B_kv:
                if B_suf % B_kv != 0:
                    raise ValueError(f"B_suf={B_suf} must be a multiple of B_kv={B_kv}")
                rep = B_suf // B_kv
                k_pref = k_pref.repeat(rep, 1, 1, 1)
                v_pref = v_pref.repeat(rep, 1, 1, 1)

            norm_dt = layer.input_layernorm.weight.dtype
            q_dt = layer.self_attn.q_proj.weight.dtype
            k_dt = layer.self_attn.k_proj.weight.dtype
            v_dt = layer.self_attn.v_proj.weight.dtype
            o_dt = layer.self_attn.o_proj.weight.dtype
            post_dt = layer.post_attention_layernorm.weight.dtype

            h_norm = layer.input_layernorm(hidden.to(norm_dt))
            h_shape = (B_suf, L_suf, -1, self.head_dim)
            is_even_sa = (
                self.self_attn_every_n_layers > 0 and (layer_idx % self.self_attn_every_n_layers == 0)
            )

            if is_even_sa:
                q_suf = apply_rope(layer.self_attn.q_proj(h_norm.to(q_dt)).view(h_shape), suffix_position_ids)
                k_suf = apply_rope(layer.self_attn.k_proj(h_norm.to(k_dt)).view(h_shape), suffix_position_ids)
                v_suf = layer.self_attn.v_proj(h_norm.to(v_dt)).view(h_shape)

                k_cat = torch.cat([k_pref.to(k_suf.dtype), k_suf], dim=1)
                v_cat = torch.cat([v_pref.to(v_suf.dtype), v_suf], dim=1)
                att_out = self.eager_attention_forward(
                    full_att_2d_masks, B_suf, self.head_dim, q_suf, k_cat, v_cat
                )
            else:
                q_suf = apply_rope(layer.self_attn.q_proj(h_norm.to(q_dt)).view(h_shape), zero_based_pos)
                k_flat = k_pref.to(k_dt).reshape(B_suf, L_pref, -1)
                v_flat = v_pref.to(v_dt).reshape(B_suf, L_pref, -1)
                k_exp = layer.self_attn.k_proj(k_flat).view(B_suf, L_pref, -1, self.head_dim)
                v_exp = layer.self_attn.v_proj(v_flat).view(B_suf, L_pref, -1, self.head_dim)
                ca_mask = full_att_2d_masks[:, -L_suf:, :L_pref]
                att_out = self.eager_attention_forward(
                    ca_mask, B_suf, self.head_dim, q_suf, k_exp, v_exp
                )

            u_ell = hidden.to(o_dt) + layer.self_attn.o_proj(att_out.to(o_dt))

            key_str = str(layer_idx)
            if (not is_even_sa) and self.memory_hooks is not None and key_str in self.memory_hooks:
                assert t_emb is not None
                hidden = self.memory_hooks[key_str](
                    u_ell,
                    layer.post_attention_layernorm,
                    layer.mlp,
                    t_emb=t_emb,
                    mem_tokens=mem_tokens,
                    mem_mask=mem_mask,
                )
            else:
                hidden = u_ell + layer.mlp(layer.post_attention_layernorm(u_ell.to(post_dt))).to(o_dt)

        final_dt = self.lm_expert.norm.weight.dtype
        return self.lm_expert.norm(hidden.to(final_dt))


def _load_hf_config_offline_first(model_id: str) -> Any:
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_id, local_files_only=True)
    except Exception:
        return AutoConfig.from_pretrained(model_id)


def load_smolvla_tokenizer(model_id: str = SMOLVLM2_MODEL_ID) -> Any:
    """Load `SmolVLM2` tokenizer offline-first WITHOUT importing `AutoProcessor` (`num2words`-free!)."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"
    return tok


def tokenize_smolvla_prompts(
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_length: int = 48,
    device: Optional[torch.device | str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact `smolvla_new_line_processor` + `tokenizer_processor` (`padding='max_length'`, `right`)."""
    formatted = [format_smolvla_prompt(p) for p in prompts]
    tokenizer.padding_side = "right"
    enc = tokenizer(
        formatted,
        max_length=int(max_length),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(dtype=torch.long)
    mask = enc["attention_mask"].to(dtype=torch.bool)
    if device is not None:
        ids = ids.to(device=device, non_blocking=True)
        mask = mask.to(device=device, non_blocking=True)
    return ids, mask


def cast_smolvla_precision(
    model: "SmolVLADeltaLoopPolicy",
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> "SmolVLADeltaLoopPolicy":
    """Move `SmolVLADeltaLoopPolicy` to `(device, dtype)` while preserving `fp32` projections & GDN/gate scalars!"""
    dev = torch.device(device)
    model.to(device=dev, dtype=dtype)
    # Match `smolvla_base` mixed precision: continuous state/action projections, odd VLM->Expert KV projections,
    # halting gate, and GDN log-decay/calibration scalars stay in exact `float32`!
    fp32_prefixes = (
        "state_proj.",
        "action_in_proj.",
        "action_out_proj.",
        "action_time_mlp_in.",
        "action_time_mlp_out.",
        "gripper_head.",
        "halting_gate.",
    )
    fp32_suffixes = (
        "A_log",
        "dt_bias",
        "gamma_ctx",
        "anchor_role_emb",
        "calibrated_delta",
        "vis_anchor_proj",
        "_u1_loss_weights",
    )
    for name, p in model.named_parameters():
        if name.startswith(fp32_prefixes) or name.endswith(fp32_suffixes):
            p.data = p.data.float()
    # Odd `lm_expert` cross-attention `k_proj` / `v_proj` are stored as F32 in `lerobot/smolvla_base`
    sa_every = model.vlm_with_expert.self_attn_every_n_layers
    for layer_idx, layer in enumerate(model.vlm_with_expert.lm_expert.layers):
        if sa_every > 0 and (layer_idx % sa_every != 0):
            layer.self_attn.k_proj.float()
            layer.self_attn.v_proj.float()
    for name, b in model.named_buffers():
        if (name.startswith(fp32_prefixes) or name.endswith(fp32_suffixes)) and b is not None and b.is_floating_point():
            b.data = b.data.float()
    return model


# =============================================================================
# 4. Unified `SmolVLADeltaLoopPolicy` (`smolvla_m0_base` .. `smolvla_m7_deltaloop`)
# =============================================================================


class SmolVLADeltaLoopPolicy(nn.Module):
    """Unified Pretrained `SmolVLA` + `DeltaLoop` Policy Suite (`U6`)."""

    def __init__(self, cfg: SmolVLADeltaLoopConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.arch: ArchSmolVLA = cfg.arch
        self._mem_cfg = cfg.to_mem_helper_config()
        d_mem = cfg.d_mem

        attach_hooks = self.arch != "smolvla_m0_base"
        self.vlm_with_expert = SmolVLMWithExpertBackbone(cfg, attach_memory_hooks=attach_hooks)

        d_vlm = self.vlm_with_expert.vlm_hidden_size
        d_exp = self.vlm_with_expert.expert_hidden_size

        # Exact pretrained top-level linear layers matching `model.safetensors`
        self.state_proj = nn.Linear(cfg.max_state_dim, d_vlm)
        self.action_in_proj = nn.Linear(cfg.max_action_dim, d_exp)
        self.action_out_proj = nn.Linear(d_exp, cfg.max_action_dim)
        self.action_time_mlp_in = nn.Linear(2 * d_exp, d_exp)
        self.action_time_mlp_out = nn.Linear(d_exp, d_exp)

        if not cfg.train_state_proj:
            self.state_proj.requires_grad_(False)

        # Shared U1 Gripper BCE head on the 16 executed suffix hidden states
        self.gripper_head: Optional[nn.Linear] = None
        if cfg.use_gripper_head:
            self.gripper_head = nn.Linear(d_exp, 1)
            nn.init.normal_(self.gripper_head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.gripper_head.bias)

        # Lightweight current readout context encoder `E_t` (`vis_dim=960 -> d_mem=384`)
        # Instantiated when needed for Prelude/Core (`M1, M5, M7`) or U4 grounding (`all` if enabled)
        need_ctx_enc = (self.arch in ("smolvla_m1_rdvla_loop4", "smolvla_m5_gdn2l_l1", "smolvla_m7_deltaloop")) or cfg.use_grounding_head
        self.readout_ctx_encoder: Optional[SharedObservationEncoder] = (
            SharedObservationEncoder(self._mem_cfg) if need_ctx_enc else None
        )

        # Architecture-specific Memory & Multi-Hop Readout modules (`512 KB` exact iso-state!)
        self.slot_resampler: Optional[PerceiverSlotResampler] = None
        self.gdn1: Optional[SlotGDNMemoryWriter] = None
        self.gdn2: Optional[SlotGDNMemoryWriter] = None
        self.mid_norm: Optional[nn.RMSNorm] = None
        self.mid_ffn: Optional[SwiGLU] = None
        self.readout1: Optional[DampedSlotMemoryReadout] = None
        self.readout2: Optional[DampedSlotMemoryReadout] = None
        self.prelude: Optional[VLAPrelude] = None
        self.core: Optional[VLARecurrentCore] = None
        self.coda: Optional[VLACoda] = None
        self.halting_gate: Optional[CRNHaltingGate] = None
        self.anchor_role_emb: Optional[nn.Parameter] = None
        self.framesamp_refine: Optional[ContextTransformerBlock] = None
        self.aux_head: Optional[nn.Linear] = None

        if self.arch == "smolvla_m0_base":
            pass

        elif self.arch == "smolvla_m1_rdvla_loop4":
            self.prelude = VLAPrelude(self._mem_cfg)
            self.core = VLARecurrentCore(self._mem_cfg)
            self.coda = VLACoda(d_mem)
            self.aux_head = nn.Linear(d_mem, cfg.aux_dim)

        elif self.arch == "smolvla_m2_framesamp":
            self.slot_resampler = PerceiverSlotResampler(self._mem_cfg, num_slots=1)
            self.slot_resampler.temporal_conv = None
            self.framesamp_refine = ContextTransformerBlock(d_mem, cfg.mem_heads, d_mem * 2)
            self.coda = VLACoda(d_mem)
            self.aux_head = nn.Linear(d_mem, cfg.aux_dim)

        elif self.arch in ("smolvla_m5_gdn2l_l1", "smolvla_m7_deltaloop"):
            d_v_half = cfg.gdn_d_v // 2
            self.slot_resampler = PerceiverSlotResampler(self._mem_cfg)
            self.gdn1 = SlotGDNMemoryWriter(
                d_model=d_mem,
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
            self.mid_norm = nn.RMSNorm(d_mem)
            self.mid_ffn = SwiGLU(d_mem, d_mem * 2)
            self.gdn2 = SlotGDNMemoryWriter(
                d_model=d_mem,
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
                d_mem, cfg.gdn_heads, cfg.gdn_d_k, d_v_half, eps_read=cfg.readout_eps
            )
            self.readout2 = DampedSlotMemoryReadout(
                d_mem, cfg.gdn_heads, cfg.gdn_d_k, d_v_half, eps_read=cfg.readout_eps
            )
            self.prelude = VLAPrelude(self._mem_cfg)
            self.core = VLARecurrentCore(self._mem_cfg)
            self.coda = VLACoda(d_mem)
            self.aux_head = nn.Linear(d_mem, cfg.aux_dim)
            if self.arch == "smolvla_m7_deltaloop":
                self.halting_gate = CRNHaltingGate(
                    d_model=d_mem, num_heads=cfg.gdn_heads, max_loops=cfg.max_loops
                )
        else:
            raise ValueError(f"Unknown SmolVLA architecture: {self.arch!r}")

        if cfg.anchor_hop0_coda and self.is_multi_depth:
            self.anchor_role_emb = nn.Parameter(torch.zeros((2, d_mem), dtype=torch.float32))

        # Shared training-only `U4` 2D `<row_y, col_x>` & subgoal language alignment heads
        self.grounding_xy_head: Optional[nn.Sequential] = None
        self.grounding_sg_head: Optional[nn.Sequential] = None
        if cfg.use_grounding_head:
            self.grounding_xy_head = nn.Sequential(
                nn.RMSNorm(d_mem),
                nn.Linear(d_mem, d_mem),
                nn.SiLU(),
                nn.Linear(d_mem, 2),
            )
            self.grounding_sg_head = nn.Sequential(
                nn.RMSNorm(d_mem),
                nn.Linear(d_mem, d_mem),
                nn.SiLU(),
                nn.Linear(d_mem, cfg.subgoal_dim),
            )

        if cfg.use_u1_loss_weights:
            w_hd = torch.zeros((cfg.chunk_size, cfg.action_dim), dtype=torch.float32)
            for h_i in range(cfg.chunk_size):
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

        gen = torch.Generator()
        gen.manual_seed(20261112)
        vis_anchor_mat = torch.randn((d_vlm, 32), generator=gen, dtype=torch.float32) / math.sqrt(d_vlm)
        self.register_buffer("vis_anchor_proj", vis_anchor_mat, persistent=True)

    # -------------------------------------------------------------------------
    # Pretrained Safetensors Loading & Parameter Introspection
    # -------------------------------------------------------------------------

    def load_pretrained_smolvla(self, safetensors_path: str | Path) -> Dict[str, int]:
        """Load `lerobot/smolvla_base` `model.safetensors` into VLM + Action Expert + Projections."""
        from safetensors.torch import load_file

        raw_sd = load_file(str(safetensors_path), device="cpu")
        mapped: Dict[str, torch.Tensor] = {}
        skipped_lm_head = 0
        for k, v in raw_sd.items():
            clean_k = k[len("model.") :] if k.startswith("model.") else k
            if clean_k.startswith("vlm_with_expert.vlm.lm_head."):
                skipped_lm_head += 1
                continue
            mapped[clean_k] = v

        incompatible = self.load_state_dict(mapped, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys when loading smolvla_base safetensors: {incompatible.unexpected_keys[:10]}"
            )
        # Verify ZERO pretrained backbone keys were missed!
        pretrained_prefixes = (
            "vlm_with_expert.vlm.",
            "vlm_with_expert.lm_expert.",
            "state_proj.",
            "action_in_proj.",
            "action_out_proj.",
            "action_time_mlp_in.",
            "action_time_mlp_out.",
        )
        missed_pretrained = [
            mk for mk in incompatible.missing_keys if mk.startswith(pretrained_prefixes)
        ]
        if missed_pretrained:
            raise RuntimeError(f"Missed pretrained smolvla_base keys: {missed_pretrained[:10]}")

        if self.cfg.freeze_vlm:
            self.vlm_with_expert.vlm.eval()
            self.vlm_with_expert.vlm.requires_grad_(False)
        if not self.cfg.train_state_proj:
            self.state_proj.requires_grad_(False)

        return {
            "loaded_tensors": len(mapped),
            "skipped_lm_head": skipped_lm_head,
            "new_deltaloop_tensors": len(incompatible.missing_keys),
        }

    @property
    def uses_history(self) -> bool:
        return self.arch in ("smolvla_m2_framesamp", "smolvla_m5_gdn2l_l1", "smolvla_m7_deltaloop")

    @property
    def uses_gdn(self) -> bool:
        return self.arch in ("smolvla_m5_gdn2l_l1", "smolvla_m7_deltaloop")

    @property
    def is_multi_depth(self) -> bool:
        return self.arch in ("smolvla_m1_rdvla_loop4", "smolvla_m7_deltaloop")

    @property
    def default_train_loops(self) -> int:
        return self.cfg.max_loops if self.is_multi_depth else 1

    def state_size_bytes(self) -> int:
        if self.arch in ("smolvla_m0_base", "smolvla_m1_rdvla_loop4"):
            return 0
        if self.arch == "smolvla_m2_framesamp":
            return self.cfg.framesamp_buffer_size * self.cfg.d_mem * 2
        assert self.gdn1 is not None and self.gdn2 is not None
        occ_elems = 2 * self.cfg.gdn_heads * self.cfg.gdn_d_k
        return (self.gdn1.state_numel + self.gdn2.state_numel + occ_elems) * 4

    def parameter_breakdown(self) -> Dict[str, int]:
        vlm_frozen = sum(p.numel() for p in self.vlm_with_expert.vlm.parameters())
        expert_core = sum(p.numel() for p in self.vlm_with_expert.lm_expert.parameters())
        projs = sum(
            sum(p.numel() for p in m.parameters())
            for m in (
                self.state_proj,
                self.action_in_proj,
                self.action_out_proj,
                self.action_time_mlp_in,
                self.action_time_mlp_out,
            )
        )
        if self.gripper_head is not None:
            projs += sum(p.numel() for p in self.gripper_head.parameters())
        hooks = (
            sum(p.numel() for p in self.vlm_with_expert.memory_hooks.parameters())
            if self.vlm_with_expert.memory_hooks is not None
            else 0
        )
        halt = sum(p.numel() for p in self.halting_gate.parameters()) if self.halting_gate is not None else 0
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_active": total,
            "trainable": trainable,
            "vlm_frozen": vlm_frozen,
            "pretrained_expert_and_projs": expert_core + projs,
            "expert_memory_hooks": hooks,
            "memory_loop_and_aux": total - vlm_frozen - expert_core - projs - hooks - halt,
            "halting_gate": halt,
        }

    def build_differential_param_groups(
        self,
        *,
        lr_expert: float = 5e-5,
        lr_heads: float = 1e-4,
        lr_mem: float = 2e-4,
        wd_pretrained: float = 1e-4,
        wd_mem: float = 1e-3,
    ) -> List[Dict[str, Any]]:
        """Differential AdamW groups (`docs/PRETRAINED_VLA_DESIGN.md` §5.3)."""
        expert_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []
        mem_params: List[nn.Parameter] = []

        head_prefixes = (
            "state_proj.",
            "action_in_proj.",
            "action_out_proj.",
            "action_time_mlp_in.",
            "action_time_mlp_out.",
            "gripper_head.",
        )
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("halting_gate."):
                # Stage-II trains halting_gate separately with frozen backbone
                continue
            if name.startswith("vlm_with_expert.lm_expert."):
                expert_params.append(p)
            elif name.startswith(head_prefixes):
                head_params.append(p)
            else:
                mem_params.append(p)

        groups: List[Dict[str, Any]] = []
        if expert_params:
            groups.append({"params": expert_params, "lr": lr_expert, "weight_decay": wd_pretrained, "group_name": "expert"})
        if head_params:
            groups.append({"params": head_params, "lr": lr_heads, "weight_decay": wd_pretrained, "group_name": "heads"})
        if mem_params:
            groups.append({"params": mem_params, "lr": lr_mem, "weight_decay": wd_mem, "group_name": "mem_loop"})
        return groups

    def split_write_vs_other_params(self) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        """Partition trainable parameters with active grads for separate gradient norm clipping!"""
        write_prefixes = (
            "slot_resampler.",
            "gdn1.",
            "gdn2.",
            "mid_norm.",
            "mid_ffn.",
            "framesamp_refine.",
        )
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
    # A3 Cutoff, U5 Striding & Packed Episode-Prefix GDN Scan
    # -------------------------------------------------------------------------

    @staticmethod
    def compute_a3_cutoff_excl(step_idx: int, exec_start_idx: int, exclusion_window: int = 16) -> int:
        return DeltaLoopVLAPolicy.compute_a3_cutoff_excl(step_idx, exec_start_idx, exclusion_window)

    def compute_written_frame_indices(
        self, cutoff_excl: int, exec_start_idx: int, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, int]:
        return DeltaLoopVLAPolicy.compute_written_frame_indices(
            cutoff_excl, exec_start_idx, self._mem_cfg, device=device
        )

    @staticmethod
    def compute_visual_motion_rms(front_seq: torch.Tensor) -> torch.Tensor:
        return DeltaLoopVLAPolicy.compute_visual_motion_rms(front_seq)

    @staticmethod
    def select_framesamp_slot_indices(
        num_eligible_strided: int,
        num_demo_strided: int,
        budget: int = 32,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, int]:
        return DeltaLoopVLAPolicy.select_framesamp_slot_indices(
            num_eligible_strided, num_demo_strided, budget=budget, device=device
        )

    def compute_packed_gdn_states(
        self,
        episode_slot_seqs: List[torch.Tensor],
        sample_ep_ids: torch.Tensor,
        sample_written_counts: torch.Tensor,
        *,
        episode_motion_seqs: Optional[List[torch.Tensor]] = None,
        force_pytorch: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Exact 2-layer warm-cache deduplicated prefix GDN scan (`initial_state=None`)."""
        assert self.gdn1 is not None and self.gdn2 is not None and self.mid_norm is not None and self.mid_ffn is not None
        B = int(sample_ep_ids.shape[0])
        device = sample_ep_ids.device
        Nw = self.cfg.num_write_slots
        H = self.cfg.gdn_heads
        K = self.cfg.gdn_d_k
        Vh = self.cfg.gdn_d_v // 2

        counts_list = sample_written_counts.tolist()
        ep_ids_list = sample_ep_ids.tolist()

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

        z_all = torch.cat(nonempty_z, dim=0)
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

        S1_out, occ1_out = DeltaLoopVLAPolicy._scan_episode_prefix_states(
            k1_all, v1_all, g1_all, b1_all, ep_tok_offset, counts_list, ep_ids_list, Nw, Vh, force_pytorch
        )
        S2_out, occ2_out = DeltaLoopVLAPolicy._scan_episode_prefix_states(
            k2_all, v2_all, g2_all, b2_all, ep_tok_offset, counts_list, ep_ids_list, Nw, Vh, force_pytorch
        )
        return {"S1": S1_out, "occ1": occ1_out, "S2": S2_out, "occ2": occ2_out}

    # -------------------------------------------------------------------------
    # VLM Vision/Prefix Encoding & Single-Prefill KV Cache
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def encode_images_to_vlm_tokens(
        self,
        front_rgb: np.ndarray | torch.Tensor,
        wrist_rgb: np.ndarray | torch.Tensor,
        *,
        batch_size: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode `(front_rgb, wrist_rgb)` into `(N, 64, 960)` VLM connector tokens (`0` extra ViT!)."""
        dev = next(self.vlm_with_expert.vlm.parameters()).device
        vis_dt = self.vlm_with_expert.vlm.model.vision_model.dtype
        f_arr = torch.from_numpy(np.ascontiguousarray(front_rgb)) if isinstance(front_rgb, np.ndarray) else front_rgb
        w_arr = torch.from_numpy(np.ascontiguousarray(wrist_rgb)) if isinstance(wrist_rgb, np.ndarray) else wrist_rgb
        if f_arr.ndim == 3:
            f_arr = f_arr.unsqueeze(0)
        if w_arr.ndim == 3:
            w_arr = w_arr.unsqueeze(0)
        total_n = int(f_arr.shape[0])
        if total_n == 0:
            empty = torch.empty((0, 64, self.vlm_with_expert.vlm_hidden_size), device=dev, dtype=vis_dt)
            return empty, empty.clone()

        pair_bs = max(1, batch_size // 2)
        f_chunks: List[torch.Tensor] = []
        w_chunks: List[torch.Tensor] = []
        for s in range(0, total_n, pair_bs):
            e = min(total_n, s + pair_bs)
            m = e - s
            f_px = preprocess_smolvla_rgb(
                f_arr[s:e], image_size=(self.cfg.image_size, self.cfg.image_size), device=dev, out_dtype=vis_dt
            )
            w_px = preprocess_smolvla_rgb(
                w_arr[s:e], image_size=(self.cfg.image_size, self.cfg.image_size), device=dev, out_dtype=vis_dt
            )
            both = self.vlm_with_expert.embed_image(torch.cat([f_px, w_px], dim=0))
            f_chunks.append(both[:m])
            w_chunks.append(both[m:])

        f_out = torch.cat(f_chunks, dim=0) if len(f_chunks) > 1 else f_chunks[0]
        w_out = torch.cat(w_chunks, dim=0) if len(w_chunks) > 1 else w_chunks[0]
        return f_out, w_out

    def apply_synchronized_front_aug(
        self,
        front_vlm: torch.Tensor,                      # (B, 64, 960)
        wrist_vlm: torch.Tensor,                      # (B, 64, 960)
        prev_front_vlm: Optional[torch.Tensor] = None,
        aug_theta: Optional[torch.Tensor] = None,     # Optional (B, 2, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply synchronized sub-grid crop + noise + patch dropout to current `front_vlm` (`U3/U4`)."""
        B = int(front_vlm.shape[0])
        eff_theta = aug_theta
        if eff_theta is None and self.training and self.cfg.subgrid_aug:
            eff_theta = sample_subgrid_crop_affine(
                B,
                scale_range=(self.cfg.subgrid_scale_min, 1.0),
                max_extra_shift=self.cfg.subgrid_max_shift,
                device=front_vlm.device,
            )

        f_out = front_vlm
        w_out = wrist_vlm
        pf_out = prev_front_vlm

        if eff_theta is not None:
            with torch.autocast(device_type=front_vlm.device.type, enabled=False):
                f_out = apply_subgrid_affine_to_tokens(f_out, eff_theta, grid_hw=(8, 8))
                if pf_out is not None:
                    pf_out = (
                        f_out
                        if pf_out is front_vlm
                        else apply_subgrid_affine_to_tokens(pf_out, eff_theta, grid_hw=(8, 8))
                    )

        if self.training and self.cfg.subgrid_aug and self.cfg.feat_noise_std > 0.0:
            f_scale = f_out.detach().float().std(dim=-1, keepdim=True).clamp(min=1e-3).to(f_out.dtype)
            w_scale = w_out.detach().float().std(dim=-1, keepdim=True).clamp(min=1e-3).to(w_out.dtype)
            f_out = f_out + torch.randn_like(f_out) * (self.cfg.feat_noise_std * f_scale)
            w_out = w_out + torch.randn_like(w_out) * (self.cfg.feat_noise_std * w_scale)

        if self.training and self.cfg.subgrid_aug and self.cfg.patch_dropout > 0.0:
            keep = F.dropout(
                torch.ones((B, f_out.shape[1], 1), device=f_out.device, dtype=f_out.dtype),
                p=self.cfg.patch_dropout,
                training=True,
            )
            f_out = f_out * keep

        return f_out, w_out, pf_out, eff_theta

    def embed_prefix_from_tokens(
        self,
        front_vlm: torch.Tensor,      # (B, 64, 960) unscaled connector tokens
        wrist_vlm: torch.Tensor,      # (B, 64, 960) unscaled connector tokens
        lang_tokens: torch.Tensor,    # (B, L_lang) int64
        lang_masks: torch.Tensor,     # (B, L_lang) bool
        state_norm: torch.Tensor,     # (B, 8) or (B, 32) MEAN_STD normalized state
        *,
        front_valid: Optional[torch.Tensor] = None,
        wrist_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact `modeling_smolvla.embed_prefix` returning `(embs, pad_masks, att_masks, lang_raw)`."""
        B = int(front_vlm.shape[0])
        device = front_vlm.device
        vlm_dt = self.vlm_with_expert.vlm.model.text_model.layers[0].self_attn.q_proj.weight.dtype
        d_vlm = self.vlm_with_expert.vlm_hidden_size
        scale_vis = torch.tensor(d_vlm**0.5, dtype=vlm_dt, device=device)

        f_emb = front_vlm.to(vlm_dt) * scale_vis
        w_emb = wrist_vlm.to(vlm_dt) * scale_vis
        Nf = int(f_emb.shape[1])
        Nw = int(w_emb.shape[1])

        f_mask = (
            front_valid[:, None].expand(B, Nf).bool()
            if front_valid is not None
            else torch.ones((B, Nf), dtype=torch.bool, device=device)
        )
        w_mask = (
            wrist_valid[:, None].expand(B, Nw).bool()
            if wrist_valid is not None
            else torch.ones((B, Nw), dtype=torch.bool, device=device)
        )

        with torch.no_grad():
            lang_raw = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_raw.to(vlm_dt) * math.sqrt(d_vlm)
        Nl = int(lang_emb.shape[1])

        st_pad = pad_vector(state_norm, self.cfg.max_state_dim).to(self.state_proj.weight.dtype)
        st_emb = self.state_proj(st_pad).to(vlm_dt).unsqueeze(1)  # (B, 1, 960)
        st_mask = torch.ones((B, 1), dtype=torch.bool, device=device)

        embs = torch.cat([f_emb, w_emb, lang_emb, st_emb], dim=1)
        pad_masks = torch.cat([f_mask, w_mask, lang_masks.bool(), st_mask], dim=1)
        att_list = [0] * (Nf + Nw + Nl) + [1]
        att_masks = torch.tensor(att_list, dtype=torch.bool, device=device)[None, :].expand(B, -1)
        return embs, pad_masks, att_masks, lang_raw

    def prefill_vlm_kv_from_tokens(
        self,
        front_vlm: torch.Tensor,
        wrist_vlm: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state_norm: torch.Tensor,
        *,
        front_valid: Optional[torch.Tensor] = None,
        wrist_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Single-pass VLM prefix prefill (`L_prefix = 177`) returning reusable `past_key_values`."""
        embs, pad_masks, att_masks, lang_raw = self.embed_prefix_from_tokens(
            front_vlm,
            wrist_vlm,
            lang_tokens,
            lang_masks,
            state_norm,
            front_valid=front_valid,
            wrist_valid=wrist_valid,
        )
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        pos_ids = torch.cumsum(pad_masks, dim=1) - 1
        grad_on = torch.is_grad_enabled() and self.cfg.train_state_proj and embs.requires_grad
        with torch.set_grad_enabled(grad_on):
            past_kv = self.vlm_with_expert.prefill_prefix_kv(embs, att_2d, pos_ids)
        return {
            "past_key_values": past_kv,
            "prefix_pad_masks": pad_masks,
            "prefix_att_masks": att_masks,
            "prefix_embs": embs,
            "lang_raw": lang_raw,
        }

    def encode_readout_context(
        self,
        front_vlm: torch.Tensor,          # (B, 64, 960)
        wrist_vlm: torch.Tensor,          # (B, 64, 960) or (B, 16, 960)
        state_norm: torch.Tensor,         # (B, 8)
        lang_raw: torch.Tensor,           # (B, L_lang <= 64, 960)
        lang_masks: torch.Tensor,         # (B, L_lang <= 64) bool
        *,
        prev_front_vlm: Optional[torch.Tensor] = None,
        prev_wrist_vlm: Optional[torch.Tensor] = None,
        prev_state_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode current observation into `(E_t, ctx_mask)` (`(B, N_ctx, 384)`) for Prelude/Core & U4."""
        if self.readout_ctx_encoder is None:
            return None, None

        B = int(front_vlm.shape[0])
        w_16 = pool_vlm_tokens_8x8_to_4x4(wrist_vlm)
        pw_16 = pool_vlm_tokens_8x8_to_4x4(prev_wrist_vlm) if prev_wrist_vlm is not None else None

        # Pad `lang_raw` (`L_lang = 48`) to `64` tokens expected by `SharedObservationEncoder.text_pos`
        L_l = int(lang_raw.shape[1])
        if L_l < 64:
            pad_l = 64 - L_l
            txt_tok = F.pad(lang_raw, (0, 0, 0, pad_l), value=0.0)
            txt_m = F.pad(lang_masks.bool(), (0, pad_l), value=False)
        else:
            txt_tok = lang_raw[:, :64]
            txt_m = lang_masks[:, :64].bool()

        ctx_tokens, ctx_mask = self.readout_ctx_encoder(
            front_vlm,
            w_16,
            state_norm[:, :8],
            txt_tok,
            txt_m,
            prev_front_emb=prev_front_vlm,
            prev_wrist_emb=pw_16,
            prev_state_norm=prev_state_norm[:, :8] if prev_state_norm is not None else None,
            aug_theta=None,
        )
        return ctx_tokens, ctx_mask

    def encode_history_frames_raw(
        self,
        front_vlm: torch.Tensor,    # (N, 64, 960)
        wrist_vlm: torch.Tensor,    # (N, 64, 960) or (N, 16, 960)
        state_norm: torch.Tensor,   # (N, 8)
        step_idx: torch.Tensor,     # (N,) int64
        is_demo: torch.Tensor,      # (N,) int64/bool
    ) -> torch.Tensor:
        """Encode strided history frames into `(N, N_w, 384)` before `apply_episode_slot_conv`."""
        assert self.slot_resampler is not None
        w_16 = pool_vlm_tokens_8x8_to_4x4(wrist_vlm)
        return self.slot_resampler.encode_frames_raw(
            front_vlm, w_16, state_norm[:, :8], step_idx, is_demo
        )

    # -------------------------------------------------------------------------
    # Multi-Hop Coda Readout (`M0, M1, M2, M5, M7`) & Denoising Expert Step
    # -------------------------------------------------------------------------

    def run_coda_readout(
        self,
        ctx_tokens: Optional[torch.Tensor],
        ctx_mask: Optional[torch.Tensor],
        mem_state: Optional[Dict[str, torch.Tensor]] = None,
        *,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        num_loops: Optional[int] = None,
        halt_mode: Literal["fixed", "learned_gate", "coda_conv"] = "fixed",
        halt_threshold: float = 0.5,
        coda_conv_tau: float = 0.08,
    ) -> Dict[str, Any]:
        """Run Coda memory readout / multi-hop reasoning loop (`L=1..4`)."""
        if ctx_tokens is not None:
            B = int(ctx_tokens.shape[0])
            dev = ctx_tokens.device
            dt = ctx_tokens.dtype
        else:
            assert batch_size is not None and device is not None
            B = int(batch_size)
            dev = device
            dt = self.action_in_proj.weight.dtype

        if self.arch == "smolvla_m0_base":
            return {
                "coda": None,
                "coda_per_depth": [None],
                "mem_mask": None,
                "loop_depths": torch.ones((B,), device=dev, dtype=torch.float32),
            }

        if self.arch == "smolvla_m2_framesamp":
            assert self.framesamp_refine is not None and self.coda is not None and mem_state is not None
            buf_dt = self.coda.proj.weight.dtype
            buf_tokens = mem_state["buf_tokens"].to(buf_dt)
            buf_mask = mem_state["buf_mask"]
            refined = self.framesamp_refine(buf_tokens, buf_mask)
            c_out = self.coda(refined) * buf_mask.to(buf_dt).unsqueeze(-1)
            return {
                "coda": c_out,
                "coda_per_depth": [c_out],
                "mem_mask": buf_mask,
                "loop_depths": torch.ones((B,), device=dev, dtype=torch.float32),
            }

        assert (
            ctx_tokens is not None
            and ctx_mask is not None
            and self.prelude is not None
            and self.core is not None
            and self.coda is not None
        )
        h = self.prelude(ctx_tokens, ctx_mask)

        if self.arch == "smolvla_m5_gdn2l_l1":
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
                "loop_depths": torch.ones((B,), device=dev, dtype=torch.float32),
            }

        L_max = (
            (num_loops if num_loops is not None else self.cfg.max_loops)
            if self.is_multi_depth
            else 1
        )
        H_gdn = self.cfg.gdn_heads

        coda_list: List[torch.Tensor] = []
        raw_coda_list: List[torch.Tensor] = []
        h_list: List[torch.Tensor] = []
        m_list: List[torch.Tensor] = []
        dh_norms: List[torch.Tensor] = []
        dc_norms: List[torch.Tensor] = []
        r_norms: List[torch.Tensor] = []
        coverages: List[torch.Tensor] = []

        can_early_break = (
            (not self.training)
            and B == 1
            and L_max > 1
            and halt_mode in ("learned_gate", "coda_conv")
            and (halt_mode == "coda_conv" or self.halting_gate is not None)
        )
        surv_b1 = torch.ones((1,), device=dev, dtype=torch.float32)
        cdf_b1 = torch.zeros((1,), device=dev, dtype=torch.float32)
        use_anchor = bool(self.cfg.anchor_hop0_coda and self.anchor_role_emb is not None)

        prev_c: Optional[torch.Tensor] = None
        for l_idx in range(L_max):
            e_l = self.core.get_loop_emb(l_idx, dt)
            if self.arch == "smolvla_m1_rdvla_loop4":
                m_l = torch.zeros_like(h)
                r_norm = torch.zeros((B, H_gdn), device=dev, dtype=torch.float32)
                cov = torch.zeros((B, H_gdn), device=dev, dtype=torch.float32)
            else:
                assert self.readout1 is not None and self.readout2 is not None and mem_state is not None
                ro1 = self.readout1(h + e_l, mem_state["S1"], mem_state.get("occ1"))
                ro2 = self.readout2(h + e_l, mem_state["S2"], mem_state.get("occ2"))
                m_l = 0.5 * (ro1["m"] + ro2["m"])
                r_norm = 0.5 * (ro1["r_norm"] + ro2["r_norm"])
                cov = 0.5 * (ro1["coverage"] + ro2["coverage"])

            h_next = self.core(h, m_l, ctx_tokens, ctx_mask, l_idx=l_idx)
            c_next = self.coda(h_next)

            dh = torch.linalg.vector_norm((h_next - h).float(), ord=2, dim=-1).mean(dim=-1, keepdim=True)
            if prev_c is None:
                dc = torch.linalg.vector_norm(c_next.float(), ord=2, dim=-1).mean(dim=-1, keepdim=True)
                rel_dc = torch.ones((B,), device=dev, dtype=torch.float32)
            else:
                dc = torch.linalg.vector_norm((c_next - prev_c).float(), ord=2, dim=-1).mean(dim=-1, keepdim=True)
                prev_norm = torch.linalg.vector_norm(prev_c.float(), ord=2, dim=-1).mean(dim=-1)
                rel_dc = dc.squeeze(-1) / (prev_norm + 1e-5)

            raw_coda_list.append(c_next)
            if use_anchor:
                assert self.anchor_role_emb is not None
                c0_anchor = raw_coda_list[0]
                role0 = self.anchor_role_emb[0].to(dtype=c_next.dtype).view(1, 1, -1)
                role1 = self.anchor_role_emb[1].to(dtype=c_next.dtype).view(1, 1, -1)
                c_mem = torch.cat([c0_anchor + role0, c_next + role1], dim=1)
            else:
                c_mem = c_next

            coda_list.append(c_mem)
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
                            "coda": c_mem,
                            "coda_per_depth": coda_list,
                            "mem_mask": None,
                            "loop_depths": torch.full((1,), float(l_idx + 1), device=dev),
                        }
                elif halt_mode == "coda_conv" and l_idx >= 1:
                    if float(rel_dc.item()) < coda_conv_tau or l_idx == L_max - 1:
                        return {
                            "coda": c_mem,
                            "coda_per_depth": coda_list,
                            "mem_mask": None,
                            "loop_depths": torch.full((1,), float(l_idx + 1), device=dev),
                        }

        halt_out = None
        chosen_coda = coda_list[-1]
        loop_depths = torch.full((B,), float(L_max), device=dev, dtype=torch.float32)

        if self.halting_gate is not None:
            halt_out = self.halting_gate.forward_all(
                h_list, m_list, dh_norms, dc_norms, r_norms, coverages
            )
            if halt_mode == "learned_gate" and L_max > 1:
                p_tilde = halt_out["p_tilde"]
                cdf = torch.cumsum(p_tilde, dim=1)
                hit = cdf >= halt_threshold
                hit[:, -1] = True
                chosen_idx = torch.argmax(hit.int(), dim=1)
                c_stack = torch.stack(coda_list, dim=1)
                gather_idx = chosen_idx.view(B, 1, 1, 1).expand(B, 1, c_stack.shape[2], c_stack.shape[3])
                chosen_coda = torch.gather(c_stack, dim=1, index=gather_idx).squeeze(1)
                loop_depths = (chosen_idx + 1).float()

        if halt_mode == "coda_conv" and L_max > 1:
            raw_stack = torch.stack(raw_coda_list, dim=1)
            c_stack = torch.stack(coda_list, dim=1)
            hit_list = [torch.zeros((B,), device=dev, dtype=torch.bool)]
            for l_i in range(1, L_max):
                diff_n = torch.linalg.vector_norm(
                    (raw_stack[:, l_i] - raw_stack[:, l_i - 1]).float(), ord=2, dim=-1
                ).mean(dim=-1)
                base_n = torch.linalg.vector_norm(
                    raw_stack[:, l_i - 1].float(), ord=2, dim=-1
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

    def embed_suffix(
        self, noisy_actions_32: torch.Tensor, timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact `modeling_smolvla.embed_suffix` (`att_masks = [1] * chunk_size`!)."""
        in_dt = self.action_in_proj.weight.dtype
        exp_dt = self.vlm_with_expert.lm_expert.layers[0].self_attn.q_proj.weight.dtype
        action_emb = self.action_in_proj(noisy_actions_32.to(in_dt))
        B, T_suf, _ = action_emb.shape
        device = action_emb.device

        t_sin = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            min_period=self.cfg.min_period,
            max_period=self.cfg.max_period,
            device=device,
        ).to(dtype=in_dt)

        t_exp = t_sin[:, None, :].expand_as(action_emb)
        at_emb = self.action_time_mlp_in(torch.cat([action_emb, t_exp], dim=2))
        at_emb = F.silu(at_emb)
        at_emb = self.action_time_mlp_out(at_emb).to(exp_dt)

        pad_masks = torch.ones((B, T_suf), dtype=torch.bool, device=device)
        att_masks = torch.ones((B, T_suf), dtype=torch.bool, device=device)
        return at_emb, pad_masks, att_masks, t_sin

    def denoise_step(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Dict[int, Dict[str, torch.Tensor]],
        x_t: torch.Tensor,                              # (B_suf, chunk_size, 32)
        timestep: torch.Tensor,                         # (B_suf,)
        *,
        mem_tokens: Optional[torch.Tensor] = None,      # Optional (B_suf, N_c, 384)
        mem_mask: Optional[torch.Tensor] = None,        # Optional (B_suf, N_c) bool
        return_gripper_logits: bool = False,
    ) -> Any:
        """One denoising forward of the `98.25M` Action Expert (`0.0` diff vs `VLAFlowMatching.denoise_step`)."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, t_sin = self.embed_suffix(x_t, timestep)
        B_suf, L_suf = int(suffix_pad_masks.shape[0]), int(suffix_pad_masks.shape[1])

        p_pad = prefix_pad_masks
        if int(p_pad.shape[0]) != B_suf:
            rep = B_suf // int(p_pad.shape[0])
            p_pad = p_pad.repeat(rep, 1)

        L_pref = int(p_pad.shape[1])
        prefix_pad_2d = p_pad[:, None, :].expand(B_suf, L_suf, L_pref)
        suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)

        prefix_offsets = torch.sum(p_pad, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        suffix_out = self.vlm_with_expert.forward_expert_with_kv(
            suffix_embs=suffix_embs,
            full_att_2d_masks=full_att_2d,
            suffix_position_ids=position_ids,
            past_key_values=past_key_values,
            t_emb=t_sin,
            mem_tokens=mem_tokens,
            mem_mask=mem_mask,
        )
        suffix_out = suffix_out[:, -L_suf:]
        out_dt = self.action_out_proj.weight.dtype
        v_32 = self.action_out_proj(suffix_out.to(out_dt)).float()

        if return_gripper_logits:
            grip_logits = (
                self.gripper_head(suffix_out[:, : self.cfg.exec_horizon, :].to(self.gripper_head.weight.dtype))
                .squeeze(-1)
                .float()
                if self.gripper_head is not None
                else None
            )
            return v_32, grip_logits
        return v_32

    # -------------------------------------------------------------------------
    # CRN Multi-Depth Flow-Matching (`t=1` Noise -> `t=0` Clean) + U1/U4/Aux Loss
    # -------------------------------------------------------------------------

    def _select_pointer_coda(self, coda_tokens: torch.Tensor) -> torch.Tensor:
        """Select concentrated `N_c` hop-l pointer registers when `anchor_hop0_coda=True` (`U8`)."""
        if self.cfg.anchor_hop0_coda and self.anchor_role_emb is not None:
            nc = self.cfg.num_coda_tokens
            if int(coda_tokens.shape[1]) == 2 * nc:
                return coda_tokens[:, nc:, :]
        return coda_tokens

    def _pool_grounding_feature(
        self,
        ctx_tokens: torch.Tensor,
        coda_tokens: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vis_n = min(80, int(ctx_tokens.shape[1]))
        ctx_pool = ctx_tokens[:, :vis_n, :].float().mean(dim=1)
        if coda_tokens is None:
            return ctx_pool
        c_ptr = self._select_pointer_coda(coda_tokens)
        if mem_mask is not None:
            mf = mem_mask.float().unsqueeze(-1)
            c_pool = (c_ptr.float() * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
        else:
            nz = (c_ptr.float().abs().sum(dim=-1, keepdim=True) > 1e-12).float()
            c_pool = (c_ptr.float() * nz).sum(dim=1) / nz.sum(dim=1).clamp(min=1.0)
        return ctx_pool + c_pool

    def predict_grounding_xy(
        self,
        ctx_tokens: Optional[torch.Tensor],
        coda_tokens: Optional[torch.Tensor] = None,
        *,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.grounding_xy_head is None or ctx_tokens is None:
            return None
        dt = self.grounding_xy_head[1].weight.dtype  # type: ignore[index]
        feat = self._pool_grounding_feature(ctx_tokens, coda_tokens, mem_mask=mem_mask).to(dt)
        return 1.25 * torch.tanh(self.grounding_xy_head(feat).float())

    @staticmethod
    def sample_smolvla_time(batch_size: int, device: torch.device) -> torch.Tensor:
        """Exact `0.999 * Beta(1.5, 1.0) + 0.001` via analytic inverse-CDF (`u = r^(2/3)`)."""
        u_beta = torch.rand((batch_size,), device=device, dtype=torch.float32).clamp(min=1e-7).pow(1.0 / 1.5)
        return u_beta * 0.999 + 0.001

    def compute_crn_flow_and_aux_loss(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Dict[int, Dict[str, torch.Tensor]],
        ctx_tokens: Optional[torch.Tensor],
        coda_per_depth: List[Optional[torch.Tensor]],
        mem_mask: Optional[torch.Tensor],
        target_actions: torch.Tensor,                     # (B, chunk_size, 8) MEAN_STD float32
        *,
        aux_targets: Optional[torch.Tensor] = None,       # (B, 49) float32
        eta_aux: float = 0.1,
        time_override: Optional[torch.Tensor] = None,     # Optional (B,) in (0, 1]
        noise_override: Optional[torch.Tensor] = None,    # Optional (B, chunk_size, 32)
        grounding_xy_target: Optional[torch.Tensor] = None,
        grounding_xy_mask: Optional[torch.Tensor] = None,
        subgoal_emb_target: Optional[torch.Tensor] = None,
        subgoal_emb_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute CRN-shared `SmolVLA` Flow-Matching loss (`u_t = eps - a`, `t=1 -> 0`) across `l=0..L-1`."""
        B = int(target_actions.shape[0])
        device = target_actions.device
        act_8 = target_actions.float()
        act_32 = pad_vector(act_8, self.cfg.max_action_dim)

        time_t = (
            time_override.float()
            if time_override is not None
            else self.sample_smolvla_time(B, device)
        )
        noise_32 = (
            noise_override.float()
            if noise_override is not None
            else torch.randn_like(act_32)
        )
        t_exp = time_t[:, None, None]
        # SmolVLA flow path: t = 1.0 is pure noise, t = 0.0 is clean action!
        x_t = t_exp * noise_32 + (1.0 - t_exp) * act_32
        u_target_8 = (noise_32 - act_32)[:, :, : self.cfg.action_dim]  # (B, chunk_size, 8)

        use_grip = self.gripper_head is not None and self.cfg.gripper_bce_weight > 0.0
        # Note: in RoboMME MEAN_STD, raw gripper is in {-1, +1} with mean ~ -0.105, std ~ 0.994 -> >= 0.0 is open (+1)
        grip_target = (act_8[:, : self.cfg.exec_horizon, 7] >= 0.0).float() if use_grip else None
        w_u1 = self._u1_loss_weights

        L = len(coda_per_depth)
        grip_loss = torch.zeros((), device=device, dtype=torch.float32)

        if L == 1:
            if use_grip:
                v_32, grip_logits = self.denoise_step(
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    time_t,
                    mem_tokens=coda_per_depth[0],
                    mem_mask=mem_mask,
                    return_gripper_logits=True,
                )
            else:
                v_32 = self.denoise_step(
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    time_t,
                    mem_tokens=coda_per_depth[0],
                    mem_mask=mem_mask,
                )
                grip_logits = None

            sq_err = (v_32[:, :, : self.cfg.action_dim] - u_target_8) ** 2
            if w_u1 is not None:
                per_sample = torch.sum(sq_err * w_u1.unsqueeze(0), dim=(1, 2))
            else:
                per_sample = torch.mean(sq_err, dim=(1, 2))

            if grip_logits is not None and grip_target is not None:
                g_bce = F.binary_cross_entropy_with_logits(grip_logits, grip_target, reduction="none").mean(dim=-1)
                grip_loss = g_bce.mean()
                per_sample = per_sample + self.cfg.gripper_bce_weight * g_bce

            fm_loss = per_sample.mean()
            per_depth_sample_loss = per_sample.unsqueeze(1)
        else:
            # Fuse all L depths into ONE batched `lm_expert` forward reusing `past_key_values`!
            c_cat = torch.cat([c for c in coda_per_depth if c is not None], dim=0)
            x_rep = x_t.repeat(L, 1, 1)
            t_rep = time_t.repeat(L)
            mm_rep = mem_mask.repeat(L, 1) if mem_mask is not None else None

            if use_grip:
                v_flat, grip_flat = self.denoise_step(
                    prefix_pad_masks,
                    past_key_values,
                    x_rep,
                    t_rep,
                    mem_tokens=c_cat,
                    mem_mask=mm_rep,
                    return_gripper_logits=True,
                )
            else:
                v_flat = self.denoise_step(
                    prefix_pad_masks,
                    past_key_values,
                    x_rep,
                    t_rep,
                    mem_tokens=c_cat,
                    mem_mask=mm_rep,
                )
                grip_flat = None

            v_8 = v_flat[:, :, : self.cfg.action_dim].view(L, B, self.cfg.chunk_size, self.cfg.action_dim)
            sq_all = (v_8 - u_target_8.unsqueeze(0)) ** 2
            if w_u1 is not None:
                per_depth_sample_loss = torch.sum(
                    sq_all * w_u1.view(1, 1, self.cfg.chunk_size, self.cfg.action_dim), dim=(2, 3)
                ).transpose(0, 1)
            else:
                per_depth_sample_loss = torch.mean(sq_all, dim=(2, 3)).transpose(0, 1)

            if grip_flat is not None and grip_target is not None:
                g_all = grip_flat.view(L, B, self.cfg.exec_horizon)
                g_bce = F.binary_cross_entropy_with_logits(
                    g_all, grip_target.unsqueeze(0).expand(L, -1, -1), reduction="none"
                ).mean(dim=-1).transpose(0, 1)
                grip_loss = g_bce.mean()
                per_depth_sample_loss = per_depth_sample_loss + self.cfg.gripper_bce_weight * g_bce

            fm_loss = per_depth_sample_loss.mean()

        aux_loss = torch.zeros((), device=device, dtype=torch.float32)
        if self.aux_head is not None and aux_targets is not None and coda_per_depth[0] is not None:
            aux_dt = self.aux_head.weight.dtype
            aux_losses = []
            for c_l in coda_per_depth:
                assert c_l is not None
                c_ptr = self._select_pointer_coda(c_l)
                if mem_mask is not None:
                    mf = mem_mask.to(c_ptr.dtype).unsqueeze(-1)
                    c_pool = (c_ptr * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                else:
                    c_pool = c_ptr.mean(dim=1)
                pred_aux = self.aux_head(c_pool.to(aux_dt)).float()
                aux_losses.append(F.mse_loss(pred_aux, aux_targets.float()))
            aux_loss = torch.stack(aux_losses).mean()

        ground_loss = torch.zeros((), device=device, dtype=torch.float32)
        if (
            eta_aux > 0.0
            and self.grounding_xy_head is not None
            and ctx_tokens is not None
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
            "per_depth_fm_loss": per_depth_sample_loss,
        }

    # -------------------------------------------------------------------------
    # Inference Euler Sampler (`t = 1.0 -> 0.0`, `dt = -1 / num_steps`)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sample_action_chunk_from_kv(
        self,
        prefix_pad_masks: torch.Tensor,
        past_key_values: Dict[int, Dict[str, torch.Tensor]],
        *,
        mem_tokens: Optional[torch.Tensor] = None,
        mem_mask: Optional[torch.Tensor] = None,
        ode_steps: Optional[int] = None,
        noise_override: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Integrate `SmolVLA` ODE from `t=1.0` (`noise`) to `t=0.0` (`clean`), returning `(B, chunk_size, 8)`."""
        B = int(prefix_pad_masks.shape[0])
        device = prefix_pad_masks.device
        steps = max(1, int(self.cfg.num_ode_steps if ode_steps is None else ode_steps))
        dt = -1.0 / float(steps)

        if noise_override is not None:
            x_t = noise_override.to(device=device, dtype=torch.float32)
        else:
            x_t = torch.randn(
                (B, self.cfg.chunk_size, self.cfg.max_action_dim),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )

        for step in range(steps):
            t_val = 1.0 + step * dt
            t_tensor = torch.full((B,), t_val, dtype=torch.float32, device=device)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_t,
                t_tensor,
                mem_tokens=mem_tokens,
                mem_mask=mem_mask,
            )
            x_t = x_t + dt * v_t

        lim = float(self.cfg.sample_clamp)
        return x_t[:, :, : self.cfg.action_dim].clamp(-lim, lim)
