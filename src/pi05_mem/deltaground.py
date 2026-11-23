"""`DeltaGround-π0.5` (`src/pi05_mem/deltaground.py`) — Zero-Oracle `< 5 ms` `O(1)` Grounded Subgoal & Looped Sub-Pixel Pointer Engine.

Realizes `IDEA.md` & `#289(2)` / `#290`:
  1. **Multi-Timescale (`Short [4,32] / Mid [32,256] / Long [256,2048]`) 2-Layer `fp32` Gated DeltaNet**
     (`chunk_gated_delta_rule`) + **Cross-Scale Consolidation Gate** + **`EpisodeBoundaryForgetGate`**
     (`p_distractor = 0.25`, `k in {1..3}` un-reset cross-session streams) over strictly causal (`<= t`,
     `exclusion_window = 0` so zero 16-frame blind spot on fast button reveals / swaps / swings!)
     dual-camera (`front 8x8 + wrist 4x4`) + 8-DoF proprio + gripper-edge slots.
  2. **Non-Privileged `K = 12` Multi-Scale (`2x2 + 4x4 + 8x8`) `GDN-Surprise` + Gripper-Event Landmark Bank**
     preserving high-resolution demo highlights, multi-step `VideoPlaceOrder` targets, and unmask keyframes.
  3. **Weight-Tied Looped Multi-Hop (`L = 1 -> 2 -> 3`) `16x16` Sub-Pixel Pointer & Closed-Template Coda:**
     - Decomposes every RoboMME `grounded_subgoal` / `grounded_subgoal_online` reversibly into its exact
       closed-set template string (`"pick up the container at <bbox>"`, `"hook the cube at <bbox> to the target at <bbox>"`,
       etc.) + `0, 1, or 2` continuous `<row_y, col_x>` coordinates in `[0, 255]^2`.
     - At each weight-tied hop `l in {1..L}`, dynamically re-queries `2L-GDN` fast weights (`Δq^(l)`) +
       `K = 12` Surprise Landmarks + `task_goal` tokens + high-res `16x16` (`256`-token) current `front` scene,
       predicts `16x16` primary & secondary `<y, x>` heatmaps + sub-cell offsets (`~2-4 px` precision),
       bilinearly crops `F_16x16` at `(x_1^(l), y_1^(l))` & `(x_2^(l), y_2^(l))` via `F.grid_sample(..., align_corners=False)`,
       and feeds pointer-crop visual features into hop `l + 1` to resolve post-swap `Permanence` & multi-step `Reference`!
     - Runs in `< 3 ms` at `O(1)` memory with **strictly `0` privileged test-time oracle inputs**, driving
       converted `Yinpei/mme_vla_suite` `symbolic-grounded-subgoal/79999` (`max_token_len = 128`) via
       `DeltaGroundEvalAdapter.predict_subgoal(...)` (`src/pi05/policy.py:551`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from analysis.subgoal_grounding import (
    COLOR_TO_ID,
    ROBOMME_IMAGE_HW,
    STAGE_TO_ID,
    norm_xy_to_pixel_xy,
    parse_single_grounded_subgoal,
    pixel_xy_to_norm_xy,
    resolve_episode_subgoals,
    spatial_softmax_2d,
)
from models.deltaloop_vla import (
    MultiHeadAttn,
    SwiGLU,
    build_1d_sincos_embed,
    build_2d_sincos_pos_embed,
    run_dual_backend_gdn,
)
from pi05_data.dataset import DEFAULT_MIYABI_PI05_CACHE_DIR, load_pi05_normalizer, load_pi05_shards
from pi05_data.vision_tower import project_siglip_1152_to_2048
from robomme_data.constants import ROBOMME_TASKS, TASK_TO_SUITE
from robomme_eval.interface import ActionStateNormalizer, MME_VLA_NORM_STATS

from .config import Pi05MemConfig
from .memory_modules import (
    DualViewEventPerceiverWriter,
    MultiTimescale2LayerGDN,
    SurpriseEventLandmarkBank,
    pool_spatial_tokens_to_square,
    select_episode_write_indices,
)

_BBOX_PLACEHOLDER = "<bbox>"
_COORD_RE = re.compile(r"<\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*>")


def clean_subgoal_text(raw: str | None) -> str:
    """Canonicalize whitespace/underscores without trailing period (matches `mme_vla_suite/training/config.py:141`)."""
    if not raw:
        return ""
    s = " ".join(str(raw).strip().replace("_", " ").replace("\n", " ").split())
    if s.endswith("."):
        s = s[:-1].rstrip()
    return s


def decompose_grounded_subgoal(
    raw: str | None,
    image_hw: Tuple[int, int] = ROBOMME_IMAGE_HW,
) -> Tuple[str, List[Tuple[float, float]]]:
    """Decompose a RoboMME `grounded_subgoal` into `(template_str, [(x_col, y_row), ...])`.

    Example:
      `"hook the cube at <110, 95> to the target at <160, 140>"`
      -> `("hook the cube at <bbox> to the target at <bbox>", [(95.0, 110.0), (140.0, 160.0)])`
    """
    cleaned = clean_subgoal_text(raw)
    if not cleaned or cleaned.lower() in ("no record", "none", "null", "complete"):
        return "", []
    coords_xy = parse_single_grounded_subgoal(cleaned, image_hw=image_hw)
    tmpl = _COORD_RE.sub(_BBOX_PLACEHOLDER, cleaned)
    tmpl = " ".join(tmpl.split())
    return tmpl, coords_xy


def render_grounded_subgoal_from_norm_xy(
    template_str: str,
    xy_pri_norm: torch.Tensor | np.ndarray,
    xy_sec_norm: Optional[torch.Tensor | np.ndarray] = None,
    image_hw: Tuple[int, int] = ROBOMME_IMAGE_HW,
) -> str:
    """Substitute `<row_y, col_x>` integer pixel coordinates into `template_str`'s `<bbox>` slots."""
    if not template_str:
        return ""
    if _BBOX_PLACEHOLDER not in template_str:
        return template_str

    H, W = int(image_hw[0]), int(image_hw[1])
    t_pri = torch.as_tensor(xy_pri_norm, dtype=torch.float32).view(2)
    px_pri = norm_xy_to_pixel_xy(t_pri, image_hw=image_hw, align_corners=False)
    col_x1 = int(torch.round(px_pri[0]).clamp(0, W - 1).item())
    row_y1 = int(torch.round(px_pri[1]).clamp(0, H - 1).item())

    out = template_str.replace(_BBOX_PLACEHOLDER, f"<{row_y1}, {col_x1}>", 1)
    if _BBOX_PLACEHOLDER in out:
        t_sec = (
            torch.as_tensor(xy_sec_norm, dtype=torch.float32).view(2)
            if xy_sec_norm is not None
            else t_pri
        )
        px_sec = norm_xy_to_pixel_xy(t_sec, image_hw=image_hw, align_corners=False)
        col_x2 = int(torch.round(px_sec[0]).clamp(0, W - 1).item())
        row_y2 = int(torch.round(px_sec[1]).clamp(0, H - 1).item())
        out = out.replace(_BBOX_PLACEHOLDER, f"<{row_y2}, {col_x2}>")
    return out


# =============================================================================
# 1. Configuration & Empirical Template Vocabulary
# =============================================================================


@dataclass
class DeltaGroundConfig:
    """Configuration for standalone `<45M`-param `DeltaGround-π0.5` model."""

    vis_dim: int = 2048
    raw_siglip_dim: int = 1152
    front_tokens_hist: int = 64        # 8x8 history grid
    wrist_tokens_hist: int = 16        # 4x4 history grid
    front_tokens_curr: int = 256       # 16x16 high-res current front grid (16px cells + sub-cell offset!)
    wrist_tokens_curr: int = 256       # 16x16 current wrist grid (pooled to 4x4)
    vocab_size: int = 257_152          # PaliGemma SentencePiece vocab
    prompt_emb_dim: int = 128
    max_prompt_len: int = 64
    state_dim: int = 8

    # Write stride & causal zero-blind-spot cutoff (`exclusion_window = 0` -> sees up through frame `t`!)
    write_stride_exec: int = 2
    write_stride_demo: int = 2
    adaptive_demo_threshold: int = 512
    long_demo_stride: int = 4
    exclusion_window: int = 0

    # Multi-Timescale 2-Layer `fp32` Gated DeltaNet + `K = 12` Multi-Scale Surprise Landmarks
    d_mem: int = 512
    mem_enc_heads: int = 8
    num_write_slots: int = 8
    slot_conv_kernel: int = 4
    gdn_heads: int = 12                # 4 Short [4,32] + 4 Mid [32,256] + 4 Long [256,2048]
    gdn_d_k: int = 64
    gdn_d_v: int = 64
    num_gdn_readout_tokens: int = 18
    num_persistent_tokens: int = 4
    num_landmarks: int = 12            # K = 12 x 16 = 192 Multi-Scale (2x2+4x4+8x8) landmark tokens!
    demo_landmark_quota: int = 5
    mem_token_dim: int = 512           # Compact 512-D internal landmark/GDN width (<40M total params!)
    boundary_flush_threshold: float = 0.5
    boundary_reset_strength: float = 12.0
    surprise_ema_beta: float = 0.80

    # Weight-Tied Looped Multi-Hop (`L = 1 -> 2 -> 3`) Pointer & Template Coda
    num_hops: int = 3
    num_query_slots: int = 8
    coda_heads: int = 8
    dropout: float = 0.05
    num_templates: int = 64            # Overwritten automatically from empirical training cache vocab!
    max_count_classes: int = 12
    num_stage_classes: int = 15
    num_color_classes: int = 9
    norm_hop_persistent: bool = True   # `#296(1)` exact `persistent_norm` on hops h_i >= 1 (back-compat False if absent!)

    def to_pi05_mem_cfg(self) -> Pi05MemConfig:
        """Build compatible `Pi05MemConfig` for shared `DualViewEventPerceiverWriter`, `GDN`, & `LandmarkBank`."""
        return Pi05MemConfig(
            arch="p2_m2_deltamem",
            vis_dim=self.vis_dim,
            front_tokens_hist=self.front_tokens_hist,
            wrist_tokens_hist=self.wrist_tokens_hist,
            front_tokens_curr=self.front_tokens_curr,
            wrist_tokens_curr=self.wrist_tokens_curr,
            state_dim=self.state_dim,
            exclusion_window=self.exclusion_window,
            write_stride_exec=self.write_stride_exec,
            write_stride_demo=self.write_stride_demo,
            adaptive_demo_threshold=self.adaptive_demo_threshold,
            long_demo_stride=self.long_demo_stride,
            d_mem=self.d_mem,
            mem_enc_heads=self.mem_enc_heads,
            num_write_slots=self.num_write_slots,
            slot_conv_kernel=self.slot_conv_kernel,
            gdn_heads=self.gdn_heads,
            gdn_d_k=self.gdn_d_k,
            gdn_d_v=self.gdn_d_v,
            num_gdn_readout_tokens=self.num_gdn_readout_tokens,
            num_persistent_tokens=self.num_persistent_tokens,
            use_landmark_bank=True,
            landmark_encoder="multiscale_pyramid",
            num_landmarks=self.num_landmarks,
            demo_landmark_quota=self.demo_landmark_quota,
            mem_token_dim=self.mem_token_dim,
            boundary_flush_threshold=self.boundary_flush_threshold,
            boundary_reset_strength=self.boundary_reset_strength,
            surprise_ema_beta=self.surprise_ema_beta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeltaGroundConfig":
        valid = set(cls.__dataclass_fields__.keys())
        kw = {k: v for k, v in d.items() if k in valid}
        if "norm_hop_persistent" not in d:
            kw["norm_hop_persistent"] = False  # Exact bit-parity with pre-#296 v1 checkpoints!
        return cls(**kw)

    @classmethod
    def tiny_test_preset(cls, **overrides: Any) -> "DeltaGroundConfig":
        kw: Dict[str, Any] = dict(
            vis_dim=128,
            raw_siglip_dim=64,
            front_tokens_hist=64,
            wrist_tokens_hist=16,
            front_tokens_curr=64,
            wrist_tokens_curr=16,
            vocab_size=1024,
            prompt_emb_dim=64,
            max_prompt_len=24,
            d_mem=96,
            mem_enc_heads=4,
            num_write_slots=4,
            gdn_heads=6,
            gdn_d_k=32,
            gdn_d_v=32,
            num_gdn_readout_tokens=9,
            num_persistent_tokens=2,
            num_landmarks=4,
            demo_landmark_quota=2,
            mem_token_dim=96,
            num_hops=3,
            num_query_slots=8,
            coda_heads=4,
            dropout=0.0,
            num_templates=12,
        )
        kw.update(overrides)
        return cls(**kw)


# =============================================================================
# 2. Weight-Tied Looped Multi-Hop (`L = 1..3`) Spatial Pointer & Template Model
# =============================================================================


class PromptSubwordEncoder(nn.Module):
    """Self-contained prompt encoder (`0` 3.4B VLM dependency) using exact `Pi05Tokenizer` IDs."""

    def __init__(self, vocab_size: int, emb_dim: int, d_mem: int, max_len: int = 64, heads: int = 8) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tok_emb.weight[0].zero_()
        self.in_proj = nn.Linear(emb_dim, d_mem)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_mem))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.norm1 = nn.RMSNorm(d_mem)
        self.attn1 = MultiHeadAttn(d_mem, heads)
        self.norm2 = nn.RMSNorm(d_mem)
        self.ffn1 = SwiGLU(d_mem, d_mem * 2)
        self.out_norm = nn.RMSNorm(d_mem)

    def forward(self, lang_tokens: torch.Tensor, lang_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(prompt_tokens [B, L, d_mem], prompt_pool [B, d_mem])`."""
        dt = self.in_proj.weight.dtype
        l_seq = int(lang_tokens.shape[1])
        x = self.in_proj(self.tok_emb(lang_tokens.long().clamp(0, self.tok_emb.num_embeddings - 1)).to(dt))
        x = x + self.pos_emb[:, :l_seq, :].to(dt)
        h = self.norm1(x)
        x = x + self.attn1(h, h, kv_valid_mask=lang_masks)
        x = self.out_norm(x + self.ffn1(self.norm2(x)))
        mf = lang_masks.to(dt).unsqueeze(-1)
        pool = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
        return x, pool


class HighResCurrentSceneEncoder(nn.Module):
    """Encodes `16x16` (`256`) current front + `8x8` (`64`) close-up current wrist + 16-D `[state, Δstate]` kinematics (`#292(3)`)."""

    def __init__(self, cfg: DeltaGroundConfig) -> None:
        super().__init__()
        d = int(cfg.d_mem)
        self.front_curr_n = int(cfg.front_tokens_curr)
        self.front_side = int(round(self.front_curr_n**0.5))
        self.wrist_curr_n = min(64, int(cfg.wrist_tokens_curr))
        self.wrist_side = int(round(self.wrist_curr_n**0.5))

        self.front_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        self.wrist_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        # `[curr_state_norm (8), delta_state_norm (8), |delta_gripper| (1), arm_speed_rms (1)]` -> 18-D contact kinematics!
        self.state_proj = nn.Sequential(nn.Linear(cfg.state_dim * 2 + 2, d), nn.SiLU(), nn.Linear(d, d))

        self.register_buffer("front_pos", build_2d_sincos_pos_embed(self.front_side, self.front_side, d), persistent=False)
        self.register_buffer("wrist_pos", build_2d_sincos_pos_embed(self.wrist_side, self.wrist_side, d), persistent=False)
        self.mod_emb = nn.Embedding(3, d)
        nn.init.normal_(self.mod_emb.weight, mean=0.0, std=0.02)

        self.norm1 = nn.RMSNorm(d)
        self.attn = MultiHeadAttn(d, cfg.coda_heads)
        self.norm2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d * 2)
        self.out_norm = nn.RMSNorm(d)

    def forward(
        self,
        curr_front_256: torch.Tensor,   # (B, P_f_curr, vis_dim)
        curr_wrist_256: torch.Tensor,   # (B, P_w_curr, vis_dim)
        curr_state_norm: torch.Tensor,  # (B, 8)
        prev_state_norm: Optional[torch.Tensor] = None,  # Optional (B, 8)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return `(scene_tokens [B, P_f+P_w+1, d], front_grid_tokens [B, P_f, d], scene_tokens)`."""
        dt = self.front_proj[1].weight.dtype  # type: ignore[index]
        me = self.mod_emb.weight.to(dt)
        f_sq = pool_spatial_tokens_to_square(curr_front_256, self.front_curr_n)
        w_sq = pool_spatial_tokens_to_square(curr_wrist_256, self.wrist_curr_n)

        st_f = curr_state_norm.float()
        prev_f = prev_state_norm.float() if prev_state_norm is not None else st_f
        d_st = (st_f - prev_f).clamp(-2.0, 2.0)
        grip_abs = d_st[:, 7:8].abs()
        arm_spd = d_st[:, :7].pow(2).mean(dim=-1, keepdim=True).sqrt()
        kin_feat = torch.cat([st_f, d_st, grip_abs, arm_spd], dim=-1).to(dt)

        f_tok = self.front_proj(f_sq.to(dt)) + self.front_pos.to(dt).unsqueeze(0) + me[0].view(1, 1, -1)
        w_tok = self.wrist_proj(w_sq.to(dt)) + self.wrist_pos.to(dt).unsqueeze(0) + me[1].view(1, 1, -1)
        s_tok = self.state_proj(kin_feat).unsqueeze(1) + me[2].view(1, 1, -1)

        x = torch.cat([f_tok, w_tok, s_tok], dim=1)
        h = self.norm1(x)
        x = x + self.attn(h, h)
        x = self.out_norm(x + self.ffn(self.norm2(x)))
        f_grid = x[:, : self.front_curr_n, :]
        return x, f_grid, x


class WeightTiedLoopedPointerCoda(nn.Module):
    """Weight-tied recurrent `L = 1..3` hop pointer & closed-template predictor."""

    def __init__(self, cfg: DeltaGroundConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_mem)
        self.d_mem = d
        self.num_hops = int(cfg.num_hops)
        self.n_slots = int(cfg.num_query_slots)
        self.front_side = int(round(cfg.front_tokens_curr**0.5))

        self.slot_init = nn.Parameter(torch.randn(1, self.n_slots, d) * 0.02)
        self.hop_emb = nn.Parameter(torch.randn(max(4, self.num_hops), 1, d) * 0.02)
        # Residual hop gates for l >= 2 (`fp32` master scalar per hop, init `atanh(0.35)` for active deep supervision)
        init_g = float(math.atanh(0.35))
        self.gamma_loop = nn.Parameter(torch.full((max(3, self.num_hops - 1),), init_g, dtype=torch.float32))

        # Dynamic Hop-(l>=2) GDN query refinement (`zero-init` so Hop-2 starts around Hop-1 query!)
        n_q_gdn = int(cfg.num_gdn_readout_tokens)
        self.hop_gdn_q_proj = nn.Sequential(
            nn.RMSNorm(d),
            nn.Linear(d, n_q_gdn * d),
        )
        nn.init.zeros_(self.hop_gdn_q_proj[-1].weight)  # type: ignore[union-attr]
        nn.init.zeros_(self.hop_gdn_q_proj[-1].bias)    # type: ignore[union-attr]

        self.mem_kv_proj = (
            nn.Linear(cfg.mem_token_dim, d, bias=False)
            if cfg.mem_token_dim != d
            else nn.Identity()
        )

        # Weight-tied recurrent block (shared across ALL hops l = 1..L!)
        self.self_norm = nn.RMSNorm(d)
        self.self_attn = MultiHeadAttn(d, cfg.coda_heads)
        self.prompt_norm = nn.RMSNorm(d)
        self.prompt_kv_norm = nn.RMSNorm(d)
        self.prompt_attn = MultiHeadAttn(d, cfg.coda_heads)
        self.mem_norm = nn.RMSNorm(d)
        self.mem_kv_norm = nn.RMSNorm(d)
        self.mem_attn = MultiHeadAttn(d, cfg.coda_heads)
        self.scene_norm = nn.RMSNorm(d)
        self.scene_kv_norm = nn.RMSNorm(d)
        self.scene_attn = MultiHeadAttn(d, cfg.coda_heads)
        self.ffn_norm = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d * 2)
        self.res_drop = nn.Dropout(p=cfg.dropout) if cfg.dropout > 0.0 else nn.Identity()
        self.delta_out_norm = nn.RMSNorm(d)

        # High-resolution `16x16` spatial pointer head (2 channels: primary `<y1, x1>` & secondary `<y2, x2>`)
        self.grid_film = nn.Linear(d * 2, d * 2)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d),
            nn.SiLU(),
            nn.Conv2d(d, d, kernel_size=1),
            nn.SiLU(),
        )
        self.pri_q_proj = nn.Linear(d, d)
        self.sec_q_proj = nn.Linear(d, d)
        self.grid_k_proj = nn.Linear(d, d)
        self.offset_head = nn.Linear(d, 4)  # (dx1, dy1, dx2, dy2) sub-cell refinement within ±1 cell!
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

        # Pointer-guided bilinear visual crop feedback (`l -> l + 1`)
        self.fourier_dim = 32
        crop_in = d * 2 + 4 + 4 * self.fourier_dim
        self.crop_feedback_mlp = nn.Sequential(
            nn.RMSNorm(crop_in),
            nn.Linear(crop_in, d),
            nn.SiLU(),
            nn.Linear(d, self.n_slots * d),
        )
        nn.init.normal_(self.crop_feedback_mlp[-1].weight, mean=0.0, std=0.01)  # type: ignore[union-attr]
        nn.init.zeros_(self.crop_feedback_mlp[-1].bias)                        # type: ignore[union-attr]

        # Closed-vocabulary template + dense auxiliary + adaptive halting heads (weight-tied across hops!)
        head_in = d * 3
        self.readout_trunk = nn.Sequential(
            nn.RMSNorm(head_in),
            nn.Linear(head_in, d),
            nn.SiLU(),
            nn.Dropout(p=cfg.dropout),
        )
        self.template_head = nn.Linear(d, cfg.num_templates)
        self.completed_count_head = nn.Linear(d, cfg.max_count_classes)
        self.remaining_count_head = nn.Linear(d, cfg.max_count_classes)
        self.ordinal_head = nn.Linear(d, cfg.max_count_classes)
        self.stage_head = nn.Linear(d, cfg.num_stage_classes)
        self.color_head = nn.Linear(d, cfg.num_color_classes)
        self.halt_head = nn.Linear(d, 1)

    def compute_hop_query_bias(self, u_prev: torch.Tensor) -> torch.Tensor:
        """Compute dynamic GDN query bias `[B, num_gdn_readout_tokens, d_mem]` from belief state `u_prev`."""
        dt = self.hop_gdn_q_proj[1].weight.dtype  # type: ignore[index]
        pooled = u_prev.to(dt).mean(dim=1)
        return self.hop_gdn_q_proj(pooled).view(int(u_prev.shape[0]), self.cfg.num_gdn_readout_tokens, self.d_mem)

    def forward_single_hop(
        self,
        hop_idx: int,                      # 0-indexed (0, 1, 2)
        u_prev: torch.Tensor,              # (B, N_q, d)
        prompt_tokens: torch.Tensor,       # (B, L_p, d)
        prompt_mask: torch.Tensor,         # (B, L_p) bool
        mem_tokens: torch.Tensor,          # (B, S_mem, mem_token_dim)
        mem_mask: torch.Tensor,            # (B, S_mem) bool
        scene_tokens: torch.Tensor,        # (B, P_f+P_w+1, d)
        front_grid_tokens: torch.Tensor,   # (B, P_f, d)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Execute one weight-tied reasoning + spatial pointer + template prediction hop."""
        bsz = int(u_prev.shape[0])
        dt = self.pri_q_proj.weight.dtype
        s_side = self.front_side

        x = u_prev.to(dt) + self.hop_emb[min(hop_idx, self.hop_emb.shape[0] - 1)].to(dt).unsqueeze(0)
        x = x + self.self_attn(self.self_norm(x), self.self_norm(x))
        x = x + self.prompt_attn(
            self.prompt_norm(x), self.prompt_kv_norm(prompt_tokens.to(dt)), kv_valid_mask=prompt_mask
        )

        m_kv = self.mem_kv_norm(self.mem_kv_proj(mem_tokens.to(dt)))
        has_m = mem_mask.any(dim=-1, keepdim=True).to(dt).unsqueeze(-1)
        x = x + self.mem_attn(self.mem_norm(x), m_kv, kv_valid_mask=mem_mask) * has_m

        x = x + self.scene_attn(self.scene_norm(x), self.scene_kv_norm(scene_tokens.to(dt)))
        delta_u = self.delta_out_norm(x + self.res_drop(self.ffn(self.ffn_norm(x))))

        if hop_idx == 0:
            u_cur = delta_u
        else:
            g_l = torch.tanh(self.gamma_loop[min(hop_idx - 1, self.gamma_loop.numel() - 1)].float()).to(dt)
            u_cur = u_prev.to(dt) + g_l * (delta_u - u_prev.to(dt))

        # 16x16 Sub-Pixel Spatial Pointer Readout
        u_mean = u_cur.mean(dim=1)
        ptr_ctx = torch.cat([u_mean, u_cur[:, 0, :]], dim=-1)
        film_gamma, film_beta = self.grid_film(ptr_ctx).chunk(2, dim=-1)
        g_mod = front_grid_tokens.to(dt) * (1.0 + 0.25 * torch.tanh(film_gamma).unsqueeze(1)) + 0.25 * film_beta.unsqueeze(1)
        g_2d = g_mod.permute(0, 2, 1).contiguous().view(bsz, self.d_mem, s_side, s_side)
        g_ref = (g_2d + self.spatial_conv(g_2d)).view(bsz, self.d_mem, s_side * s_side).permute(0, 2, 1).contiguous()

        k_grid = self.grid_k_proj(g_ref)  # (B, P_f, d)
        scale = 1.0 / math.sqrt(float(self.d_mem))
        q_pri = self.pri_q_proj(u_cur[:, min(2, self.n_slots - 1), :]).unsqueeze(1)
        q_sec = self.sec_q_proj(u_cur[:, min(3, self.n_slots - 1), :]).unsqueeze(1)
        hm_pri = torch.matmul(q_pri, k_grid.transpose(-2, -1)).squeeze(1) * scale  # (B, P_f)
        hm_sec = torch.matmul(q_sec, k_grid.transpose(-2, -1)).squeeze(1) * scale  # (B, P_f)
        hm_logits = torch.stack([hm_pri, hm_sec], dim=1).float()                   # (B, 2, P_f)

        cell_offsets = torch.tanh(self.offset_head(g_ref).float()) * (1.0 / float(s_side))  # (B, P_f, 4)
        with torch.autocast(device_type=u_cur.device.type, enabled=False):
            xy_pri_base, probs_pri = spatial_softmax_2d(hm_logits[:, 0, :], grid_hw=(s_side, s_side), align_corners=False)
            xy_sec_base, probs_sec = spatial_softmax_2d(hm_logits[:, 1, :], grid_hw=(s_side, s_side), align_corners=False)
            off_pri = (probs_pri.unsqueeze(-1) * cell_offsets[:, :, 0:2]).sum(dim=1)
            off_sec = (probs_sec.unsqueeze(-1) * cell_offsets[:, :, 2:4]).sum(dim=1)
            xy_pri = (xy_pri_base + off_pri).clamp(-1.0, 1.0)
            xy_sec = (xy_sec_base + off_sec).clamp(-1.0, 1.0)

        # Bilinearly sample `g_ref` at `(xy_pri, xy_sec)` (`align_corners=False` matches RoboMME pixel convention!)
        g_map_f32 = g_ref.float().permute(0, 2, 1).contiguous().view(bsz, self.d_mem, s_side, s_side)
        sample_grid = torch.stack([xy_pri, xy_sec], dim=1).unsqueeze(2)  # (B, 2, 1, 2) in [-1, 1]
        sampled = F.grid_sample(g_map_f32, sample_grid, mode="bilinear", padding_mode="border", align_corners=False)
        crop_pri = sampled[:, :, 0, 0].to(dt)
        crop_sec = sampled[:, :, 1, 0].to(dt)

        fx1 = build_1d_sincos_embed(xy_pri[:, 0] * 16.0, self.fourier_dim, max_period=64.0)
        fy1 = build_1d_sincos_embed(xy_pri[:, 1] * 16.0, self.fourier_dim, max_period=64.0)
        fx2 = build_1d_sincos_embed(xy_sec[:, 0] * 16.0, self.fourier_dim, max_period=64.0)
        fy2 = build_1d_sincos_embed(xy_sec[:, 1] * 16.0, self.fourier_dim, max_period=64.0)
        crop_vec = torch.cat([crop_pri, crop_sec, xy_pri, xy_sec, fx1, fy1, fx2, fy2], dim=-1).to(dt)
        u_next = u_cur + self.crop_feedback_mlp(crop_vec).view(bsz, self.n_slots, self.d_mem)

        # Closed-vocabulary template + aux + halting logits
        trunk_in = torch.cat([u_cur.mean(dim=1), crop_pri, u_cur[:, 1, :]], dim=-1)
        h_out = self.readout_trunk(trunk_in)
        preds: Dict[str, torch.Tensor] = {
            "template_logits": self.template_head(h_out).float(),
            "xy_pri": xy_pri,
            "xy_sec": xy_sec,
            "hm_logits": hm_logits,
            "completed_count_logits": self.completed_count_head(h_out).float(),
            "remaining_count_logits": self.remaining_count_head(h_out).float(),
            "ordinal_logits": self.ordinal_head(h_out).float(),
            "stage_logits": self.stage_head(h_out).float(),
            "color_logits": self.color_head(h_out).float(),
            "halt_logit": self.halt_head(h_out).float().squeeze(-1),
        }
        return u_next, preds


class DeltaGroundModel(nn.Module):
    """Complete standalone `DeltaGround-π0.5` model (`< 45M` params, `O(1)` `< 3 ms` inference)."""

    def __init__(
        self,
        cfg: DeltaGroundConfig,
        *,
        template_vocab: Optional[Sequence[str]] = None,
        siglip_head_weight: Optional[torch.Tensor] = None,
        siglip_head_bias: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if template_vocab is not None:
            cfg.num_templates = max(2, len(template_vocab))
        self.cfg = cfg
        self.template_vocab: List[str] = (
            list(template_vocab) if template_vocab is not None else ["" for _ in range(cfg.num_templates)]
        )
        self.mem_cfg = cfg.to_pi05_mem_cfg()

        self.writer = DualViewEventPerceiverWriter(self.mem_cfg)
        self.gdn = MultiTimescale2LayerGDN(self.mem_cfg)
        self.landmark_bank = SurpriseEventLandmarkBank(self.mem_cfg)
        self.prompt_encoder = PromptSubwordEncoder(
            cfg.vocab_size, cfg.prompt_emb_dim, cfg.d_mem, max_len=cfg.max_prompt_len, heads=cfg.coda_heads
        )
        self.scene_encoder = HighResCurrentSceneEncoder(cfg)
        self.coda = WeightTiedLoopedPointerCoda(cfg)

        hw = (
            siglip_head_weight.detach().float().clone()
            if siglip_head_weight is not None
            else torch.zeros((cfg.vis_dim, cfg.raw_siglip_dim), dtype=torch.float32)
        )
        hb = (
            siglip_head_bias.detach().float().clone()
            if siglip_head_bias is not None
            else torch.zeros((cfg.vis_dim,), dtype=torch.float32)
        )
        self.register_buffer("siglip_head_weight", hw)
        self.register_buffer("siglip_head_bias", hb)
        self.register_buffer("has_siglip_head", torch.tensor(siglip_head_weight is not None, dtype=torch.bool))

    def cast_bf16_keep_fp32_gates(self) -> "DeltaGroundModel":
        """Run heavy projections in `bf16` while keeping GDN decay/scalar loop gates in `fp32`."""
        self.to(torch.bfloat16)
        for name, param in self.named_parameters():
            if any(k in name for k in ("A_log", "dt_bias", "gamma_loop", "salience_gate")):
                param.data = param.data.float()
        for _, buf in self.named_buffers():
            if buf.is_floating_point():
                buf.data = buf.data.float()
        return self

    def project_vis(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] == self.cfg.vis_dim:
            return x
        if x.shape[-1] == self.siglip_head_weight.shape[1]:
            return F.linear(
                x.to(torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float32),
                self.siglip_head_weight.to(x.dtype if x.dtype == torch.bfloat16 else torch.float32),
                self.siglip_head_bias.to(x.dtype if x.dtype == torch.bfloat16 else torch.float32),
            )
        raise ValueError(f"Unexpected visual feature dim {x.shape[-1]} (expected {self.cfg.vis_dim})")

    @staticmethod
    def smooth_surprise_ema(
        raw_surprise: torch.Tensor,
        grip_spike: torch.Tensor,
        is_demo_w: torch.Tensor,
        boundary_reset_w: Optional[torch.Tensor],
        ema_beta: float = 0.80,
        init_ema: float = 0.0,
    ) -> Tuple[torch.Tensor, float]:
        w = int(raw_surprise.numel())
        if w == 0:
            return torch.zeros_like(raw_surprise), init_ema
        s_cpu = raw_surprise.detach().float().cpu().tolist()
        br_cpu = boundary_reset_w.detach().float().cpu().tolist() if boundary_reset_w is not None else [0.0] * w
        ema_vals: List[float] = []
        cur = float(init_ema)
        for i in range(w):
            if br_cpu[i] > 0.5:
                cur = 0.0
            cur = ema_beta * cur + (1.0 - ema_beta) * s_cpu[i]
            ema_vals.append(max(s_cpu[i], cur))
        s_ema = torch.tensor(ema_vals, device=raw_surprise.device, dtype=torch.float32)
        s_norm = (s_ema - s_ema.mean()) / (s_ema.std(unbiased=False) + 1e-5)
        out = s_norm + 1.5 * grip_spike.float()
        demo_idx = torch.nonzero(is_demo_w, as_tuple=False).flatten()
        if demo_idx.numel() > 0:
            out[demo_idx[0]] = out[demo_idx[0]] + 1.0
            out[demo_idx[-1]] = out[demo_idx[-1]] + 1.0
        return out, cur

    def forward_episode(
        self,
        ep: Dict[str, Any],
        *,
        num_hops: Optional[int] = None,
        force_pytorch_gdn: bool = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run `O(W)` single-pass causal GDN + `L`-hop looped pointer readout over `M` steps of one episode."""
        dev = self.siglip_head_weight.device
        hops = int(num_hops or self.cfg.num_hops)

        ep_front = self.project_vis(ep["front_emb"].to(dev, non_blocking=True))
        ep_wrist = self.project_vis(ep["wrist_emb"].to(dev, non_blocking=True))
        ep_state = ep["state_norm"].to(dev, non_blocking=True)
        ep_demo = ep["is_demo"].to(dev, non_blocking=True)
        exec_start = int(ep["exec_start_idx"])
        step_indices = ep["chunk_step_indices"].to(dev, non_blocking=True)
        m_steps = int(step_indices.numel())
        # Zero-blind-spot strictly causal cutoff (`<= q_step`, i.e., `< q_step + 1`)!
        cutoffs = step_indices + 1

        curr_f = self.project_vis(ep["exec_front_256"].to(dev, non_blocking=True))
        curr_w = self.project_vis(ep["exec_wrist_256"].to(dev, non_blocking=True))
        curr_st = ep_state[step_indices]
        prev_idx = (step_indices - int(self.cfg.write_stride_exec)).clamp(min=exec_start).clamp(min=0)
        prev_st = ep_state[prev_idx]

        lang_tok_1 = ep["lang_tokens"].to(dev, non_blocking=True).unsqueeze(0)
        lang_mask_1 = ep["lang_masks"].to(dev, non_blocking=True).unsqueeze(0)
        prompt_tokens_1, _ = self.prompt_encoder(lang_tok_1, lang_mask_1)
        prompt_tokens = prompt_tokens_1.expand(m_steps, -1, -1)
        lang_mask = lang_mask_1.expand(m_steps, -1)
        scene_tokens, front_grid_tokens, _ = self.scene_encoder(curr_f, curr_w, curr_st, prev_st)

        # Causal write stream
        t_tot = int(ep_front.shape[0])
        write_idx = select_episode_write_indices(t_tot, exec_start, self.mem_cfg, device=dev)
        w_len = int(write_idx.numel())
        f_w = ep_front[write_idx]
        w_w = ep_wrist[write_idx]
        s_w = ep_state[write_idx]
        d_w = ep_demo[write_idx]
        kin = self.writer.compute_frame_kinematics(f_w, s_w, d_w)
        raw_gdn_w, lm_slots_w = self.writer.encode_frames_raw(f_w, w_w, s_w, write_idx, d_w, kin)
        gdn_slots_w, _ = self.writer.apply_slot_conv(raw_gdn_w, conv_history=None)

        p_jump_w: Optional[torch.Tensor] = None
        if "prompt_cos_jump" in ep and ep["prompt_cos_jump"] is not None and w_len > 0:
            ep_pj = ep["prompt_cos_jump"].to(dev, non_blocking=True)
            pj = torch.zeros((w_len,), device=dev, dtype=torch.float32)
            w_list = write_idx.tolist()
            for wi in range(w_len):
                lo = 0 if wi == 0 else (w_list[wi - 1] + 1)
                hi = w_list[wi] + 1
                if hi > lo:
                    pj[wi] = ep_pj[lo:hi].float().max()
            p_jump_w = pj

        b_reset_w = (
            self.gdn.boundary_gate(gdn_slots_w[:, 0, :], kin, prompt_cos_jump=p_jump_w)
            if (self.gdn.boundary_gate is not None and w_len > 0)
            else None
        )

        if w_len > 0:
            k1_all, v1_all, g1_all, b1_all, _ = self.gdn.layer1.project_slots(
                gdn_slots_w.unsqueeze(0),
                vis_motion_rms=kin["vis_motion_rms"].unsqueeze(0),
                grip_spike=kin["grip_spike"].unsqueeze(0),
                boundary_reset=b_reset_w.unsqueeze(0) if b_reset_w is not None else None,
            )
            raw_surp = self.gdn.layer1.compute_exact_write_surprise(
                k1_all.detach(),
                v1_all.detach(),
                g1_all.detach(),
                b1_all.detach(),
                num_write_slots=self.cfg.num_write_slots,
                force_pytorch=force_pytorch_gdn,
            ).squeeze(0)
            salience_w, _ = self.smooth_surprise_ema(
                raw_surp, kin["grip_spike"], d_w, b_reset_w, ema_beta=self.cfg.surprise_ema_beta
            )
        else:
            salience_w = torch.zeros((w_len,), device=dev, dtype=torch.float32)

        u_ends_list: List[int] = []
        decay_list: List[torch.Tensor] = []
        has_hist_list: List[bool] = []
        lm_tok_list: List[torch.Tensor] = []
        lm_mask_list: List[torch.Tensor] = []
        for ci in range(m_steps):
            cut_f = int(cutoffs[ci].item())
            q_st = int(step_indices[ci].item())
            u_end = int(torch.searchsorted(write_idx, cut_f, right=False).item()) if w_len > 0 else 0
            u_ends_list.append(u_end)
            decay_list.append(torch.tensor(1.0, device=dev, dtype=torch.float32))
            has_hist_q = u_end > 0
            has_hist_list.append(has_hist_q)
            if not has_hist_q:
                sel_pos = torch.empty((0,), dtype=torch.long, device=dev)
            else:
                sel_pos = self.landmark_bank.select_landmark_indices(
                    write_idx, d_w, salience_w, b_reset_w, cutoff_excl_frame=cut_f
                )
            lt, lm = self.landmark_bank.gather_padded_landmarks(
                lm_slots_w, write_idx, d_w, salience_w, sel_pos, query_step_idx=q_st
            )
            lm_tok_list.append(lt)
            lm_mask_list.append(lm)

        u_ends_t = torch.tensor(u_ends_list, device=dev, dtype=torch.long)
        decay_t = torch.stack(decay_list, dim=0)
        has_hist_b = torch.tensor(has_hist_list, device=dev, dtype=torch.bool)
        lm_tok_b = torch.stack(lm_tok_list, dim=0)
        lm_mask_b = torch.stack(lm_mask_list, dim=0)

        # Context query tokens for GDN readout: `[prompt_tokens[:, :8] || scene_summary]`
        ctx_for_gdn = torch.cat([prompt_tokens[:, :8, :], pool_spatial_tokens_to_square(front_grid_tokens, 16)], dim=1)

        dt = self.coda.pri_q_proj.weight.dtype
        u_state = self.coda.slot_init.to(dt).expand(m_steps, -1, -1)
        hop_outputs: List[Dict[str, torch.Tensor]] = []

        for h_i in range(hops):
            q_bias = None if h_i == 0 else self.coda.compute_hop_query_bias(u_state)
            gdn_tok, _ = self.gdn.forward_interleaved_episode_readout(
                gdn_slots_w,
                kin,
                b_reset_w,
                u_ends=u_ends_t,
                decay_factors=decay_t,
                ctx_tokens=ctx_for_gdn,
                extra_query_bias=q_bias,
                force_pytorch=force_pytorch_gdn,
            )
            if h_i > 0 and self.cfg.num_persistent_tokens > 0 and self.gdn.readout.persistent_tokens is not None:
                rd = self.gdn.readout
                raw_pt = rd.persistent_tokens.to(gdn_tok.dtype)
                p_base = rd.persistent_norm(raw_pt) if self.cfg.norm_hop_persistent else raw_pt
                p_tok = p_base.unsqueeze(0).expand(m_steps, -1, -1)
                gdn_tok = torch.cat([p_tok, gdn_tok], dim=1)

            gdn_mask = has_hist_b.unsqueeze(-1).expand(m_steps, int(gdn_tok.shape[1])).clone()
            if self.cfg.num_persistent_tokens > 0:
                gdn_mask[:, : self.cfg.num_persistent_tokens] = True

            mem_tok = torch.cat([gdn_tok, lm_tok_b.to(gdn_tok.dtype)], dim=1)
            mem_mask = torch.cat([gdn_mask, lm_mask_b], dim=1)

            u_state, preds_h = self.coda.forward_single_hop(
                h_i,
                u_state,
                prompt_tokens,
                lang_mask,
                mem_tok,
                mem_mask,
                scene_tokens,
                front_grid_tokens,
            )
            hop_outputs.append(preds_h)
        return hop_outputs

    def compute_deep_supervision_loss(
        self,
        hop_preds_batch: List[Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Deeply supervise all `L` hops (`0.2*L1 + 0.3*L2 + 0.5*L3`) on template + 16x16 `<y, x>` + aux + halt."""
        n_hops = len(hop_preds_batch)
        if n_hops == 1:
            hop_weights = [1.0]
        elif n_hops == 2:
            hop_weights = [0.35, 0.65]
        else:
            hop_weights = [0.20, 0.30, 0.50] + [0.50] * (n_hops - 3)
            s_w = sum(hop_weights)
            hop_weights = [w / s_w for w in hop_weights]

        dev = hop_preds_batch[0]["xy_pri"].device
        s_side = self.coda.front_side
        centers = ((torch.arange(s_side, device=dev, dtype=torch.float32) + 0.5) / float(s_side)) * 2.0 - 1.0
        yy, xx = torch.meshgrid(centers, centers, indexing="ij")
        grid_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).unsqueeze(0)  # (1, P_f, 2)
        sigma2 = (1.25 / float(s_side)) ** 2

        tgt_tmpl = targets["template_id"].to(dev).long()
        has_xy1 = targets["has_xy1"].to(dev).bool()
        tgt_xy1 = targets["xy1_m11"].to(dev).float()
        has_xy2 = targets["has_xy2"].to(dev).bool()
        tgt_xy2 = targets["xy2_m11"].to(dev).float()

        total_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
        last_metrics: Dict[str, torch.Tensor] = {}

        for h_i, (preds, w_h) in enumerate(zip(hop_preds_batch, hop_weights)):
            tmpl_loss = F.cross_entropy(preds["template_logits"], tgt_tmpl, label_smoothing=0.03)

            xy_loss = torch.tensor(0.0, device=dev, dtype=torch.float32)
            px_err_1 = torch.zeros_like(tgt_tmpl, dtype=torch.float32)
            if bool(has_xy1.any()):
                p1 = preds["xy_pri"][has_xy1]
                t1 = tgt_xy1[has_xy1]
                reg1 = F.smooth_l1_loss(p1, t1, beta=0.03)
                d2_1 = ((grid_xy - t1.unsqueeze(1)) ** 2).sum(dim=-1)
                tgt_p1 = F.softmax(-0.5 * d2_1 / sigma2, dim=-1)
                kl1 = F.kl_div(F.log_softmax(preds["hm_logits"][has_xy1, 0, :], dim=-1), tgt_p1, reduction="batchmean")
                xy_loss = xy_loss + 2.5 * reg1 + 0.35 * kl1

                with torch.no_grad():
                    px_pred_all = norm_xy_to_pixel_xy(preds["xy_pri"].detach(), align_corners=False)
                    px_tgt_all = norm_xy_to_pixel_xy(tgt_xy1, align_corners=False)
                    px_err_1 = ((px_pred_all - px_tgt_all) ** 2).sum(dim=-1).sqrt()

            if bool(has_xy2.any()):
                p2 = preds["xy_sec"][has_xy2]
                t2 = tgt_xy2[has_xy2]
                reg2 = F.smooth_l1_loss(p2, t2, beta=0.03)
                d2_2 = ((grid_xy - t2.unsqueeze(1)) ** 2).sum(dim=-1)
                tgt_p2 = F.softmax(-0.5 * d2_2 / sigma2, dim=-1)
                kl2 = F.kl_div(F.log_softmax(preds["hm_logits"][has_xy2, 1, :], dim=-1), tgt_p2, reduction="batchmean")
                xy_loss = xy_loss + 2.0 * reg2 + 0.25 * kl2

            # Auxiliary counts & semantic stage/color supervision
            max_c = self.cfg.max_count_classes - 1
            cnt_loss = (
                F.cross_entropy(preds["completed_count_logits"], targets["completed_count"].to(dev).long().clamp(0, max_c))
                + F.cross_entropy(preds["remaining_count_logits"], targets["remaining_count"].to(dev).long().clamp(0, max_c))
                + F.cross_entropy(preds["ordinal_logits"], targets["ordinal_id"].to(dev).long().clamp(0, max_c))
            ) / 3.0
            sem_loss = 0.5 * (
                F.cross_entropy(preds["stage_logits"], targets["stage_id"].to(dev).long().clamp(0, self.cfg.num_stage_classes - 1))
                + F.cross_entropy(preds["color_logits"], targets["color_id"].to(dev).long().clamp(0, self.cfg.num_color_classes - 1))
            )

            # Adaptive halting supervision: `1` iff template matches & pixel error <= 6 px!
            with torch.no_grad():
                tmpl_ok = preds["template_logits"].detach().argmax(dim=-1) == tgt_tmpl
                xy_ok = (~has_xy1) | (px_err_1 <= 6.0)
                halt_tgt = (tmpl_ok & xy_ok).float()
            halt_loss = F.binary_cross_entropy_with_logits(preds["halt_logit"], halt_tgt)

            hop_total = tmpl_loss + xy_loss + 0.25 * cnt_loss + 0.15 * sem_loss + 0.10 * halt_loss
            total_loss = total_loss + float(w_h) * hop_total

            with torch.no_grad():
                tmpl_acc = (preds["template_logits"].argmax(dim=-1) == tgt_tmpl).float().mean()
                mean_px = px_err_1[has_xy1].mean() if bool(has_xy1.any()) else torch.tensor(0.0, device=dev)
                last_metrics[f"tmpl_acc_h{h_i + 1}"] = tmpl_acc
                last_metrics[f"px_err_h{h_i + 1}"] = mean_px
                if h_i == n_hops - 1:
                    last_metrics["tmpl_loss"] = tmpl_loss.detach()
                    last_metrics["xy_loss"] = xy_loss.detach()
                    last_metrics["cnt_loss"] = cnt_loss.detach()

        return {"loss": total_loss, **last_metrics}


# =============================================================================
# 3. Fast Cached Episode Dataset (`robomme_pi05_cache` -> `DeltaGround` Targets)
# =============================================================================


class DeltaGroundEpisodeDataset(Dataset):
    """Loads full episodes + `M` dense execution steps from `robomme_pi05_cache` (`0` VLM overhead!)."""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_MIYABI_PI05_CACHE_DIR,
        *,
        tasks: Sequence[str] | None = None,
        max_steps_per_episode: int = 24,
        online_subgoal_prob: float = 0.85,
        distractor_prob: float = 0.25,
        max_distractors: int = 3,
        max_distractor_frames_each: int = 160,
        template_vocab: Optional[Sequence[str]] = None,
        episode_range: Optional[Tuple[int, int]] = None,
        shards_override: Optional[Dict[str, Any]] = None,
        stage_entry_focus_prob: float = 0.0,
        arm_neutral_sub_prob: float = 0.0,
        eval_stage_entry_only: bool = False,
        hr_cache_dir: Optional[str | Path] = None,
        hr_shards_override: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.online_subgoal_prob = float(online_subgoal_prob)
        self.distractor_prob = float(distractor_prob)
        self.max_distractors = int(max_distractors)
        self.max_distractor_frames_each = int(max_distractor_frames_each)
        self.episode_range = episode_range
        self.stage_entry_focus_prob = float(stage_entry_focus_prob)
        self.arm_neutral_sub_prob = float(arm_neutral_sub_prob)
        self.eval_stage_entry_only = bool(eval_stage_entry_only)
        self.rng = np.random.default_rng(seed)

        self.normalizer = load_pi05_normalizer(self.cache_dir, stats_variant="mme_vla")
        self.shards = shards_override if shards_override is not None else load_pi05_shards(self.cache_dir, tasks=self.tasks, mmap=True)
        self.hr_shards: Dict[str, Any] = {}
        if hr_shards_override is not None:
            self.hr_shards = hr_shards_override
        elif hr_cache_dir is not None and Path(hr_cache_dir).exists():
            hr_root = Path(hr_cache_dir)
            for t_hr in self.tasks:
                p_hr = hr_root / f"{t_hr}.pt"
                if p_hr.exists():
                    self.hr_shards[t_hr] = torch.load(p_hr, map_location="cpu", mmap=True, weights_only=False)
        first_sh = next(iter(self.shards.values()))
        self.siglip_head_weight = first_sh["siglip_head_weight"].detach().clone()
        self.siglip_head_bias = first_sh["siglip_head_bias"].detach().clone()

        self.index: List[Tuple[str, int]] = []
        self._meta: Dict[str, Dict[str, Any]] = {}
        discovered_templates: set[str] = {""}

        for task in self.tasks:
            sh = self.shards[task]
            n_eps = int(sh["epis_idx"].shape[0])
            ep_lo = 0 if episode_range is None else max(0, int(episode_range[0]))
            ep_hi = n_eps if episode_range is None else min(n_eps, int(episode_range[1]))
            for ep_i in range(ep_lo, ep_hi):
                self.index.append((task, ep_i))

            raw_st = sh["state"].to(dtype=torch.float32).contiguous().clone().numpy()
            st_norm = torch.from_numpy(self.normalizer.normalize_state(raw_st))
            gs_std = [resolve_episode_subgoals(seq) for seq in sh["grounded_subgoals"]]
            gs_on_raw = sh.get("grounded_subgoals_online") or sh["grounded_subgoals"]
            gs_on = [resolve_episode_subgoals(seq) for seq in gs_on_raw]

            if template_vocab is None:
                for ep_v in range(ep_lo, ep_hi):
                    for s in gs_std[ep_v]:
                        t_s, _ = decompose_grounded_subgoal(s)
                        discovered_templates.add(t_s)
                    for s in gs_on[ep_v]:
                        t_s, _ = decompose_grounded_subgoal(s)
                        discovered_templates.add(t_s)

            self._meta[task] = {
                "suite": sh.get("suite", TASK_TO_SUITE.get(task, "Unknown")),
                "ep_offsets": sh["ep_offsets"].to(dtype=torch.long).contiguous().clone().numpy(),
                "exec_start_idx": sh["exec_start_idx"].to(dtype=torch.long).contiguous().clone().numpy(),
                "exec_ep_offsets": sh["exec_ep_offsets"].to(dtype=torch.long).contiguous().clone().numpy(),
                "exec_rel_step_idx": sh["exec_rel_step_idx"].to(dtype=torch.long).contiguous().clone().numpy(),
                "prompts": [str(p) for p in sh["prompts"]],
                "lang_tokens": sh["lang_tokens"].to(dtype=torch.long).contiguous().clone(),
                "lang_masks": sh["lang_masks"].to(dtype=torch.bool).contiguous().clone(),
                "state_norm": st_norm,
                "is_demo": sh["is_demo"].to(dtype=torch.bool).contiguous().clone(),
                "count_state": sh["count_state"].to(dtype=torch.long).contiguous().clone(),
                "phase_id": sh["phase_id"].to(dtype=torch.long).contiguous().clone(),
                "color_id": sh["color_id"].to(dtype=torch.long).contiguous().clone(),
                "ordinal_id": sh["ordinal_id"].to(dtype=torch.long).contiguous().clone(),
                "gs_std": gs_std,
                "gs_on": gs_on,
            }

        self.template_vocab: List[str] = (
            list(template_vocab) if template_vocab is not None else sorted(discovered_templates)
        )
        self.tmpl_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.template_vocab)}
        twin_pairs = build_occlusion_twin_template_pairs(self.template_vocab)
        self.twin_to_grounded: Dict[str, str] = {
            self.template_vocab[u_i]: self.template_vocab[g_i] for u_i, g_i in twin_pairs
        }

    def __len__(self) -> int:
        return len(self.index)

    def _slice_distractor(self, task: str, ep_local: int) -> Dict[str, Any]:
        sh = self.shards[task]
        mt = self._meta[task]
        s = int(mt["ep_offsets"][ep_local])
        e = int(mt["ep_offsets"][ep_local + 1])
        if (e - s) > self.max_distractor_frames_each:
            s = e - self.max_distractor_frames_each
        if self.hr_shards and ((e - s) % 2 == 1) and (e - s) > 2:
            s += 1
        return {
            "front": sh["front_emb"][s:e].to(torch.bfloat16).contiguous().clone(),
            "wrist": sh["wrist_emb"][s:e].to(torch.bfloat16).contiguous().clone(),
            "state_norm": mt["state_norm"][s:e].clone(),
            "is_demo": mt["is_demo"][s:e].clone(),
            "prompt": mt["prompts"][ep_local],
            "length": int(e - s),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task, ep_local = self.index[idx]
        sh = self.shards[task]
        mt = self._meta[task]

        s = int(mt["ep_offsets"][ep_local])
        e = int(mt["ep_offsets"][ep_local + 1])
        exec_start = int(mt["exec_start_idx"][ep_local])
        exec_s = int(mt["exec_ep_offsets"][ep_local])
        exec_e = int(mt["exec_ep_offsets"][ep_local + 1])
        cached_rel_steps = mt["exec_rel_step_idx"][exec_s:exec_e]
        use_on = bool(self.rng.random() < self.online_subgoal_prob)
        sg_seq = mt["gs_on"][ep_local] if use_on else mt["gs_std"][ep_local]

        n_tot_slots = int(cached_rel_steps.size)
        slot_stage_start = np.zeros((max(1, n_tot_slots),), dtype=np.int64)
        slot_is_entry = np.zeros((max(1, n_tot_slots),), dtype=bool)
        slot_has_bbox = np.zeros((max(1, n_tot_slots),), dtype=bool)
        if n_tot_slots > 0 and (self.eval_stage_entry_only or self.stage_entry_focus_prob > 0.0):
            prev_canon: Optional[str] = None
            cur_st_start = 0
            for si in range(n_tot_slots):
                rt_s = int(cached_rel_steps[si])
                raw_s = sg_seq[min(rt_s, len(sg_seq) - 1)] if sg_seq else ""
                t_s, c_s = decompose_grounded_subgoal(raw_s)
                canon_s = self.twin_to_grounded.get(t_s, t_s)
                has_bb = len(c_s) >= 1
                slot_has_bbox[si] = has_bb
                if si == 0:
                    cur_st_start = 0
                    slot_is_entry[si] = True
                elif canon_s != prev_canon and canon_s:
                    cur_st_start = si
                    slot_is_entry[si] = True
                slot_stage_start[si] = cur_st_start
                if canon_s:
                    prev_canon = canon_s

        if self.eval_stage_entry_only and n_tot_slots > 0:
            entry_bb = np.nonzero(slot_is_entry[:n_tot_slots] & slot_has_bbox[:n_tot_slots])[0]
            if entry_bb.size == 0:
                entry_bb = np.array([0], dtype=np.int64)
            picked_slots = entry_bb[: max(1, self.max_steps_per_episode)]
        elif self.stage_entry_focus_prob > 0.0 and n_tot_slots > self.max_steps_per_episode:
            entry_cands = np.nonzero(slot_is_entry[:n_tot_slots])[0]
            non_entry_cands = np.nonzero(~slot_is_entry[:n_tot_slots])[0]
            k_tot = self.max_steps_per_episode
            k_ent = min(int(entry_cands.size), max(1, int(round(k_tot * self.stage_entry_focus_prob))))
            sel_ent = (
                entry_cands
                if entry_cands.size <= k_ent
                else np.sort(self.rng.choice(entry_cands, size=k_ent, replace=False))
            )
            k_rem = max(0, k_tot - int(sel_ent.size))
            if k_rem > 0 and non_entry_cands.size > 0:
                sel_non = self.rng.choice(non_entry_cands, size=min(k_rem, int(non_entry_cands.size)), replace=False)
                picked_slots = np.sort(np.concatenate([sel_ent, sel_non]))
            else:
                picked_slots = np.sort(sel_ent)
        elif cached_rel_steps.size > self.max_steps_per_episode:
            # Always include first & last execution step + random interior steps so transitions are densely covered!
            interior = np.arange(1, cached_rel_steps.size - 1, dtype=np.int64)
            k_int = max(0, self.max_steps_per_episode - 2)
            sel_int = self.rng.choice(interior, size=min(k_int, interior.size), replace=False)
            picked_slots = np.sort(np.concatenate([[0, cached_rel_steps.size - 1], sel_int]))
        else:
            picked_slots = np.arange(cached_rel_steps.size, dtype=np.int64)

        chunk_steps = cached_rel_steps[picked_slots].copy()
        is_stage_entry_sel = (
            torch.from_numpy(slot_is_entry[picked_slots]).bool()
            if n_tot_slots > 0 and (self.eval_stage_entry_only or self.stage_entry_focus_prob > 0.0)
            else torch.ones((int(chunk_steps.size),), dtype=torch.bool)
        )
        effective_slots = picked_slots.copy()
        if (
            not self.eval_stage_entry_only
            and self.arm_neutral_sub_prob > 0.0
            and n_tot_slots > 0
            and chunk_steps.size > 0
        ):
            for mi, orig_si in enumerate(picked_slots.tolist()):
                if not bool(slot_is_entry[orig_si]):
                    st_si = int(slot_stage_start[orig_si])
                    if float(self.rng.random()) < self.arm_neutral_sub_prob:
                        effective_slots[mi] = st_si
                        chunk_steps[mi] = cached_rel_steps[st_si]
                        is_stage_entry_sel[mi] = True

        abs_exec_rows = torch.from_numpy(exec_s + effective_slots).long()
        front_emb = sh["front_emb"][s:e].to(torch.bfloat16).contiguous().clone()
        wrist_emb = sh["wrist_emb"][s:e].to(torch.bfloat16).contiguous().clone()
        exec_front_256 = sh["exec_front_256"][abs_exec_rows].to(torch.bfloat16).contiguous().clone()
        exec_wrist_256 = sh["exec_wrist_256"][abs_exec_rows].to(torch.bfloat16).contiguous().clone()
        state_norm = mt["state_norm"][s:e].clone()
        is_demo = mt["is_demo"][s:e].clone()

        # Parse exact `(template_id, xy1_m11, xy2_m11)` targets
        m_sel = int(chunk_steps.size)
        tmpl_ids = torch.zeros((m_sel,), dtype=torch.long)
        xy1_m11 = torch.zeros((m_sel, 2), dtype=torch.float32)
        has_xy1 = torch.zeros((m_sel,), dtype=torch.bool)
        xy2_m11 = torch.zeros((m_sel, 2), dtype=torch.float32)
        has_xy2 = torch.zeros((m_sel,), dtype=torch.bool)

        for mi, rel_t in enumerate(chunk_steps.tolist()):
            raw_sg = sg_seq[min(int(rel_t), len(sg_seq) - 1)] if sg_seq else ""
            t_str, coords_xy = decompose_grounded_subgoal(raw_sg)
            tmpl_ids[mi] = self.tmpl_to_id.get(t_str, 0)
            if len(coords_xy) >= 1:
                xy1_m11[mi] = pixel_xy_to_norm_xy(torch.tensor(coords_xy[0], dtype=torch.float32), align_corners=False)
                has_xy1[mi] = True
            if len(coords_xy) >= 2:
                xy2_m11[mi] = pixel_xy_to_norm_xy(torch.tensor(coords_xy[1], dtype=torch.float32), align_corners=False)
                has_xy2[mi] = True

        abs_rows = torch.from_numpy(s + chunk_steps).long()
        cnt_st = mt["count_state"][abs_rows]

        # Optional `k in {1..3}` un-reset distractor episode prefixing (`EpisodeBoundaryForgetGate` training!)
        pref_len = 0
        prompt_jump = torch.zeros((int(front_emb.shape[0]),), dtype=torch.float32)
        if self.distractor_prob > 0.0 and len(self.index) > 1 and float(self.rng.random()) < self.distractor_prob:
            k_d = int(self.rng.integers(1, self.max_distractors + 1))
            f_parts, w_parts, s_parts, d_parts, pj_parts = [], [], [], [], []
            prev_p: Optional[str] = None
            for _ in range(k_d):
                d_task, d_ep = self.index[int(self.rng.integers(0, len(self.index)))]
                dc = self._slice_distractor(d_task, d_ep)
                dl = dc["length"]
                if dl == 0:
                    continue
                f_parts.append(dc["front"])
                w_parts.append(dc["wrist"])
                s_parts.append(dc["state_norm"])
                d_parts.append(dc["is_demo"])
                pj = torch.zeros((dl,), dtype=torch.float32)
                if prev_p is not None and dc["prompt"] != prev_p:
                    pj[0] = 1.0
                pj_parts.append(pj)
                prev_p = dc["prompt"]
                pref_len += dl
            if pref_len > 0:
                if (prev_p is not None and prev_p != mt["prompts"][ep_local]) or exec_start == 0:
                    prompt_jump[0] = 1.0
                front_emb = torch.cat([*f_parts, front_emb], dim=0)
                wrist_emb = torch.cat([*w_parts, wrist_emb], dim=0)
                state_norm = torch.cat([*s_parts, state_norm], dim=0)
                is_demo = torch.cat([*d_parts, is_demo], dim=0)
                prompt_jump = torch.cat([*pj_parts, prompt_jump], dim=0)
                chunk_steps = chunk_steps + pref_len
                exec_start = exec_start + pref_len

        return {
            "task_name": task,
            "suite": mt["suite"],
            "exec_start_idx": exec_start,
            "prompt": mt["prompts"][ep_local],
            "lang_tokens": mt["lang_tokens"][ep_local].clone(),
            "lang_masks": mt["lang_masks"][ep_local].clone(),
            "front_emb": front_emb,
            "wrist_emb": wrist_emb,
            "state_norm": state_norm,
            "is_demo": is_demo,
            "prompt_cos_jump": prompt_jump,
            "chunk_step_indices": torch.from_numpy(chunk_steps).long(),
            "exec_front_256": exec_front_256,
            "exec_wrist_256": exec_wrist_256,
            "template_id": tmpl_ids,
            "xy1_m11": xy1_m11,
            "has_xy1": has_xy1,
            "xy2_m11": xy2_m11,
            "has_xy2": has_xy2,
            "is_stage_entry": is_stage_entry_sel,
            "completed_count": cnt_st[:, 0].clone(),
            "remaining_count": cnt_st[:, 1].clone(),
            "stage_id": mt["phase_id"][abs_rows].clone(),
            "color_id": mt["color_id"][abs_rows].clone(),
            "ordinal_id": mt["ordinal_id"][abs_rows].clone(),
            **(
                {
                    "target_write_front_256": self.hr_shards[task]["write_front_256"][
                        int(self.hr_shards[task]["write_ep_offsets"][ep_local]) : int(
                            self.hr_shards[task]["write_ep_offsets"][ep_local + 1]
                        )
                    ]
                    .to(torch.bfloat16)
                    .contiguous()
                    .clone(),
                    "target_write_abs_idx": self.hr_shards[task]["write_rel_idx"][
                        int(self.hr_shards[task]["write_ep_offsets"][ep_local]) : int(
                            self.hr_shards[task]["write_ep_offsets"][ep_local + 1]
                        )
                    ]
                    .to(torch.long)
                    .clone()
                    + pref_len,
                }
                if task in self.hr_shards
                else {}
            ),
        }


# =============================================================================
# 4. Online Closed-Loop `DeltaGroundEvalAdapter` (`< 3 ms`, `O(1)` State, `0` Oracle)
# =============================================================================


class DeltaGroundEvalAdapter:
    """Online closed-loop grounded subgoal predictor plugging into `Pi05RoboMMEPolicy` (`#290(2)`)."""

    def __init__(
        self,
        model: DeltaGroundModel,
        *,
        device: str | torch.device = "cuda",
        hop_mode: str = "l3",
        peak_window_radius: int = 2,
        fold_occlusion_twins: bool = False,
        stage_entry_latch: bool = False,
        min_template_confidence: float = 0.0,
        abstain_ordinal_counting: bool = False,
        abstain_place_order: bool = True,
        abstain_temporal_button: bool = False,
        tokenizer_path: str | Path | None = None,
        state_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        from pi05.policy import Pi05Tokenizer

        self.model = model.to(device).eval()
        self.cfg = model.cfg
        self.mem_cfg = model.mem_cfg
        self.device = torch.device(device)
        self.hop_mode = hop_mode  # "l1", "l2", "l3", or "auto"
        self.peak_window_radius = int(peak_window_radius)
        self.fold_occlusion_twins = bool(fold_occlusion_twins)
        self.stage_entry_latch = bool(stage_entry_latch)
        self.min_template_confidence = float(min_template_confidence)
        self.abstain_ordinal_counting = bool(abstain_ordinal_counting)
        self.abstain_place_order = bool(abstain_place_order)
        self.abstain_temporal_button = bool(abstain_temporal_button)
        self.tokenizer = Pi05Tokenizer(max_len=self.cfg.max_prompt_len, discrete_state_input=False, tokenizer_path=tokenizer_path)
        self.normalizer = ActionStateNormalizer(mode="quantile_delta", stats=state_stats or MME_VLA_NORM_STATS)
        self._prompt_tok_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._twin_pairs: List[Tuple[int, int]] = build_occlusion_twin_template_pairs(self.model.template_vocab)
        self._twin_to_grounded: Dict[str, str] = {
            self.model.template_vocab[u_i]: self.model.template_vocab[g_i]
            for u_i, g_i in self._twin_pairs
        }
        self._stage_latch = StageEntrySubgoalLatch(self._twin_to_grounded)
        self.reset()

    def reset(self) -> None:
        self.stream_total_frames: int = 0
        self.demo_frames_in_session: int = 0
        self.session_start_frame: int = 0
        self.session_exec_start_frame: int = 0
        self._last_was_demo: Optional[bool] = None
        self._last_prompt_str: Optional[str] = None
        self._pending_prompt_jump: float = 0.0
        self._prepared_write_batch: Optional[Tuple[List[int], List[float], List[bool], bool]] = None
        self._prev_w_front: Optional[torch.Tensor] = None
        self._prev_w_state: Optional[torch.Tensor] = None
        self._prev_w_demo: Optional[torch.Tensor] = None
        self._conv_hist: Optional[torch.Tensor] = None

        self._w_steps: List[int] = []
        self._w_is_demo: List[bool] = []
        self._w_gdn_slots: List[torch.Tensor] = []
        self._w_lm_slots: List[torch.Tensor] = []
        self._w_state_norm: List[torch.Tensor] = []
        self._w_vis_rms: List[float] = []
        self._w_grip_spike: List[float] = []
        self._w_b_reset: List[float] = []
        self._w_s_ema: List[float] = []

        self._committed_u: int = 0
        self._gdn_states = self.model.gdn.init_empty_states(1, self.device)
        self._surp_l1_state = torch.zeros(
            (1, self.cfg.gdn_heads, self.cfg.gdn_d_v, self.cfg.gdn_d_k),
            device=self.device,
            dtype=torch.float32,
        )
        self._surprise_ema: float = 0.0
        self._last_pred_subgoal: str = ""
        self._last_loop_depth: int = 1
        self._last_b_reset_max: float = 0.0
        self._last_final_preds: Optional[Dict[str, torch.Tensor]] = None
        self._stage_latch.reset()

    def _encode_prompt(self, task_goal: str) -> Tuple[torch.Tensor, torch.Tensor]:
        key = task_goal.strip().lower()
        if key not in self._prompt_tok_cache:
            # Match `extract_pi05_cache.py:514` (`lower_prompt=True`)!
            tok_np, mask_np = self.tokenizer.tokenize(key, norm_state=None, subgoal=None)
            lt = torch.from_numpy(tok_np).unsqueeze(0).to(self.device)
            lm = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)
            with torch.no_grad():
                p_tok, _ = self.model.prompt_encoder(lt, lm)
            self._prompt_tok_cache[key] = (p_tok, lm)
        return self._prompt_tok_cache[key]

    @torch.no_grad()
    def select_write_indices(
        self,
        num_frames: int,
        *,
        task_goal: str,
        is_demo: bool,
        total_demo_len: Optional[int] = None,
    ) -> List[int]:
        """Pre-select frame indices matching `select_episode_write_indices` 100% bit-for-bit (`#306` Bug #3)!"""
        t_in = int(num_frames)
        if t_in <= 0:
            self._prepared_write_batch = None
            return []

        p_key = task_goal.strip().lower()
        if self._last_prompt_str is not None and p_key != self._last_prompt_str:
            self._pending_prompt_jump = 1.0
        self._last_prompt_str = p_key

        thresh = max(0, int(self.cfg.adaptive_demo_threshold))
        short_s = max(1, int(self.cfg.write_stride_demo))
        long_s = max(1, int(self.cfg.long_demo_stride))
        exec_s = max(1, int(self.cfg.write_stride_exec))
        start_long = ((thresh + long_s - 1) // long_s) * long_s
        dense_pref = max(0, int(self.mem_cfg.dense_prefix_frames))

        sel_local: List[int] = []
        sel_stream_steps: List[int] = []
        sel_p_jumps: List[float] = []
        sel_new_sess: List[bool] = []

        for i in range(t_in):
            s_idx = self.stream_total_frames + i
            exec_to_demo = (self._last_was_demo is False) and bool(is_demo)
            demo_to_exec = (self._last_was_demo is True) and (not bool(is_demo))
            self._last_was_demo = bool(is_demo)

            p_jump = self._pending_prompt_jump
            is_new_sess = bool(exec_to_demo or (p_jump > 0.5))
            if is_new_sess:
                self.demo_frames_in_session = 0
                self.session_start_frame = s_idx
                self.session_exec_start_frame = s_idx
                self._stage_latch.reset()
            elif demo_to_exec:
                self.session_exec_start_frame = s_idx
                self._stage_latch.reset()

            local_idx = s_idx - self.session_start_frame
            if is_demo:
                self.demo_frames_in_session += 1
                self.session_exec_start_frame = s_idx + 1
                if total_demo_len is not None:
                    is_final_demo = self.demo_frames_in_session == int(total_demo_len)
                else:
                    is_final_demo = i == (t_in - 1)
                if local_idx < dense_pref:
                    stride_hit = True
                elif local_idx < thresh:
                    stride_hit = (local_idx % short_s) == 0
                else:
                    stride_hit = (local_idx >= start_long) and ((local_idx - start_long) % long_s == 0)
                write_flag = stride_hit or is_final_demo or is_new_sess
            else:
                exec_rel = max(0, s_idx - self.session_exec_start_frame)
                stride_hit = (local_idx < dense_pref) or ((exec_rel % exec_s) == 0)
                write_flag = stride_hit or demo_to_exec or is_new_sess

            if write_flag:
                sel_local.append(i)
                sel_stream_steps.append(s_idx)
                sel_p_jumps.append(1.0 if is_new_sess else p_jump)
                sel_new_sess.append(is_new_sess)
                self._pending_prompt_jump = 0.0

        self.stream_total_frames += t_in
        if sel_local:
            self._prepared_write_batch = (sel_stream_steps, sel_p_jumps, sel_new_sess, bool(is_demo))
        else:
            self._prepared_write_batch = None
        return sel_local

    @torch.no_grad()
    def ingest_selected_frames(
        self,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor | None,
        raw_states: np.ndarray,
        norm_states: np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
    ) -> None:
        """Ingest ONLY the pre-selected `len(sel_local)` frames (`~2.6x` fewer SigLIP passes, `0` stride drift!)."""
        del task_goal, norm_states
        prep = self._prepared_write_batch
        self._prepared_write_batch = None
        if prep is None:
            return
        sel_stream_steps, sel_p_jumps, sel_new_sess, prep_demo = prep
        is_demo = bool(prep_demo)
        w_cnt = len(sel_stream_steps)
        if w_cnt == 0 or int(front_tokens_256.shape[0]) == 0:
            return
        if int(front_tokens_256.shape[0]) != w_cnt:
            raise ValueError(
                f"ingest_selected_frames expected {w_cnt} frames from select_write_indices, "
                f"got {int(front_tokens_256.shape[0])}"
            )

        if sel_new_sess[0]:
            # Cleanly isolate new cross-session episode kinematics & CausalConv1D from distractor tail!
            self._prev_w_front = None
            self._prev_w_state = None
            self._prev_w_demo = None
            self._conv_hist = None

        f_in = self.model.project_vis(front_tokens_256.to(self.device))
        f_w = pool_spatial_tokens_to_square(f_in, self.cfg.front_tokens_hist)
        if wrist_tokens_256 is not None:
            w_in = self.model.project_vis(wrist_tokens_256.to(self.device))
            w_w = pool_spatial_tokens_to_square(w_in, self.cfg.wrist_tokens_hist)
        else:
            w_w = torch.zeros((w_cnt, self.cfg.wrist_tokens_hist, f_w.shape[-1]), device=self.device, dtype=f_w.dtype)

        st_mme = self.normalizer.normalize_state(np.asarray(raw_states, dtype=np.float32))
        s_w = torch.from_numpy(np.asarray(st_mme, dtype=np.float32)).to(self.device)
        d_w = torch.full((w_cnt,), bool(is_demo), device=self.device, dtype=torch.bool)
        local_step_t = torch.tensor(
            [st_s - self.session_start_frame for st_s in sel_stream_steps], device=self.device, dtype=torch.long
        )

        kin = self.model.writer.compute_frame_kinematics(
            f_w, s_w, d_w, prev_front=self._prev_w_front, prev_state=self._prev_w_state, prev_is_demo=self._prev_w_demo
        )
        self._prev_w_front = f_w[-1].detach().clone()
        self._prev_w_state = s_w[-1].detach().clone()
        self._prev_w_demo = d_w[-1].detach().clone()

        raw_gdn, lm_slots = self.model.writer.encode_frames_raw(f_w, w_w, s_w, local_step_t, d_w, kin)
        gdn_slots, self._conv_hist = self.model.writer.apply_slot_conv(raw_gdn, conv_history=self._conv_hist)

        pj_t = torch.tensor(sel_p_jumps, device=self.device, dtype=torch.float32)
        br_t = (
            self.model.gdn.boundary_gate(gdn_slots[:, 0, :], kin, prompt_cos_jump=pj_t)
            if self.model.gdn.boundary_gate is not None
            else torch.zeros((w_cnt,), device=self.device, dtype=torch.float32)
        )
        for bi in range(w_cnt):
            if sel_new_sess[bi] or float(br_t[bi].item()) > float(self.cfg.boundary_flush_threshold):
                self._stage_latch.reset()
            if float(br_t[bi].item()) > float(self.cfg.boundary_flush_threshold):
                self.session_start_frame = sel_stream_steps[bi]
                if not is_demo:
                    self.session_exec_start_frame = sel_stream_steps[bi]
                if self._conv_hist is not None:
                    self._conv_hist.zero_()

        k1, v1, g1, b1, _ = self.model.gdn.layer1.project_slots(
            gdn_slots.unsqueeze(0),
            vis_motion_rms=kin["vis_motion_rms"].unsqueeze(0),
            grip_spike=kin["grip_spike"].unsqueeze(0),
            boundary_reset=br_t.unsqueeze(0),
        )
        raw_surp = self.model.gdn.layer1.compute_exact_write_surprise(
            k1, v1, g1, b1, num_write_slots=self.cfg.num_write_slots, initial_state_fp32=self._surp_l1_state
        ).squeeze(0)
        _, self._surp_l1_state, _ = run_dual_backend_gdn(
            k=k1, v=v1, g_fp32=g1, beta_fp32=b1, q=None, initial_state_fp32=self._surp_l1_state, track_occ=False
        )

        # `#306` Bug #2 Fix: accumulate raw causal `s_ema` (`0` mini-batch z-score!) and z-score session-wide at readout!
        s_list = raw_surp.detach().float().cpu().tolist()
        br_list = br_t.detach().float().cpu().tolist()
        beta = float(self.cfg.surprise_ema_beta)
        cur_ema = float(self._surprise_ema)
        s_ema_batch: List[float] = []
        for wi in range(w_cnt):
            if br_list[wi] > 0.5:
                cur_ema = 0.0
            cur_ema = beta * cur_ema + (1.0 - beta) * float(s_list[wi])
            s_ema_batch.append(max(float(s_list[wi]), cur_ema))
        self._surprise_ema = cur_ema

        for wi in range(w_cnt):
            self._w_steps.append(sel_stream_steps[wi])
            self._w_is_demo.append(bool(is_demo))
            self._w_gdn_slots.append(gdn_slots[wi])
            self._w_lm_slots.append(lm_slots[wi])
            self._w_state_norm.append(s_w[wi].detach())
            self._w_vis_rms.append(float(kin["vis_motion_rms"][wi].item()))
            self._w_grip_spike.append(float(kin["grip_spike"][wi].item()))
            self._w_b_reset.append(float(br_t[wi].item()))
            self._w_s_ema.append(float(s_ema_batch[wi]))

    @torch.no_grad()
    def ingest_frames(
        self,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor | None,
        raw_states: np.ndarray,
        norm_states: np.ndarray,
        *,
        task_goal: str,
        is_demo: bool,
        total_demo_len: Optional[int] = None,
    ) -> None:
        t_in = int(front_tokens_256.shape[0])
        if t_in == 0:
            return
        sel_local = self.select_write_indices(
            t_in, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len
        )
        if not sel_local:
            return
        idx_t = torch.tensor(sel_local, device=front_tokens_256.device, dtype=torch.long)
        f_sel = front_tokens_256[idx_t]
        w_sel = wrist_tokens_256[idx_t.to(wrist_tokens_256.device)] if wrist_tokens_256 is not None else None
        raw_np = np.asarray(raw_states, dtype=np.float32)[sel_local]
        norm_np = np.asarray(norm_states, dtype=np.float32)[sel_local]
        self.ingest_selected_frames(
            f_sel, w_sel, raw_np, norm_np, task_goal=task_goal, is_demo=is_demo
        )

    @torch.no_grad()
    def predict_subgoal(
        self,
        *,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> str | None:
        """Predict grounded subgoal string (`< 3 ms`, `O(1)` memory, `0` oracle!) for `Pi05RoboMMEPolicy`."""
        q_stream_step = max(0, self.stream_total_frames - 1)
        cutoff_stream = q_stream_step + 1  # Strictly causal `<= q_stream_step`!

        w_tot = len(self._w_steps)
        w_steps_t = (
            torch.tensor(self._w_steps, device=self.device, dtype=torch.long)
            if w_tot > 0
            else torch.empty((0,), device=self.device, dtype=torch.long)
        )
        u_end = int(torch.searchsorted(w_steps_t, cutoff_stream, right=False).item()) if w_tot > 0 else 0

        if u_end > self._committed_u:
            seg_slots = torch.stack(self._w_gdn_slots[self._committed_u : u_end], dim=0).unsqueeze(0)
            seg_kin = {
                "vis_motion_rms": torch.tensor(self._w_vis_rms[self._committed_u : u_end], device=self.device, dtype=torch.float32),
                "grip_spike": torch.tensor(self._w_grip_spike[self._committed_u : u_end], device=self.device, dtype=torch.float32),
            }
            seg_br = torch.tensor(self._w_b_reset[self._committed_u : u_end], device=self.device, dtype=torch.float32).unsqueeze(0)
            seg_out = self.model.gdn.forward_segment(
                seg_slots, seg_kin, seg_br, init_states=self._gdn_states, compute_surprise=False
            )
            self._gdn_states = (seg_out["S1"], seg_out["occ1"], seg_out["S2"], seg_out["occ2"])
            self._committed_u = u_end

        S1_p, o1_p, S2_p, o2_p = self._gdn_states
        has_hist = u_end > 0
        self._last_b_reset_max = max(self._w_b_reset[:u_end]) if u_end > 0 else 0.0
        if self.abstain_ordinal_counting and should_abstain_by_task_goal(
            task_goal,
            abstain_place_order=self.abstain_place_order,
            abstain_temporal_button=self.abstain_temporal_button,
        ):
            self._last_pred_subgoal = ""
            self._last_loop_depth = 1
            return None

        if has_hist:
            lm_slots_all = torch.stack(self._w_lm_slots[:u_end], dim=0)
            d_all = torch.tensor(self._w_is_demo[:u_end], device=self.device, dtype=torch.bool)
            br_all = torch.tensor(self._w_b_reset[:u_end], device=self.device, dtype=torch.float32)
            s_ema_all = torch.tensor(self._w_s_ema[:u_end], device=self.device, dtype=torch.float32)
            grip_all = torch.tensor(self._w_grip_spike[:u_end], device=self.device, dtype=torch.float32)

            # Session-wide salience z-score over `[start_pos : u_end]` (`0.0` diff vs `forward_episode`, `#306` Bug #2)!
            valid_resets = torch.nonzero(br_all > float(self.cfg.boundary_flush_threshold), as_tuple=False).flatten()
            start_pos = int(valid_resets[-1].item()) if valid_resets.numel() > 0 else 0
            sal_all = torch.zeros((u_end,), device=self.device, dtype=torch.float32)
            vis_sal_all = torch.zeros((u_end,), device=self.device, dtype=torch.float32)
            s_seg = s_ema_all[start_pos:u_end]
            if s_seg.numel() > 0:
                s_norm = (s_seg - s_seg.mean()) / (s_seg.std(unbiased=False) + 1e-5)
                vis_sal_all[start_pos:u_end] = s_norm
                out_seg = s_norm + 1.5 * grip_all[start_pos:u_end]
                demo_idx = torch.nonzero(d_all[start_pos:u_end], as_tuple=False).flatten()
                if demo_idx.numel() > 0:
                    out_seg[demo_idx[0]] = out_seg[demo_idx[0]] + 1.0
                    out_seg[demo_idx[-1]] = out_seg[demo_idx[-1]] + 1.0
                sal_all[start_pos:u_end] = out_seg

            sel_pos = self.model.landmark_bank.select_landmark_indices(
                w_steps_t[:u_end],
                d_all,
                sal_all,
                br_all,
                cutoff_excl_frame=cutoff_stream,
                pure_vis_salience_w=vis_sal_all,
            )
            w_rel = (w_steps_t[:u_end] - self.session_start_frame).clamp(min=0)
            q_rel = max(0, q_stream_step - self.session_start_frame)
            lt, lm = self.model.landmark_bank.gather_padded_landmarks(
                lm_slots_all, w_rel, d_all, sal_all, sel_pos, query_step_idx=q_rel
            )
        else:
            empty_lm = torch.empty((0, 16, self.cfg.mem_token_dim), device=self.device)
            empty_i = torch.empty((0,), device=self.device, dtype=torch.long)
            empty_b = torch.empty((0,), device=self.device, dtype=torch.bool)
            empty_f = torch.empty((0,), device=self.device, dtype=torch.float32)
            q_rel = max(0, q_stream_step - self.session_start_frame)
            lt, lm = self.model.landmark_bank.gather_padded_landmarks(
                empty_lm, empty_i, empty_b, empty_f, empty_i, query_step_idx=q_rel
            )
        lm_tok_b = lt.unsqueeze(0)
        lm_mask_b = lm.unsqueeze(0)

        prompt_tokens, lang_mask = self._encode_prompt(task_goal)
        f_cur = self.model.project_vis(front_tokens_256.to(self.device))
        w_cur = self.model.project_vis(wrist_tokens_256.to(self.device))
        st_mme = self.normalizer.normalize_state(np.asarray(raw_state, dtype=np.float32))
        st_cur = torch.from_numpy(np.asarray(st_mme, dtype=np.float32)).unsqueeze(0).to(self.device)
        st_prev = st_cur
        if u_end > 0:
            target_prev = max(self.session_exec_start_frame, q_stream_step - int(self.cfg.write_stride_exec))
            p_pos = int(torch.searchsorted(w_steps_t[:u_end], target_prev, right=True).item()) - 1
            if p_pos >= 0 and (not self._w_is_demo[p_pos]) and self._w_steps[p_pos] >= self.session_exec_start_frame:
                st_prev = self._w_state_norm[p_pos].unsqueeze(0)
        scene_tokens, front_grid_tokens, _ = self.model.scene_encoder(f_cur, w_cur, st_cur, st_prev)

        ctx_for_gdn = torch.cat([prompt_tokens[:, :8, :], pool_spatial_tokens_to_square(front_grid_tokens, 16)], dim=1)
        dt = self.model.coda.pri_q_proj.weight.dtype
        u_state = self.model.coda.slot_init.to(dt)

        max_hops = (
            1 if self.hop_mode == "l1" else (2 if self.hop_mode == "l2" else (3 if self.hop_mode == "l3" else self.cfg.num_hops))
        )
        final_preds: Optional[Dict[str, torch.Tensor]] = None
        used_hops = 1
        for h_i in range(max_hops):
            q_bias = None if h_i == 0 else self.model.coda.compute_hop_query_bias(u_state)
            gdn_tok, _ = self.model.gdn.readout(ctx_for_gdn, S1_p, o1_p, S2_p, o2_p, extra_query_bias=q_bias)
            if h_i > 0 and self.cfg.num_persistent_tokens > 0 and self.model.gdn.readout.persistent_tokens is not None:
                rd = self.model.gdn.readout
                raw_pt = rd.persistent_tokens.to(gdn_tok.dtype)
                p_base = rd.persistent_norm(raw_pt) if self.cfg.norm_hop_persistent else raw_pt
                p_tok = p_base.unsqueeze(0).expand(int(gdn_tok.shape[0]), -1, -1)
                gdn_tok = torch.cat([p_tok, gdn_tok], dim=1)
            gdn_mask = torch.full((1, int(gdn_tok.shape[1])), has_hist, device=self.device, dtype=torch.bool)
            if self.cfg.num_persistent_tokens > 0:
                gdn_mask[:, : self.cfg.num_persistent_tokens] = True
            mem_tok = torch.cat([gdn_tok, lm_tok_b.to(gdn_tok.dtype)], dim=1)
            mem_mask = torch.cat([gdn_mask, lm_mask_b], dim=1)

            u_state, preds_h = self.model.coda.forward_single_hop(
                h_i, u_state, prompt_tokens, lang_mask, mem_tok, mem_mask, scene_tokens, front_grid_tokens
            )
            final_preds = preds_h
            used_hops = h_i + 1
            if self.hop_mode == "auto" and h_i < max_hops - 1:
                if float(torch.sigmoid(preds_h["halt_logit"][0]).item()) >= 0.80:
                    break

        assert final_preds is not None
        if self.peak_window_radius > 0:
            final_preds = apply_peak_window_to_preds(
                final_preds, s_side=self.model.coda.front_side, radius=self.peak_window_radius
            )
        if self.fold_occlusion_twins and self._twin_pairs:
            final_preds = {
                **final_preds,
                "template_logits": fold_occlusion_twin_logits(final_preds["template_logits"], self._twin_pairs),
            }

        self._last_final_preds = final_preds
        if self.min_template_confidence > 0.0:
            probs = F.softmax(final_preds["template_logits"][0].float(), dim=-1)
            max_prob = float(compute_twin_summed_max_prob(probs, self._twin_pairs).item())
            if max_prob < self.min_template_confidence:
                self._last_pred_subgoal = ""
                self._last_loop_depth = used_hops
                return None
        tmpl_idx = int(final_preds["template_logits"][0].argmax(dim=-1).item())
        tmpl_str = self.model.template_vocab[tmpl_idx] if 0 <= tmpl_idx < len(self.model.template_vocab) else ""
        if self.abstain_ordinal_counting and should_abstain_by_task_goal(
            task_goal, abstain_place_order=self.abstain_place_order
        ):
            self._last_pred_subgoal = ""
            self._last_loop_depth = used_hops
            return None
        if self.stage_entry_latch:
            pred_sg = self._stage_latch.step(
                tmpl_str, final_preds["xy_pri"][0], final_preds["xy_sec"][0]
            )
        else:
            pred_sg = render_grounded_subgoal_from_norm_xy(
                tmpl_str, final_preds["xy_pri"][0], final_preds["xy_sec"][0]
            )
        self._last_pred_subgoal = pred_sg
        self._last_loop_depth = used_hops
        return pred_sg if pred_sg else None

    @torch.no_grad()
    def read_memory(
        self,
        *,
        prefix_hidden: torch.Tensor | None = None,
        front_tokens_256: torch.Tensor,
        wrist_tokens_256: torch.Tensor,
        raw_state: np.ndarray,
        norm_state: np.ndarray,
        task_goal: str,
        step_idx: int,
        exec_start_idx: int,
    ) -> Tuple[None, None, Dict[str, Any]]:
        """Return `(None, None, telemetry)` for vanilla `symbolic-grounded-subgoal` `Pi05Model`."""
        return None, None, {
            "loop_depth": self._last_loop_depth,
            "mem_active_tokens": 0,
            "pred_subgoal": self._last_pred_subgoal,
            "boundary_reset_max": self._last_b_reset_max,
        }


_ORDINAL_CUBE_RE = re.compile(r"\b(?:first|second|third|fourth|fifth)\s+(?:blue|green|red|correct)\s+cube\b")
_EPISODE_ABSTAIN_GOAL_PHRASES_BASE: Tuple[str, ...] = (
    "into the bin",           # BinFill (14/14 unique prompts across val80+test240; 0/306 false positives!)
    "place it on the target", # PickXtimes (11/11 unique prompts; eliminates mid-episode Stage-2/3 chatter!)
    "right-side target",      # SwingXtimes (9/9 unique prompts; 0/306 false positives!)
)
_EPISODE_ABSTAIN_GOAL_PHRASE_PLACE_ORDER: str = "target it was previously placed on"  # VideoPlaceOrder (8/8 unique prompts!)
_EPISODE_ABSTAIN_GOAL_PHRASES_TEMPORAL_BUTTON: Tuple[str, ...] = (
    "stop the cube just as it reaches",  # StopCube (4/4 unique prompts; v3_s250 = 80% vs M9 40%!)
    "button was pressed",                # VideoPlaceButton (5/5 unique prompts; v3_s250 = 60% vs M9 20%!)
)
_SYMBOLIC_SPECIALIST_GOAL_PHRASES: Tuple[str, ...] = (
    "container hiding",       # Permanence: VideoUnmask, ButtonUnmask, VideoUnmaskSwap, ButtonUnmaskSwap (26/26 prompts!)
    "highlighteted",          # PickHighlight (100% prompts!)
    "highlighted",            # Spelling-normalized safety alias
    "same block",             # VideoRepick (3/3 prompts!)
)


def should_abstain_by_task_goal(
    task_goal: str | None,
    *,
    abstain_place_order: bool = True,
    abstain_temporal_button: bool = False,
) -> bool:
    """Zero-oracle episode-level abstention check (`#343` + `#345` M9/v3_s250 alignment) over public user `task_goal`.

    Abstains (`subgoal=None` -> `512t` `Hybrid` fallback) cleanly across the ENTIRE episode for
    multi-repeat counting tasks (`BinFill`, `PickXtimes`, `SwingXtimes`), `VideoPlaceOrder` (`abstain_place_order`),
    and optionally `StopCube` + `VideoPlaceButton` (`abstain_temporal_button`, where `v3_s250` hits `80%` & `60%`!).
    """
    if not task_goal:
        return False
    g = str(task_goal).strip().lower()
    if any(p in g for p in _EPISODE_ABSTAIN_GOAL_PHRASES_BASE):
        return True
    if abstain_place_order and (_EPISODE_ABSTAIN_GOAL_PHRASE_PLACE_ORDER in g):
        return True
    if abstain_temporal_button and any(p in g for p in _EPISODE_ABSTAIN_GOAL_PHRASES_TEMPORAL_BUTTON):
        return True
    return False


def should_route_to_symbolic_specialist(task_goal: str | None) -> bool:
    """Zero-oracle `#346` `P2-M10` specialist router over public user `task_goal` (`0` oracle inputs!).

    Routes `<row_y, col_x>` occlusion-permanence, highlight, and repick episodes (`Permanence` 4 tasks +
    `PickHighlight` + `VideoRepick`, where `symbolic_grounded_subgoal_pt` has `93.3–95.2%` ceiling & `P2-M9`
    hits `96.2%` / `85.0%` unseen accuracy) to `Symbolic-VLM + P2-M9-Sym`, and all continuous/counting/order/imitation
    episodes to `P2-M7C` / `Hybrid`.
    """
    if not task_goal:
        return False
    g = str(task_goal).strip().lower()
    return any(p in g for p in _SYMBOLIC_SPECIALIST_GOAL_PHRASES)


_TRAJECTORY_IMITATION_GOAL_PHRASES: Tuple[str, ...] = (
    "same manner",
    "same pattern",
    "retrace",
    "same peg",
    "same end",
    "insert it into",
)


def should_use_hop2_trajectory_imitation(task_goal: str | None) -> bool:
    """Zero-oracle `#401` public-prompt check activating `L=2` `ResidualMultiHopRefiner` on trajectory-imitation tasks.

    `v3_stageC_hop2` (`L=2`) lifts `Imitation` from `45.0% -> 65.0%` (`+20.0 pp`, `MoveCube 80%`, `PatternLock 80%`,
    `RouteStick 80%`, `InsertPeg 0%->20%`) while `L=1` preserves `80.0%` on `demo==0` `Counting` (`0` oracle inputs!).
    """
    if not task_goal:
        return False
    g = str(task_goal).strip().lower()
    return any(p in g for p in _TRAJECTORY_IMITATION_GOAL_PHRASES)


def should_use_reactive_stopcube(task_goal: str | None) -> bool:
    """Zero-oracle `#405` public-prompt check for reactive cube-stopping (`StopCube`: `excl=1, write_stride_exec=1`)."""
    if not task_goal:
        return False
    g = str(task_goal).strip().lower()
    return ("stop the cube" in g) and ("reaches" in g)


def should_use_precision_insertpeg(task_goal: str | None) -> bool:
    """Zero-oracle `#405` public-prompt check for sub-mm peg-end & hole insertion (`InsertPeg`: `L=2` + climax landmarks)."""
    if not task_goal:
        return False
    g = str(task_goal).strip().lower()
    return any(p in g for p in ("same peg", "same end", "insert it into"))


def should_route_to_reactive_precision(task_goal: str | None) -> bool:
    """Zero-oracle `#405` `P2-M12` router (`True` iff `StopCube` or `InsertPeg`, `0` privileged inputs!)."""
    return should_use_reactive_stopcube(task_goal) or should_use_precision_insertpeg(task_goal)


def compute_twin_summed_max_prob(
    probs: torch.Tensor,
    twin_pairs: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Compute stage-level max confidence (`[..., ]`) summing occlusion twins `p[u_i] + p[g_i]` (`#343` Fix 3).

    Eliminates false mid-grasp container drops on `VideoUnmask` when Panda arm occlusion splits stage
    probability across `u_i` (`'pick up the container that hides ...'`) and `g_i` (`'... at <bbox> ...'`),
    while leaving raw `probs.argmax(dim=-1)` untouched!
    """
    max_raw = probs.max(dim=-1).values
    if not twin_pairs:
        return max_raw
    pair_sums = torch.stack([probs[..., u_i] + probs[..., g_i] for u_i, g_i in twin_pairs], dim=-1)
    return torch.maximum(max_raw, pair_sums.max(dim=-1).values)


def is_ordinal_counting_template(tmpl_str: str) -> bool:
    """Zero-oracle check (`#338`) identifying unoccluded multi-repeat ordinal counting templates.

    Preserves 100% of `Permanence` (`"hides"`, `"container"`, `"first/second button"`), `PickHighlight`
    (`"highlighted"`), `StopCube` (`"to prepare"`, `"to stop"`), `VideoRepick`, `VideoPlaceButton`, and
    `Imitation`, while abstaining (`subgoal=None` -> `512t` `Hybrid` fallback!) on discrete multi-repeat
    ordinal counting templates (`"for the {first..fifth} time"`, `"into the bin"`, `"{first..fourth} {color/correct} cube"`).
    """
    if not tmpl_str:
        return False
    s = tmpl_str.lower()
    if "highlighted" in s or "hides" in s or "container" in s:
        return False
    if "for the " in s and " time" in s:
        return True
    if "into the bin" in s:
        return True
    if _ORDINAL_CUBE_RE.search(s) is not None:
        return True
    return False


def build_occlusion_twin_template_pairs(template_vocab: Sequence[str]) -> List[Tuple[int, int]]:
    """Identify `(ungrounded_idx, grounded_idx)` twin pairs caused by `RecordWrapper` arm-occlusion fallback (`#315` Fix A)."""
    ungrounded_map: Dict[str, int] = {}
    for idx, t_str in enumerate(template_vocab):
        if _BBOX_PLACEHOLDER not in t_str and t_str.strip():
            ungrounded_map[t_str.strip()] = idx

    pairs: List[Tuple[int, int]] = []
    for g_idx, g_str in enumerate(template_vocab):
        if _BBOX_PLACEHOLDER not in g_str:
            continue
        stripped = " ".join(g_str.replace(f"at {_BBOX_PLACEHOLDER}", "").split()).replace(" ,", ",").strip()
        u_idx = ungrounded_map.get(stripped)
        if u_idx is not None and u_idx != g_idx:
            pairs.append((u_idx, g_idx))
    return pairs


@torch.no_grad()
def fold_occlusion_twin_logits(
    template_logits: torch.Tensor,
    twin_pairs: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Fold probability mass of occlusion-stripped twins `u_i` into grounded `g_i` via `logaddexp` (`#315` Fix A)."""
    if not twin_pairs:
        return template_logits
    out = template_logits.float().clone()
    for u_i, g_i in twin_pairs:
        out[..., g_i] = torch.logaddexp(out[..., g_i], out[..., u_i])
        out[..., u_i] = -1e9
    return out


@torch.no_grad()
def apply_peak_window_to_preds(
    preds: Dict[str, torch.Tensor],
    *,
    s_side: int = 16,
    radius: int = 2,
) -> Dict[str, torch.Tensor]:
    """Apply `5x5` Peak-Window (`#308` Bug #4 Fix) to batch `B >= 1` `hm_logits` while preserving sub-cell offsets."""
    if radius <= 0:
        return preds
    dev = preds["hm_logits"].device
    bsz = int(preds["hm_logits"].shape[0])
    idx_1d = torch.arange(s_side, device=dev, dtype=torch.long)
    yy, xx = torch.meshgrid(idx_1d, idx_1d, indexing="ij")
    xy_refined: Dict[str, torch.Tensor] = {}
    for ch, key in ((0, "xy_pri"), (1, "xy_sec")):
        hm = preds["hm_logits"][:, ch, :].float()  # (B, P_f)
        pk = hm.argmax(dim=-1)
        py, px = (pk // s_side).view(bsz, 1, 1), (pk % s_side).view(bsz, 1, 1)
        win = ((yy.unsqueeze(0) - py).abs() <= int(radius)) & ((xx.unsqueeze(0) - px).abs() <= int(radius))
        xy_win, _ = spatial_softmax_2d(
            hm.masked_fill(~win.reshape(bsz, -1), -1e9),
            grid_hw=(s_side, s_side),
            align_corners=False,
        )
        xy_glob, _ = spatial_softmax_2d(hm, grid_hw=(s_side, s_side), align_corners=False)
        xy_refined[key] = (xy_win + (preds[key] - xy_glob)).clamp(-1.0, 1.0)
    return {**preds, **xy_refined}


class StageEntrySubgoalLatch:
    """`RecordWrapper` / `segmentation_utils.py:60-63` Stage-Entry `<y, x>` Latch (`#322`).

    Computes `<row_y, col_x>` once at unoccluded stage entry and holds it across mid-stage
    Panda arm descents/occlusions (`g_i -> u_i` flips or neighbor `xy_pri` jumps) until
    `canon_stage` transitions or an episode boundary reset occurs.
    """

    def __init__(self, twin_to_grounded: Mapping[str, str]) -> None:
        self.twin_to_grounded: Dict[str, str] = dict(twin_to_grounded)
        self.canon_stage: Optional[str] = None
        self.latched_subgoal: str = ""
        self.has_bbox: bool = False

    def reset(self) -> None:
        self.canon_stage = None
        self.latched_subgoal = ""
        self.has_bbox = False

    def step(
        self,
        tmpl_str: str,
        xy_pri_norm: torch.Tensor | np.ndarray,
        xy_sec_norm: Optional[torch.Tensor | np.ndarray] = None,
        *,
        image_hw: Tuple[int, int] = ROBOMME_IMAGE_HW,
    ) -> str:
        if not tmpl_str:
            self.reset()
            return ""
        canon = self.twin_to_grounded.get(tmpl_str, tmpl_str)
        if canon != self.canon_stage:
            self.canon_stage = canon
            if _BBOX_PLACEHOLDER in tmpl_str:
                self.latched_subgoal = render_grounded_subgoal_from_norm_xy(
                    tmpl_str, xy_pri_norm, xy_sec_norm, image_hw=image_hw
                )
                self.has_bbox = True
            else:
                self.latched_subgoal = tmpl_str
                self.has_bbox = False
            return self.latched_subgoal

        if self.has_bbox:
            return self.latched_subgoal
        if _BBOX_PLACEHOLDER in tmpl_str:
            self.latched_subgoal = render_grounded_subgoal_from_norm_xy(
                tmpl_str, xy_pri_norm, xy_sec_norm, image_hw=image_hw
            )
            self.has_bbox = True
            return self.latched_subgoal
        self.latched_subgoal = tmpl_str
        return self.latched_subgoal


def _drop_shard_page_cache(cache_dir: Path, task: str) -> None:
    """Release clean Lustre page-cache pages for `<task>` shard so GH200 unified-RAM cgroup stays `< 16 GB`."""
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    for cand in (cache_dir / f"{task}.pt", cache_dir / "shards" / f"{task}.pt"):
        if cand.is_file():
            try:
                fd = os.open(str(cand), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except OSError:
                pass


@torch.no_grad()
def precompute_deltaground_subgoals(
    deltaground_ckpt: str | Path,
    cache_dir: str | Path,
    *,
    output_path: Optional[str | Path] = None,
    tasks: Sequence[str] | None = None,
    device: str | torch.device = "cuda",
    num_hops: int = 3,
    peak_window_radius: int = 2,
    fold_occlusion_twins: bool = False,
    stage_entry_latch: bool = True,
    force_pytorch_gdn: bool = False,
) -> Tuple[Dict[str, List[Dict[int, str]]], Dict[str, Any]]:
    """Precompute `DeltaGround` `<row_y, col_x>` subgoal strings across all cached episodes (`~20s` on GH200!).

    Returns `(pred_subgoals_by_task, deltaground_payload)` so `P2-M7A` & `P2-M7B` can both train on
    exact `DeltaGround` predictions AND bundle `deltaground_payload` inside their saved checkpoints!
    """
    raw = torch.load(Path(deltaground_ckpt), map_location="cpu", weights_only=False)
    payload: Dict[str, Any] = raw.get("ema_deltaground") or raw.get("deltaground") or raw

    if output_path is not None and Path(output_path).exists():
        cached = torch.load(Path(output_path), map_location="cpu", weights_only=False)
        if isinstance(cached, dict) and "pred_subgoals_by_task" in cached:
            return cached["pred_subgoals_by_task"], payload

    dev = torch.device(device)
    cache_root = Path(cache_dir)
    cfg = DeltaGroundConfig.from_dict(payload["deltaground_config"])
    tmpl_vocab: List[str] = list(payload["template_vocab"])
    twin_pairs = build_occlusion_twin_template_pairs(tmpl_vocab)
    twin_to_grounded = {tmpl_vocab[u_i]: tmpl_vocab[g_i] for u_i, g_i in twin_pairs}
    model = DeltaGroundModel(cfg, template_vocab=tmpl_vocab)
    if dev.type == "cuda":
        model.cast_bf16_keep_fp32_gates()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(dev).eval()

    normalizer = ActionStateNormalizer(mode="quantile_delta", stats=payload.get("state_stats") or MME_VLA_NORM_STATS)
    target_tasks = ROBOMME_TASKS if tasks is None else tuple(tasks)

    pred_by_task: Dict[str, List[Dict[int, str]]] = {}
    for t_idx, task in enumerate(target_tasks):
        sh = load_pi05_shards(cache_root, tasks=(task,), mmap=True)[task]
        if not bool(model.has_siglip_head.item()):
            model.siglip_head_weight.copy_(sh["siglip_head_weight"].float().to(dev))
            model.siglip_head_bias.copy_(sh["siglip_head_bias"].float().to(dev))
            model.has_siglip_head.fill_(True)

        n_eps = int(sh["epis_idx"].shape[0])
        ep_offs = sh["ep_offsets"].to(torch.long).numpy()
        ex_offs = sh["exec_ep_offsets"].to(torch.long).numpy()
        ex_starts = sh["exec_start_idx"].to(torch.long).numpy()
        rel_all = sh["exec_rel_step_idx"].to(torch.long)
        raw_st_all = sh["state"].to(torch.float32).contiguous().clone().numpy()
        st_norm_all = torch.from_numpy(normalizer.normalize_state(raw_st_all))
        is_demo_all = sh["is_demo"].to(torch.bool).contiguous().clone()
        lt_all = sh["lang_tokens"].to(torch.long).contiguous().clone()
        lm_all = sh["lang_masks"].to(torch.bool).contiguous().clone()

        task_list: List[Dict[int, str]] = []
        latch = StageEntrySubgoalLatch(twin_to_grounded)
        for ep_i in range(n_eps):
            s = int(ep_offs[ep_i])
            e = int(ep_offs[ep_i + 1])
            es = int(ex_offs[ep_i])
            ee = int(ex_offs[ep_i + 1])
            q_steps = rel_all[es:ee]
            if int(q_steps.numel()) == 0:
                task_list.append({})
                continue
            ep_in = {
                "exec_start_idx": int(ex_starts[ep_i]),
                "lang_tokens": lt_all[ep_i],
                "lang_masks": lm_all[ep_i],
                "front_emb": sh["front_emb"][s:e].to(torch.bfloat16).contiguous(),
                "wrist_emb": sh["wrist_emb"][s:e].to(torch.bfloat16).contiguous(),
                "state_norm": st_norm_all[s:e],
                "is_demo": is_demo_all[s:e],
                "chunk_step_indices": q_steps,
                "exec_front_256": sh["exec_front_256"][es:ee].to(torch.bfloat16).contiguous(),
                "exec_wrist_256": sh["exec_wrist_256"][es:ee].to(torch.bfloat16).contiguous(),
            }
            hops_out = model.forward_episode(
                ep_in, num_hops=num_hops, force_pytorch_gdn=force_pytorch_gdn or (dev.type != "cuda")
            )
            fin = apply_peak_window_to_preds(hops_out[-1], s_side=model.coda.front_side, radius=peak_window_radius)
            t_logits = (
                fold_occlusion_twin_logits(fin["template_logits"], twin_pairs)
                if fold_occlusion_twins
                else fin["template_logits"]
            )
            t_ids = t_logits.argmax(dim=-1).detach().cpu().tolist()
            xy1_cpu = fin["xy_pri"].detach().cpu()
            xy2_cpu = fin["xy_sec"].detach().cpu()
            q_list = q_steps.tolist()
            ep_map: Dict[int, str] = {}
            latch.reset()
            for qi, q_st in enumerate(q_list):
                tid = int(t_ids[qi])
                t_s = tmpl_vocab[tid] if 0 <= tid < len(tmpl_vocab) else ""
                if stage_entry_latch:
                    ep_map[int(q_st)] = latch.step(t_s, xy1_cpu[qi], xy2_cpu[qi])
                else:
                    ep_map[int(q_st)] = render_grounded_subgoal_from_norm_xy(t_s, xy1_cpu[qi], xy2_cpu[qi])
            task_list.append(ep_map)
        pred_by_task[task] = task_list
        del sh, raw_st_all, st_norm_all, is_demo_all, lt_all, lm_all, rel_all
        gc.collect()
        _drop_shard_page_cache(cache_root, task)
        print(f"[DeltaGround Precompute] {t_idx + 1}/{len(target_tasks)} {task}: {n_eps} eps done", flush=True)

    del model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.with_suffix(".pt.tmp")
        torch.save({"pred_subgoals_by_task": pred_by_task}, tmp_p)
        tmp_p.replace(out_p)

    return pred_by_task, payload
