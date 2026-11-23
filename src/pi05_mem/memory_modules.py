"""Phase-2 Memory Modules for `DeltaMem-π0.5` (`src/pi05_mem/memory_modules.py`).

Implements all core memory & auxiliary supervision mechanisms:
  1. `DualViewEventPerceiverWriter`:
     Compresses dual-camera (`front` + `wrist`) spatial tokens + 2D `SpatialSoftmax` keypoints +
     8-DoF proprio + velocity + explicit gripper-edge (`Δg_t`, `|Δg_t|`, contact edge) into:
       - `N_w` object/event GDN write slots per written frame (with per-slot `CausalConv1D` along time),
       - `M_lm` high-resolution spatial landmark slots per written frame (for exact geometry recall).
  2. `MultiTimescale2LayerGDN`:
     Directly realizes `IDEA.md`'s 3-timescale (`Short / Mid / Long`) `fp32` Gated DeltaNet with
     Cross-Scale Consolidation Gate + Phase-1's winning 2-layer read-before-write composition (`M5`),
     fused `(d_v + 1)` key-occupancy tracking, Invariant A1 (`g` strictly on slot `s=0`), Invariant A3
     demo-preserving causal cutoff, **exact parallel shifted-query GDN write-surprise**, and
     **Instruction / Teleport Boundary Forget Gate** (protecting `demo -> exec` transitions while
     flushing cross-session `k=3` distractor clutter automatically!).
  3. `SurpriseEventLandmarkBank`:
     Non-privileged online selector picking up to `K` keyframes strictly prior to `τ_cutoff(t)` using
     GDN write-surprise + gripper/velocity event spikes + conditioning demo start/end anchors.
  4. `T2MemFastWeightMemory` (`P2-B1`):
     Clean PyTorch `T²Mem`-style (`arXiv:2609.36720` / LaCT-adapted) 1-layer online self-supervised
     fast-weight TTT baseline with mini-batch residual reconstruction gradient updates, LayerNorm VJP,
     grad-norm clipping, and observation-grounded query readout.
  5. `PerceptualFrameSampMemory` (`P2-B2`):
     Non-privileged strided keyframe memory baseline matching `FrameSamp+Modul`.
  6. `GroundedAuxiliarySuite` + Zero-Oracle Self-Predicted `<x, y>` Subgoal Token Injector:
     Dense training-time supervision on `<x, y>` coordinates + 2D spatial heatmap KL, pass/pick/swing
     completed & remaining counts, stage/color classification, and subgoal latent distillation, plus
     optional zero-init self-predicted `<x, y>` pointer token injection at test time (`0` oracle!).
  7. `ResidualMultiHopRefiner` + `StageCCRNGate` (`P2-M3`):
     Zero-initialized residual Hop-2 (`M^(1) + tanh(gamma) * ΔM^(2)`, `gamma_0 = 0`) + `M_CRN = 8`
     relative-quantile halting gate guaranteeing Step-0 `L=2 == L=1` parity.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from analysis.subgoal_grounding import STAGE_VOCAB, norm_xy_to_pixel_xy, spatial_softmax_2d
from models.deltaloop_vla import (
    MultiHeadAttn,
    SwiGLU,
    build_1d_sincos_embed,
    build_2d_sincos_pos_embed,
    l2_normalize_fp32,
    run_dual_backend_gdn,
    safe_masked_sdpa,
)

from .config import Pi05MemConfig


# =============================================================================
# 1. Helper Utilities: Adaptive Grid Pooling & Causal Cutoff (Invariant A3)
# =============================================================================


def pool_spatial_tokens_to_square(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Pool `(..., P_in, D)` square spatial patch grid cleanly to `(..., target_tokens, D)`."""
    p_in = int(tokens.shape[-2])
    if p_in == target_tokens:
        return tokens
    s_in = int(round(p_in**0.5))
    s_out = int(round(target_tokens**0.5))
    if s_in * s_in != p_in or s_out * s_out != target_tokens:
        raise ValueError(f"Expected square token counts, got {p_in} -> {target_tokens}")
    lead = tokens.shape[:-2]
    d = int(tokens.shape[-1])
    orig_dtype = tokens.dtype
    feat = tokens.float().reshape(-1, s_in, s_in, d).permute(0, 3, 1, 2).contiguous()
    pooled = F.adaptive_avg_pool2d(feat, output_size=(s_out, s_out))
    return pooled.permute(0, 2, 3, 1).reshape(*lead, target_tokens, d).to(orig_dtype)


def compute_a3_cutoff_frame(
    step_idx: int,
    exec_start_idx: int,
    exclusion_window: int = 16,
) -> int:
    """Invariant A3: Preserve 100% of `is_demo` prefix while blocking short-lag self-retrieval."""
    return max(int(exec_start_idx), max(0, int(step_idx) - int(exclusion_window) + 1))


def select_episode_write_indices(
    total_frames: int,
    exec_start_idx: int,
    cfg: Pi05MemConfig,
    device: Optional[torch.device] = None,
    *,
    write_stride_exec_override: Optional[int] = None,
) -> torch.Tensor:
    """Deterministic causal write frame indices matching online `adapter._should_write_frame` 100% bit-for-bit!"""
    demo_end = min(max(0, int(exec_start_idx)), int(total_frames))
    thresh = max(0, int(cfg.adaptive_demo_threshold))
    short_s = max(1, int(cfg.write_stride_demo))
    long_s = max(1, int(cfg.long_demo_stride))
    raw_exec_s = cfg.write_stride_exec if write_stride_exec_override is None else write_stride_exec_override
    exec_stride = max(1, int(raw_exec_s))
    dense_pref = max(0, int(cfg.dense_prefix_frames))

    idx_set: set[int] = set()
    if dense_pref > 0:
        idx_set.update(range(0, min(int(total_frames), dense_pref)))

    if demo_end > 0:
        first_end = min(demo_end, thresh)
        idx_set.update(range(0, first_end, short_s))
        if demo_end > thresh:
            # Exact match to `local_idx >= thresh` -> `local_idx % long_s == 0`
            start_long = ((thresh + long_s - 1) // long_s) * long_s
            idx_set.update(range(start_long, demo_end, long_s))
        idx_set.add(demo_end - 1)  # Always anchor final conditioning demo frame!

    if total_frames > demo_end:
        idx_set.update(range(demo_end, int(total_frames), exec_stride))

    idx_list = sorted(idx_set)
    return torch.tensor(idx_list, dtype=torch.long, device=device)


# =============================================================================
# 2. Dual-View + Proprio + Gripper-Edge Perceiver Writer & Observation Encoder
# =============================================================================


class SpatialKeypointExtractor(nn.Module):
    """2D `SpatialSoftmax` continuous sub-patch keypoint bottleneck (`align_corners=False`)."""

    def __init__(self, d_in: int, kp_front: int, kp_wrist: int, d_out: int) -> None:
        super().__init__()
        self.kp_front = int(kp_front)
        self.kp_wrist = int(kp_wrist)
        self.front_head = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, self.kp_front))
        self.wrist_head = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, self.kp_wrist))
        self.mlp = nn.Sequential(
            nn.Linear(2 * (self.kp_front + self.kp_wrist), d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, front_tok: torch.Tensor, wrist_tok: torch.Tensor) -> torch.Tensor:
        b = int(front_tok.shape[0])
        dt = self.front_head[1].weight.dtype  # type: ignore[index]
        sf = int(round(front_tok.shape[1] ** 0.5))
        sw = int(round(wrist_tok.shape[1] ** 0.5))
        fl = self.front_head(front_tok.to(dt)).transpose(1, 2)
        wl = self.wrist_head(wrist_tok.to(dt)).transpose(1, 2)
        with torch.autocast(device_type=front_tok.device.type, enabled=False):
            xy_f, _ = spatial_softmax_2d(fl.float(), grid_hw=(sf, sf), align_corners=False)
            xy_w, _ = spatial_softmax_2d(wl.float(), grid_hw=(sw, sw), align_corners=False)
        cat = torch.cat([xy_f.reshape(b, -1), xy_w.reshape(b, -1)], dim=-1).to(dt)
        return self.mlp(cat).unsqueeze(1)  # (B, 1, d_out)


class DualViewEventPerceiverWriter(nn.Module):
    """Encodes written frames `(front + wrist + 8D state + Δstate + gripper-edge)` into:
      1. `gdn_slots`: `(W, N_w, d_mem)` object/event slots (after per-slot `CausalConv1D`),
      2. `lm_slots`:  `(W, M_lm, mem_token_dim)` high-resolution spatial landmark slots (`1024-D`),
      3. `event_features`: dict with `vis_motion_rms (W,)`, `grip_spike (W,)`, `joint_jump (W,)`.
    """

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_mem)
        self.d_mem = d
        self.num_write_slots = int(cfg.num_write_slots)
        self.num_lm_slots = int(cfg.tokens_per_landmark)
        self.front_hist_n = int(cfg.front_tokens_hist)
        self.wrist_hist_n = int(cfg.wrist_tokens_hist)
        self.conv_width = int(cfg.slot_conv_kernel)

        self.front_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        self.wrist_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        # 8 (state_norm) + 8 (delta_state) + 4 (gripper_val, delta_g, |delta_g|, contact_edge_flag) = 20
        self.proprio_proj = nn.Sequential(
            nn.Linear(2 * cfg.state_dim + 4, d),
            nn.SiLU(),
            nn.Linear(d, d),
        )

        self.kp_extractor: Optional[SpatialKeypointExtractor] = None
        self.kp_emb: Optional[nn.Parameter] = None
        if cfg.use_spatial_softmax:
            self.kp_extractor = SpatialKeypointExtractor(
                cfg.vis_dim, cfg.spatial_kp_front, cfg.spatial_kp_wrist, d
            )
            self.kp_emb = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        sf = int(round(self.front_hist_n**0.5))
        sw = int(round(self.wrist_hist_n**0.5))
        self.register_buffer("front_pos", build_2d_sincos_pos_embed(sf, sf, d), persistent=False)
        self.register_buffer("wrist_pos", build_2d_sincos_pos_embed(sw, sw, d), persistent=False)

        self.view_emb = nn.Embedding(3, d)  # 0=front, 1=wrist, 2=proprio+gripper_edge
        self.demo_emb = nn.Embedding(2, d)  # 0=exec, 1=is_demo
        self.time_proj = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.view_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.demo_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.time_proj.weight, mean=0.0, std=0.02)

        # GDN write-slot Perceiver queries (`N_w`) + Landmark spatial queries (`M_lm`)
        total_q = self.num_write_slots + self.num_lm_slots
        self.slot_queries = nn.Parameter(torch.randn(total_q, d) * 0.02)
        self.q_norm = nn.RMSNorm(d)
        self.kv_norm = nn.RMSNorm(d)
        self.cross_attn = MultiHeadAttn(d, cfg.mem_enc_heads)
        self.ffn_norm = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d * 2)

        if self.conv_width > 1:
            self.temporal_conv = nn.Conv1d(d, d, kernel_size=self.conv_width, groups=d, bias=True)
            nn.init.zeros_(self.temporal_conv.weight)
            self.temporal_conv.weight.data[:, 0, -1] = 1.0  # Identity on current frame at init
            nn.init.zeros_(self.temporal_conv.bias)
        else:
            self.temporal_conv = None

        self.gdn_slot_norm = nn.RMSNorm(d)
        self.use_multiscale_pyramid = cfg.landmark_encoder == "multiscale_pyramid"
        if self.use_multiscale_pyramid:
            self.num_lm_slots = 16
            # Dual-Stream (`arXiv:2610.13262` Eq. 4) Zero-Init Multi-Scale Pyramid:
            # 16 tokens (`4x4`) fusing W_4(4x4) + zero-init W_8(2x2 sub-block of 8x8) + zero-init W_2(2x2) + wrist/proprio
            self.pyr_norm = nn.LayerNorm(cfg.vis_dim)
            self.pyr_w4 = nn.Linear(cfg.vis_dim, cfg.mem_token_dim)
            self.pyr_w8 = nn.Linear(4 * cfg.vis_dim, cfg.mem_token_dim, bias=False)
            self.pyr_w2 = nn.Linear(cfg.vis_dim, cfg.mem_token_dim, bias=False)
            self.pyr_wp = nn.Linear(d + cfg.vis_dim, cfg.mem_token_dim, bias=False)
            nn.init.zeros_(self.pyr_w8.weight)
            nn.init.zeros_(self.pyr_w2.weight)
            self.register_buffer("pyr_pos4", build_2d_sincos_pos_embed(4, 4, cfg.mem_token_dim), persistent=False)
            self.pyr_out_norm = nn.RMSNorm(cfg.mem_token_dim)
            self.lm_out_proj = nn.Identity()
        else:
            self.lm_out_proj = nn.Sequential(
                nn.RMSNorm(d),
                nn.Linear(d, cfg.mem_token_dim),
            )

    @staticmethod
    def compute_frame_kinematics(
        front_emb: torch.Tensor,         # (W, P_f, D_vis)
        state_norm: torch.Tensor,        # (W, 8)
        is_demo: torch.Tensor,           # (W,) bool
        prev_front: Optional[torch.Tensor] = None,
        prev_state: Optional[torch.Tensor] = None,
        prev_is_demo: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute strictly causal visual motion RMS, Δstate, gripper-edge spikes, & teleport jumps."""
        w = int(front_emb.shape[0])
        dev = front_emb.device
        if w == 0:
            z1 = torch.zeros((0,), device=dev, dtype=torch.float32)
            return {
                "delta_state": torch.zeros((0, state_norm.shape[-1]), device=dev, dtype=torch.float32),
                "grip_feat": torch.zeros((0, 4), device=dev, dtype=torch.float32),
                "vis_motion_rms": z1,
                "grip_spike": z1,
                "arm_jump_norm": z1,
                "vis_cos_cut": z1,
                "exec_to_demo": torch.zeros((0,), device=dev, dtype=torch.bool),
                "demo_to_exec": torch.zeros((0,), device=dev, dtype=torch.bool),
            }

        f_f32 = front_emb.float()
        s_f32 = state_norm.float()
        d_bool = is_demo.bool()

        if prev_front is not None and prev_state is not None and prev_is_demo is not None:
            f_prev = torch.cat([prev_front.float().unsqueeze(0), f_f32[:-1]], dim=0)
            s_prev = torch.cat([prev_state.float().unsqueeze(0), s_f32[:-1]], dim=0)
            d_prev = torch.cat([prev_is_demo.bool().view(1), d_bool[:-1]], dim=0)
            has_prev = torch.ones((w,), device=dev, dtype=torch.bool)
        else:
            f_prev = torch.cat([f_f32[:1], f_f32[:-1]], dim=0)
            s_prev = torch.cat([s_f32[:1], s_f32[:-1]], dim=0)
            d_prev = torch.cat([d_bool[:1], d_bool[:-1]], dim=0)
            has_prev = torch.ones((w,), device=dev, dtype=torch.bool)
            has_prev[0] = False

        demo_to_exec = has_prev & d_prev & (~d_bool)
        exec_to_demo = has_prev & (~d_prev) & d_bool

        # Zero finite-differences across `demo -> exec` boundary so scene reset never Pollutes Δstate
        same_phase = has_prev & (~demo_to_exec) & (~exec_to_demo)
        same_f = same_phase.float().unsqueeze(-1)

        delta_s = (s_f32 - s_prev) * same_f
        df = (f_f32 - f_prev) * same_f.unsqueeze(-1)
        vis_rms = torch.sqrt(torch.mean(df * df, dim=(-2, -1)) + 1e-8) * same_phase.float()

        g_curr = s_f32[:, 7:8]
        dg = delta_s[:, 7:8]
        abs_dg = dg.abs()
        edge_flag = (abs_dg > 0.20).float()
        grip_feat = torch.cat([g_curr, dg, abs_dg, edge_flag], dim=-1)  # (W, 4)

        arm_vel = torch.linalg.vector_norm(delta_s[:, :7], ord=2, dim=-1)
        grip_spike = abs_dg.squeeze(-1) + 0.25 * arm_vel

        # Raw un-masked single-step joint jump & SigLIP scene-cut cosine during execution
        raw_arm_jump = torch.linalg.vector_norm(s_f32[:, :7] - s_prev[:, :7], ord=2, dim=-1)
        exec_cont = (has_prev & (~d_prev) & (~d_bool)).float()
        arm_jump_exec = raw_arm_jump * exec_cont

        f_bar = F.normalize(f_f32.mean(dim=1), dim=-1)
        fp_bar = F.normalize(f_prev.mean(dim=1), dim=-1)
        cos_sim = (f_bar * fp_bar).sum(dim=-1)
        vis_cos_cut = F.relu(0.92 - cos_sim) * exec_cont

        return {
            "delta_state": delta_s,
            "grip_feat": grip_feat,
            "vis_motion_rms": vis_rms,
            "grip_spike": grip_spike,
            "arm_jump_norm": arm_jump_exec,
            "vis_cos_cut": vis_cos_cut,
            "exec_to_demo": exec_to_demo,
            "demo_to_exec": demo_to_exec,
        }

    def encode_frames_raw(
        self,
        front_emb: torch.Tensor,         # (W, P_f, D_vis)
        wrist_emb: torch.Tensor,         # (W, P_w, D_vis)
        state_norm: torch.Tensor,        # (W, 8)
        step_idx: torch.Tensor,          # (W,)
        is_demo: torch.Tensor,           # (W,) bool
        kin: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode `W` frames into `(raw_gdn_slots [W, N_w, d], lm_slots [W, M_lm, mem_token_dim])`."""
        w = int(front_emb.shape[0])
        dt = self.front_proj[1].weight.dtype  # type: ignore[index]
        if w == 0:
            return (
                torch.empty((0, self.num_write_slots, self.d_mem), device=front_emb.device, dtype=dt),
                torch.empty((0, self.num_lm_slots, self.cfg.mem_token_dim), device=front_emb.device, dtype=dt),
            )

        f_in = pool_spatial_tokens_to_square(front_emb, self.front_hist_n)
        w_in = pool_spatial_tokens_to_square(wrist_emb, self.wrist_hist_n)
        ve = self.view_emb.weight.to(dt)

        f_tok = self.front_proj(f_in.to(dt)) + self.front_pos.to(dt).unsqueeze(0) + ve[0].view(1, 1, -1)
        w_tok = self.wrist_proj(w_in.to(dt)) + self.wrist_pos.to(dt).unsqueeze(0) + ve[1].view(1, 1, -1)

        prop_in = torch.cat(
            [state_norm.float(), (kin["delta_state"] * 4.0).clamp(-4.0, 4.0), kin["grip_feat"]],
            dim=-1,
        ).to(dt)
        p_tok = self.proprio_proj(prop_in).unsqueeze(1) + ve[2].view(1, 1, -1)

        kv_parts = [f_tok, w_tok]
        if self.kp_extractor is not None and self.kp_emb is not None:
            kv_parts.append(self.kp_extractor(f_in, w_in) + self.kp_emb.to(dt))
        kv_parts.append(p_tok)
        kv = self.kv_norm(torch.cat(kv_parts, dim=1))

        t_pe = self.time_proj(build_1d_sincos_embed(step_idx.float() / 4.0, self.d_mem).to(dt))
        d_pe = self.demo_emb(is_demo.long().clamp(0, 1)).to(dt)
        bias = (t_pe + d_pe).unsqueeze(1)

        q_all = self.q_norm(self.slot_queries.to(dt).unsqueeze(0).expand(w, -1, -1) + bias)
        h_all = q_all + self.cross_attn(q_all, kv)
        h_all = h_all + self.ffn(self.ffn_norm(h_all))

        raw_gdn = h_all[:, : self.num_write_slots, :]
        if self.use_multiscale_pyramid:
            # Dual-Stream (`arXiv:2610.13262` Eq. 4) Multi-Scale 2x2 + 4x4 + 8x8 Pyramid -> 16 tokens/frame!
            f_64 = pool_spatial_tokens_to_square(front_emb, 64)
            z8 = self.pyr_norm(f_64.to(dt)).view(w, 8, 8, -1)
            # Group each 2x2 sub-block of 8x8 into its corresponding 4x4 cell `[W, 16, 4 * D_vis]`
            z8_sub = z8.view(w, 4, 2, 4, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(w, 16, -1)
            z4 = z8.view(w, 4, 2, 4, 2, -1).mean(dim=(2, 4)).reshape(w, 16, -1)
            z2_grid = z4.view(w, 2, 2, 2, 2, -1).mean(dim=(2, 4))  # (W, 2, 2, D_vis)
            # Broadcast each 2x2 quadrant back to its 2x2 children in the 4x4 grid `[W, 16, D_vis]`
            z2_up = z2_grid[:, :, None, :, None, :].expand(w, 2, 2, 2, 2, -1).reshape(w, 16, -1)
            wp_ctx = torch.cat([w_in.to(dt).mean(dim=1), p_tok.squeeze(1)], dim=-1)  # (W, D_vis + d_mem)
            pyr = (
                self.pyr_w4(z4)
                + self.pyr_w8(z8_sub)
                + self.pyr_w2(z2_up)
                + self.pyr_wp(wp_ctx).unsqueeze(1)
                + self.pyr_pos4.to(dt).unsqueeze(0)
            )
            lm_slots = self.pyr_out_norm(pyr)
        else:
            lm_slots = self.lm_out_proj(h_all[:, self.num_write_slots :, :])
        return raw_gdn, lm_slots

    def apply_slot_conv(
        self,
        raw_gdn_slots: torch.Tensor,                 # (W, N_w, d_mem)
        conv_history: Optional[torch.Tensor] = None,  # Optional (K-1, N_w, d_mem) for online streaming!
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Strictly causal per-slot 1D conv along frame axis with exact online streaming state."""
        w = int(raw_gdn_slots.shape[0])
        dt = self.slot_queries.dtype
        if w == 0:
            return raw_gdn_slots.to(dt), conv_history

        x = raw_gdn_slots.to(dt)
        if self.temporal_conv is None or self.conv_width <= 1:
            return self.gdn_slot_norm(x), None

        pad_len = self.conv_width - 1
        if conv_history is None:
            hist = torch.zeros((pad_len, self.num_write_slots, self.d_mem), device=x.device, dtype=dt)
        else:
            hist = conv_history.to(device=x.device, dtype=dt)
            if hist.shape[0] < pad_len:
                z_pad = torch.zeros(
                    (pad_len - hist.shape[0], self.num_write_slots, self.d_mem),
                    device=x.device,
                    dtype=dt,
                )
                hist = torch.cat([z_pad, hist], dim=0)
            elif hist.shape[0] > pad_len:
                hist = hist[-pad_len:]

        x_full = torch.cat([hist, x], dim=0)  # (pad_len + W, N_w, D)
        next_hist = x_full[-pad_len:].detach().clone()

        # Reshape `(N_w, D, pad_len + W)` -> depthwise conv with 0 extra padding!
        x_in = x_full.permute(1, 2, 0).contiguous()  # (N_w, D, pad_len + W)
        y = F.silu(self.temporal_conv(x_in))         # (N_w, D, W)
        out = self.gdn_slot_norm(y.permute(2, 0, 1).contiguous())  # (W, N_w, D)
        return out, next_hist


class CurrentObservationContextEncoder(nn.Module):
    """Encodes current execution step `(front, wrist, state, prompt_pool)` into query context `E_t`."""

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        d = int(cfg.d_mem)
        self.d_mem = d
        self.front_n = int(cfg.front_tokens_hist)
        self.wrist_n = int(cfg.wrist_tokens_hist)

        self.front_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        self.wrist_proj = nn.Sequential(nn.LayerNorm(cfg.vis_dim), nn.Linear(cfg.vis_dim, d))
        self.state_proj = nn.Sequential(nn.Linear(cfg.state_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.prompt_proj = nn.Sequential(nn.LayerNorm(cfg.vlm_width), nn.Linear(cfg.vlm_width, d))

        sf = int(round(self.front_n**0.5))
        sw = int(round(self.wrist_n**0.5))
        self.register_buffer("front_pos", build_2d_sincos_pos_embed(sf, sf, d), persistent=False)
        self.register_buffer("wrist_pos", build_2d_sincos_pos_embed(sw, sw, d), persistent=False)
        self.mod_emb = nn.Embedding(4, d)
        nn.init.normal_(self.mod_emb.weight, mean=0.0, std=0.02)

        self.norm1 = nn.RMSNorm(d)
        self.attn = MultiHeadAttn(d, cfg.mem_enc_heads)
        self.norm2 = nn.RMSNorm(d)
        self.ffn = SwiGLU(d, d * 2)
        self.out_norm = nn.RMSNorm(d)

    def forward(
        self,
        front_emb: torch.Tensor,                 # (B, P_f, D_vis)
        wrist_emb: torch.Tensor,                 # (B, P_w, D_vis)
        state_norm: torch.Tensor,                # (B, 8)
        prompt_pool: Optional[torch.Tensor] = None,  # Optional (B, vlm_width)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(ctx_tokens [B, L_ctx, d_mem], front_spatial_tokens [B, P_f_hist, d_mem])`."""
        b = int(front_emb.shape[0])
        dt = self.front_proj[1].weight.dtype  # type: ignore[index]
        me = self.mod_emb.weight.to(dt)

        f_sq = pool_spatial_tokens_to_square(front_emb, self.front_n)
        w_sq = pool_spatial_tokens_to_square(wrist_emb, self.wrist_n)

        f_tok = self.front_proj(f_sq.to(dt)) + self.front_pos.to(dt).unsqueeze(0) + me[0].view(1, 1, -1)
        w_tok = self.wrist_proj(w_sq.to(dt)) + self.wrist_pos.to(dt).unsqueeze(0) + me[1].view(1, 1, -1)
        s_tok = self.state_proj(state_norm.to(dt)).unsqueeze(1) + me[2].view(1, 1, -1)

        parts = [f_tok, w_tok, s_tok]
        if prompt_pool is not None:
            p_tok = self.prompt_proj(prompt_pool.to(dt)).unsqueeze(1) + me[3].view(1, 1, -1)
            parts.append(p_tok)
        x = torch.cat(parts, dim=1)
        h = self.norm1(x)
        x = x + self.attn(h, h)
        x = self.out_norm(x + self.ffn(self.norm2(x)))
        return x, x[:, : self.front_n, :]


# =============================================================================
# 3. Boundary Forget Gate + Multi-Timescale (`Short/Mid/Long`) 2-Layer `fp32` GDN
# =============================================================================


class EpisodeBoundaryForgetGate(nn.Module):
    """Detects cross-session boundaries without manual `env.reset()` while protecting `demo -> exec`!

    Triggers (`r_m in [0, 1]`):
      1. Task instruction embedding change `1 - cos(p_m, p_{m-1}) > 1e-3` -> immediate reset,
      2. Execution-to-Demo transition (`~is_demo_{m-1} & is_demo_m`) -> immediate reset,
      3. Learned + physical execution teleport detector (`arm_jump_norm > 1.2` in normalized joint
         space during execution, while NEVER firing on `demo_to_exec`!).
    """

    def __init__(self, d_mem: int) -> None:
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(d_mem + 3, d_mem // 2),
            nn.SiLU(),
            nn.Linear(d_mem // 2, 1),
        )
        nn.init.zeros_(self.detector[-1].weight)  # type: ignore[union-attr]
        nn.init.constant_(self.detector[-1].bias, -4.0)  # Default `r_m ~ 0.018` inside episodes

    def forward(
        self,
        slot0_seq: torch.Tensor,                   # (W, d_mem)
        kin: Dict[str, torch.Tensor],
        prompt_cos_jump: Optional[torch.Tensor] = None,  # Optional (W,) in [0, 2]
    ) -> torch.Tensor:
        """Return boundary reset gate `r_m: (W,)` in `[0, 1]` in `float32`."""
        w = int(slot0_seq.shape[0])
        dev = slot0_seq.device
        if w == 0:
            return torch.zeros((0,), device=dev, dtype=torch.float32)

        dt = self.detector[0].weight.dtype  # type: ignore[index]
        p_jump = (
            prompt_cos_jump.float().clamp(0.0, 2.0)
            if prompt_cos_jump is not None
            else torch.zeros((w,), device=dev, dtype=torch.float32)
        )
        arm_jump = kin["arm_jump_norm"].float()
        vis_rms = kin["vis_motion_rms"].float()

        feat = torch.cat(
            [
                slot0_seq.to(dt),
                p_jump.unsqueeze(-1).to(dt),
                arm_jump.unsqueeze(-1).to(dt),
                vis_rms.unsqueeze(-1).to(dt),
            ],
            dim=-1,
        )
        learned_r = torch.sigmoid(self.detector(feat).float().squeeze(-1))

        # Hard structural cross-session cues (zero oracle: derived purely from prompt, proprio, SigLIP cos, & demo)
        hard_prompt = (p_jump > 0.02).float()
        hard_exec_to_demo = kin["exec_to_demo"].float()
        hard_teleport = (arm_jump > 1.35).float() * torch.sigmoid((arm_jump - 1.65) * 8.0)
        vis_cut = kin.get("vis_cos_cut", torch.zeros_like(arm_jump)).float()
        # Require true reset-level scene cut (`cos < 0.70`, i.e. `vis_cut > 0.22`) OR scene cut (`vis_cut > 0.14`)
        # paired with arm home-reset teleport (`arm_jump > 1.05`) so intra-episode sweeps NEVER false-flush (`r <= 0.50`)!
        coupled_cut = ((vis_cut > 0.08) & (arm_jump > 1.05)).float() * torch.sigmoid((vis_cut - 0.14) * 25.0)
        strong_cut = (vis_cut > 0.15).float() * torch.sigmoid((vis_cut - 0.22) * 25.0)
        hard_scene_cut = torch.maximum(coupled_cut, strong_cut)
        r = torch.maximum(
            learned_r,
            torch.maximum(hard_prompt, torch.maximum(hard_exec_to_demo, torch.maximum(hard_teleport, hard_scene_cut))),
        )

        # CRITICAL INVARIANT: NEVER flush conditioning demo memory on `demo -> exec` transition!
        not_demo_to_exec = (~kin["demo_to_exec"]).float()
        return (r * not_demo_to_exec).clamp(0.0, 1.0)


class MultiTimescaleSlotGDNLayer(nn.Module):
    """Single Multi-Timescale (`Short / Mid / Long`) `fp32` GDN layer enforcing Invariant A1."""

    def __init__(self, cfg: Pi05MemConfig, use_seq_readout: bool = False) -> None:
        super().__init__()
        self.d_mem = int(cfg.d_mem)
        self.num_heads = int(cfg.gdn_heads)
        self.d_k = int(cfg.gdn_d_k)
        self.d_v = int(cfg.gdn_d_v)
        self.beta_max = float(cfg.beta_max)
        self.timescale_mode = cfg.timescale_mode
        self.use_seq_readout = bool(use_seq_readout)
        self.reset_strength = float(cfg.boundary_reset_strength)
        self.yaad_huber_delta = float(cfg.yaad_huber_delta)

        self.k_proj = nn.Linear(self.d_mem, self.num_heads * self.d_k, bias=False)
        self.v_proj = nn.Linear(self.d_mem, self.num_heads * self.d_v, bias=False)
        self.beta_proj = nn.Linear(self.d_mem, self.num_heads, bias=True)
        nn.init.zeros_(self.beta_proj.weight)

        if cfg.use_motion_beta_gate:
            nn.init.constant_(self.beta_proj.bias, float(cfg.motion_beta_bias))
            self.motion_proj: Optional[nn.Linear] = nn.Linear(2, self.num_heads, bias=False)
            nn.init.constant_(self.motion_proj.weight, 2.5)
        else:
            nn.init.constant_(self.beta_proj.bias, -1.2)
            self.motion_proj = None

        # Configure 3-timescale (`Short / Mid / Long`) or ablation decay parameterizations in fp32
        if self.timescale_mode in ("three_band_consolidate", "three_band_no_consolidate"):
            if self.num_heads % 3 != 0:
                raise ValueError(f"gdn_heads ({self.num_heads}) must be divisible by 3 for 3-band mode")
            hb = self.num_heads // 3
            g_mins: List[float] = []
            g_maxs: List[float] = []
            for hl_lo, hl_hi in (cfg.short_halflife_bounds, cfg.mid_halflife_bounds, cfg.long_halflife_bounds):
                # Larger half-life -> alpha closer to 1 -> g = -ln(2)/hl closer to 0 (g_max)
                g_lo = -math.log(2.0) / max(float(hl_lo), 1.0)
                g_hi = -math.log(2.0) / max(float(hl_hi), 1.0)
                g_mins.extend([g_lo] * hb)
                g_maxs.extend([g_hi] * hb)
            self.register_buffer("g_band_min", torch.tensor(g_mins, dtype=torch.float32))
            self.register_buffer("g_band_max", torch.tensor(g_maxs, dtype=torch.float32))
            self.a_proj = nn.Linear(self.d_mem, self.num_heads, bias=True)
            nn.init.zeros_(self.a_proj.weight)
            nn.init.zeros_(self.a_proj.bias)  # Midpoint of each timescale band in log-decay space
        elif self.timescale_mode == "three_band_fixed":
            if self.num_heads % 3 != 0:
                raise ValueError(f"gdn_heads ({self.num_heads}) must be divisible by 3")
            hb = self.num_heads // 3
            hls = torch.cat(
                [
                    torch.logspace(
                        math.log10(cfg.short_halflife_bounds[0]),
                        math.log10(cfg.short_halflife_bounds[1]),
                        hb,
                    ),
                    torch.logspace(
                        math.log10(cfg.mid_halflife_bounds[0]),
                        math.log10(cfg.mid_halflife_bounds[1]),
                        hb,
                    ),
                    torch.logspace(
                        math.log10(cfg.long_halflife_bounds[0]),
                        math.log10(cfg.long_halflife_bounds[1]),
                        hb,
                    ),
                ]
            ).float()
            self.register_buffer("g_fixed", (-math.log(2.0) / hls).clamp(min=-15.0, max=-1e-7))
        elif self.timescale_mode == "uniform_same_decay":
            g_uni = torch.full((self.num_heads,), -math.log(2.0) / 96.0, dtype=torch.float32)
            self.register_buffer("g_fixed", g_uni)
        elif self.timescale_mode == "log_uniform_learnable":
            hl = torch.logspace(
                math.log10(cfg.short_halflife_bounds[0]),
                math.log10(cfg.long_halflife_bounds[1]),
                self.num_heads,
            ).float()
            target_neg_g = math.log(2.0) / hl
            dt_bias = torch.log(torch.expm1(target_neg_g).clamp(min=1e-6))
            self.a_proj = nn.Linear(self.d_mem, self.num_heads, bias=False)
            self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
            self.dt_bias = nn.Parameter(dt_bias)
        else:
            raise ValueError(f"Unsupported timescale_mode: {self.timescale_mode}")

        if self.use_seq_readout:
            self.q_proj = nn.Linear(self.d_mem, self.num_heads * self.d_k, bias=False)
            self.g_out_proj = nn.Linear(self.d_mem, self.num_heads * self.d_v, bias=True)
            # Per-timescale output projections (`Short`, `Mid`, `Long`) for Consolidation Gate!
            self.o_proj = nn.Linear(self.num_heads * self.d_v, self.d_mem, bias=False)

    def project_slots(
        self,
        frame_slots: torch.Tensor,                      # (..., W, N_w, d_mem)
        vis_motion_rms: Optional[torch.Tensor] = None,  # (..., W)
        grip_spike: Optional[torch.Tensor] = None,      # (..., W)
        boundary_reset: Optional[torch.Tensor] = None,  # (..., W) in [0, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Project `(..., W, N_w, d_mem)` -> `(k, v, g_flat, beta_flat, q)` with strict Invariant A1."""
        *lead, w, nw, _ = frame_slots.shape
        dt = self.k_proj.weight.dtype
        z = frame_slots.to(dt)

        k = l2_normalize_fp32(self.k_proj(z).view(*lead, w * nw, self.num_heads, self.d_k), dim=-1)
        v_raw = self.v_proj(z).view(*lead, w * nw, self.num_heads, self.d_v).float()
        if self.yaad_huber_delta > 0.0:
            # Miras `Yaad` (`arXiv:2504.13173`) smooth Huber value-outlier clip (preserves linear chunk kernel!)
            vn = torch.linalg.vector_norm(v_raw, ord=2, dim=-1, keepdim=True)
            v_clip = torch.clamp(self.yaad_huber_delta / (vn + 1e-6), max=1.0)
            v = v_raw * v_clip
        else:
            v = v_raw

        z_s0 = z[..., 0, :]  # (..., W, d_mem)
        if self.timescale_mode in ("three_band_consolidate", "three_band_no_consolidate"):
            gate = torch.sigmoid(self.a_proj(z_s0).float())
            g_s0 = self.g_band_min + gate * (self.g_band_max - self.g_band_min)
        elif self.timescale_mode in ("three_band_fixed", "uniform_same_decay"):
            g_s0 = self.g_fixed.expand(*lead, w, self.num_heads).clone()
        else:
            raw_a = self.a_proj(z_s0).float() + self.dt_bias.float()
            g_s0 = (-torch.exp(self.A_log.float()) * F.softplus(raw_a)).clamp(min=-15.0, max=-1e-7)

        if boundary_reset is not None:
            g_s0 = (g_s0 - self.reset_strength * boundary_reset.float().unsqueeze(-1)).clamp(min=-18.0)

        # Invariant A1: decay `g` active ONLY on slot `s=0`; slots `s=1..N_w-1` have exact `0.0` decay!
        g_slots = torch.zeros((*lead, w, nw, self.num_heads), device=z.device, dtype=torch.float32)
        g_slots[..., 0, :] = g_s0
        g_flat = g_slots.reshape(*lead, w * nw, self.num_heads)

        raw_beta = self.beta_proj(z).float()
        if self.motion_proj is not None:
            vm = (
                vis_motion_rms.float()
                if vis_motion_rms is not None
                else torch.zeros((*lead, w), device=z.device, dtype=torch.float32)
            )
            gs = (
                grip_spike.float()
                if grip_spike is not None
                else torch.zeros((*lead, w), device=z.device, dtype=torch.float32)
            )
            mg_in = torch.stack([torch.log1p(vm), torch.log1p(gs)], dim=-1).to(dt)  # (..., W, 2)
            raw_beta = raw_beta + self.motion_proj(mg_in).float().unsqueeze(-2)

        beta_flat = (self.beta_max * torch.sigmoid(raw_beta)).reshape(*lead, w * nw, self.num_heads)
        q = None
        if self.use_seq_readout:
            q = l2_normalize_fp32(self.q_proj(z).view(*lead, w * nw, self.num_heads, self.d_k), dim=-1)
        return k, v, g_flat, beta_flat, q

    @staticmethod
    @torch.no_grad()
    def compute_exact_write_surprise(
        k: torch.Tensor,                              # (B, T_slots, H, K) fp32
        v: torch.Tensor,                              # (B, T_slots, H, V) fp32
        g: torch.Tensor,                              # (B, T_slots, H) fp32
        beta: torch.Tensor,                           # (B, T_slots, H) fp32
        num_write_slots: int,
        initial_state_fp32: Optional[torch.Tensor] = None,
        force_pytorch: bool = False,
    ) -> torch.Tensor:
        """Compute bit-exact causal GDN write-surprise `||v_u - exp(g_u) S_{u-1} k_u||_2` in parallel!

        Uses the exact shifted-query identity: passing `q_tilde[:, :-1] = k[:, 1:]` into `S_curr @ q`
        returns `o_tilde[:, u-1] == S_{u-1} k_u` for all `u >= 1`, and `S_init @ k_0` at `u = 0`!
        Zero sequential Python loops, zero `(1 - beta)` division singularities, `< 1e-6` vs online step!
        """
        b, t_slots, h, k_dim = k.shape
        v_dim = v.shape[-1]
        nw = int(num_write_slots)
        w_frames = t_slots // nw
        if t_slots == 0 or w_frames == 0:
            return torch.zeros((b, 0), device=k.device, dtype=torch.float32)

        q_shift = torch.zeros_like(k)
        if t_slots > 1:
            q_shift[:, :-1] = k[:, 1:]

        o_shift, _, _ = run_dual_backend_gdn(
            k=k,
            v=v,
            g_fp32=g,
            beta_fp32=beta,
            q=q_shift,
            initial_state_fp32=initial_state_fp32,
            track_occ=False,
            force_pytorch=force_pytorch,
        )
        assert o_shift is not None
        sk_prev = torch.zeros((b, t_slots, h, v_dim), device=k.device, dtype=torch.float32)
        if initial_state_fp32 is not None:
            sk_prev[:, 0] = torch.einsum("bhvk,bhk->bhv", initial_state_fp32.float(), k[:, 0].float())
        if t_slots > 1:
            sk_prev[:, 1:] = o_shift[:, :-1].float()

        alpha = torch.exp(g.float()).unsqueeze(-1)
        delta_err = v.float() - alpha * sk_prev  # (B, T_slots, H, V)
        v_norm = torch.linalg.vector_norm(v.float(), ord=2, dim=-1) + 0.1
        rel_err = torch.linalg.vector_norm(delta_err, ord=2, dim=-1) / v_norm
        raw_surp = rel_err.mean(dim=-1).view(b, w_frames, nw).mean(dim=-1)  # (B, W)
        return raw_surp


class CrossScaleConsolidationGate(nn.Module):
    """`IDEA.md` Model 5: Zero-init gated Cross-Scale Consolidation (`Short -> Mid -> Long`) between L1 & L2."""

    def __init__(self, d_mem: int, num_heads: int, d_v: int) -> None:
        super().__init__()
        if num_heads % 3 != 0:
            self.num_bands = 1
            band_in = num_heads * d_v
        else:
            self.num_bands = 3
            band_in = (num_heads // 3) * d_v
        self.num_heads = num_heads
        self.d_v = d_v

        self.band_projs = nn.ModuleList([nn.Linear(band_in, d_mem, bias=False) for _ in range(self.num_bands)])
        self.consolidate_mlp = nn.Sequential(
            nn.RMSNorm(self.num_bands * d_mem),
            nn.Linear(self.num_bands * d_mem, d_mem * 2),
            nn.SiLU(),
            nn.Linear(d_mem * 2, d_mem),
        )
        # Zero-init consolidation output + learnable gate so L2 initially receives clean L1 residual
        nn.init.zeros_(self.consolidate_mlp[-1].weight)  # type: ignore[union-attr]
        nn.init.zeros_(self.consolidate_mlp[-1].bias)    # type: ignore[union-attr]
        self.gate_proj = nn.Linear(d_mem, d_mem, bias=True)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, z_slots_flat: torch.Tensor, raw_o_l1: torch.Tensor) -> torch.Tensor:
        """Fuse `(Short, Mid, Long)` Layer-1 GDN retrievals into consolidated input for Layer 2."""
        dt = z_slots_flat.dtype
        lead = raw_o_l1.shape[:-2]
        if self.num_bands == 3:
            hb = self.num_heads // 3
            s_out = self.band_projs[0](raw_o_l1[..., :hb, :].to(dt).reshape(*lead, hb * self.d_v))
            m_out = self.band_projs[1](raw_o_l1[..., hb : 2 * hb, :].to(dt).reshape(*lead, hb * self.d_v))
            l_out = self.band_projs[2](raw_o_l1[..., 2 * hb :, :].to(dt).reshape(*lead, hb * self.d_v))
            cat = torch.cat([s_out, m_out, l_out], dim=-1)
        else:
            cat = self.band_projs[0](raw_o_l1.to(dt).reshape(*lead, self.num_heads * self.d_v))

        cons = self.consolidate_mlp(cat)
        gate = torch.sigmoid(self.gate_proj(z_slots_flat))
        return cons * (2.0 * gate)  # Equals `cons` at init (2*sigmoid(0)=1) while `cons` itself starts at 0!


class ScalePartitionedGDNReadout(nn.Module):
    """Observation-grounded multi-timescale readout producing `N_gdn` `1024-D` memory tokens."""

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.d_mem)
        self.d_mem = d
        self.num_tokens = int(cfg.num_gdn_readout_tokens)
        self.num_persistent = int(cfg.num_persistent_tokens)
        self.num_heads = int(cfg.gdn_heads)
        self.d_k = int(cfg.gdn_d_k)
        self.d_v = int(cfg.gdn_d_v)
        self.eps_read = float(cfg.readout_eps)

        # Titans (`arXiv:2501.00663`) persistent task-memory tokens prepended to GDN readout tokens
        if self.num_persistent > 0:
            self.persistent_tokens = nn.Parameter(torch.randn(self.num_persistent, cfg.mem_token_dim) * 0.02)
            self.persistent_norm = nn.RMSNorm(cfg.mem_token_dim)
        else:
            self.persistent_tokens = None

        self.query_seeds = nn.Parameter(torch.randn(self.num_tokens, d) * 0.02)
        self.q_norm = nn.RMSNorm(d)
        self.ctx_attn = MultiHeadAttn(d, cfg.mem_enc_heads)

        # Query both Layer-1 (timescale-separated) and Layer-2 (read-before-write composed) states!
        self.q1_proj = nn.Linear(d, self.num_heads * self.d_k, bias=False)
        self.q2_proj = nn.Linear(d, self.num_heads * self.d_k, bias=False)
        self.g1_proj = nn.Linear(d, self.num_heads * self.d_v, bias=True)
        self.g2_proj = nn.Linear(d, self.num_heads * self.d_v, bias=True)

        # Scale-specific mask routing tokens `0..T/3-1 -> Short`, `T/3..2T/3-1 -> Mid`, `2T/3..T-1 -> Long`
        # plus residual full-head access so tokens specialize by timescale without losing global context
        self.scale_role_emb = nn.Parameter(torch.randn(self.num_tokens, d) * 0.02)
        self.fuse_proj = nn.Sequential(
            nn.Linear(2 * self.num_heads * self.d_v + d, d * 2),
            nn.SiLU(),
            nn.Linear(d * 2, cfg.mem_token_dim),
            nn.RMSNorm(cfg.mem_token_dim),
        )

    def _damped_read(
        self,
        q_in: torch.Tensor,
        q_proj: nn.Linear,
        g_proj: nn.Linear,
        S_past: torch.Tensor,
        occ_past: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, nt, _ = q_in.shape
        dt = q_proj.weight.dtype
        q = l2_normalize_fp32(q_proj(q_in.to(dt)).view(b, nt, self.num_heads, self.d_k), dim=-1)
        with torch.autocast(device_type=q_in.device.type, enabled=False):
            raw = torch.einsum("bhvk,bnhk->bnhv", S_past.float(), q.float())
            rms = torch.sqrt(torch.mean(raw * raw, dim=-1, keepdim=True) + self.eps_read * self.eps_read)
            r_damped = (raw / rms).to(dt)
            cov = (
                torch.sum(q.float() * occ_past.float().unsqueeze(1), dim=-1).mean(dim=(1, 2))
                if occ_past is not None
                else torch.zeros((b,), device=q_in.device, dtype=torch.float32)
            )
        gated = F.silu(g_proj(q_in.to(dt))).view(b, nt, self.num_heads, self.d_v) * r_damped
        return gated.reshape(b, nt, self.num_heads * self.d_v), cov

    def prepare_queries(
        self,
        ctx_tokens: torch.Tensor,
        extra_query_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(q_tok, Q1_fp32, Q2_fp32)` of shapes `(B, N_q, d_mem)` and `(B, N_q, H, K)`."""
        b = int(ctx_tokens.shape[0])
        dt = self.q1_proj.weight.dtype
        seeds = self.query_seeds.to(dt).unsqueeze(0).expand(b, -1, -1) + self.scale_role_emb.to(dt).unsqueeze(0)
        if extra_query_bias is not None:
            seeds = seeds + extra_query_bias.to(dt)
        q_tok = self.q_norm(seeds)
        q_tok = q_tok + self.ctx_attn(q_tok, ctx_tokens.to(dt))
        q1 = l2_normalize_fp32(self.q1_proj(q_tok.to(dt)).view(b, self.num_tokens, self.num_heads, self.d_k), dim=-1)
        q2 = l2_normalize_fp32(self.q2_proj(q_tok.to(dt)).view(b, self.num_tokens, self.num_heads, self.d_k), dim=-1)
        return q_tok, q1, q2

    def finalize_from_raw_reads(
        self,
        q_tok: torch.Tensor,              # (B, N_q, d_mem)
        raw1_fp32: torch.Tensor,          # (B, N_q, H, V) fp32
        occ1_dot_fp32: torch.Tensor,      # (B, N_q, H) fp32
        raw2_fp32: torch.Tensor,          # (B, N_q, H, V) fp32
        occ2_dot_fp32: torch.Tensor,      # (B, N_q, H) fp32
        include_persistent: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Apply Invariant C3 damped RMSNorm, output gating, fusion MLP, and persistent tokens."""
        b = int(q_tok.shape[0])
        dt = self.q1_proj.weight.dtype
        with torch.autocast(device_type=q_tok.device.type, enabled=False):
            rms1 = torch.sqrt(torch.mean(raw1_fp32 * raw1_fp32, dim=-1, keepdim=True) + self.eps_read * self.eps_read)
            rms2 = torch.sqrt(torch.mean(raw2_fp32 * raw2_fp32, dim=-1, keepdim=True) + self.eps_read * self.eps_read)
            r1_damped = (raw1_fp32 / rms1).to(dt)
            r2_damped = (raw2_fp32 / rms2).to(dt)
            cov1 = occ1_dot_fp32.mean(dim=(1, 2))
            cov2 = occ2_dot_fp32.mean(dim=(1, 2))

        r1 = (F.silu(self.g1_proj(q_tok.to(dt))).view(b, self.num_tokens, self.num_heads, self.d_v) * r1_damped).reshape(
            b, self.num_tokens, self.num_heads * self.d_v
        )
        r2 = (F.silu(self.g2_proj(q_tok.to(dt))).view(b, self.num_tokens, self.num_heads, self.d_v) * r2_damped).reshape(
            b, self.num_tokens, self.num_heads * self.d_v
        )
        mem_tokens = self.fuse_proj(torch.cat([r1, r2, q_tok], dim=-1))
        if self.persistent_tokens is not None and include_persistent:
            p_tok = self.persistent_norm(self.persistent_tokens.to(mem_tokens.dtype)).unsqueeze(0).expand(b, -1, -1)
            mem_tokens = torch.cat([p_tok, mem_tokens], dim=1)
        return mem_tokens, {"cov1": cov1, "cov2": cov2}

    def forward(
        self,
        ctx_tokens: torch.Tensor,         # (B, L_ctx, d_mem)
        S1_past: torch.Tensor,            # (B, H, V, K) fp32
        occ1_past: Optional[torch.Tensor],
        S2_past: torch.Tensor,            # (B, H, V, K) fp32
        occ2_past: Optional[torch.Tensor],
        extra_query_bias: Optional[torch.Tensor] = None,  # Optional (B, N_gdn, d_mem) for Stage-C Hop-2!
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b = int(ctx_tokens.shape[0])
        dt = self.q1_proj.weight.dtype
        seeds = self.query_seeds.to(dt).unsqueeze(0).expand(b, -1, -1) + self.scale_role_emb.to(dt).unsqueeze(0)
        if extra_query_bias is not None:
            seeds = seeds + extra_query_bias.to(dt)
        q_tok = self.q_norm(seeds)
        q_tok = q_tok + self.ctx_attn(q_tok, ctx_tokens.to(dt))

        r1, cov1 = self._damped_read(q_tok, self.q1_proj, self.g1_proj, S1_past, occ1_past)
        r2, cov2 = self._damped_read(q_tok, self.q2_proj, self.g2_proj, S2_past, occ2_past)
        mem_tokens = self.fuse_proj(torch.cat([r1, r2, q_tok], dim=-1))  # (B, N_gdn, 1024)
        if self.persistent_tokens is not None and extra_query_bias is None:
            p_tok = self.persistent_norm(self.persistent_tokens.to(mem_tokens.dtype)).unsqueeze(0).expand(b, -1, -1)
            mem_tokens = torch.cat([p_tok, mem_tokens], dim=1)
        return mem_tokens, {"cov1": cov1, "cov2": cov2}


class MultiTimescale2LayerGDN(nn.Module):
    """Complete 2-Layer `fp32` Multi-Timescale GDN Writer + Consolidation Gate + Boundary Reset."""

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_write_slots = int(cfg.num_write_slots)
        self.layer1 = MultiTimescaleSlotGDNLayer(cfg, use_seq_readout=True)
        self.use_consolidation = cfg.timescale_mode == "three_band_consolidate"
        self.consolidator: Optional[CrossScaleConsolidationGate] = (
            CrossScaleConsolidationGate(cfg.d_mem, cfg.gdn_heads, cfg.gdn_d_v)
            if self.use_consolidation
            else None
        )
        self.l2_norm = nn.RMSNorm(cfg.d_mem)
        self.layer2 = MultiTimescaleSlotGDNLayer(cfg, use_seq_readout=False)
        self.boundary_gate: Optional[EpisodeBoundaryForgetGate] = (
            EpisodeBoundaryForgetGate(cfg.d_mem) if cfg.use_boundary_reset_gate else None
        )
        self.readout = ScalePartitionedGDNReadout(cfg)

    def init_empty_states(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h, v, k = self.cfg.gdn_heads, self.cfg.gdn_d_v, self.cfg.gdn_d_k
        s1 = torch.zeros((batch_size, h, v, k), device=device, dtype=torch.float32)
        o1 = torch.zeros((batch_size, h, k), device=device, dtype=torch.float32)
        s2 = torch.zeros((batch_size, h, v, k), device=device, dtype=torch.float32)
        o2 = torch.zeros((batch_size, h, k), device=device, dtype=torch.float32)
        return s1, o1, s2, o2

    def forward_segment(
        self,
        gdn_slots: torch.Tensor,                     # (B, W, N_w, d_mem)
        kin: Dict[str, torch.Tensor],                # batched or 1D kinematics
        boundary_reset: Optional[torch.Tensor],      # (B, W) in [0, 1]
        init_states: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        compute_surprise: bool = True,
        force_pytorch: bool = False,
    ) -> Dict[str, Any]:
        """Run 2-layer `fp32` GDN across `W` written frames (`W * N_w` slots) in parallel."""
        b, w, nw, d = gdn_slots.shape
        dev = gdn_slots.device
        if init_states is None:
            s1_0, o1_0, s2_0, o2_0 = self.init_empty_states(b, dev)
        else:
            s1_0, o1_0, s2_0, o2_0 = init_states

        if w == 0:
            return {
                "S1": s1_0,
                "occ1": o1_0,
                "S2": s2_0,
                "occ2": o2_0,
                "surprise": torch.zeros((b, 0), device=dev, dtype=torch.float32),
            }

        vm = kin["vis_motion_rms"]
        gs = kin["grip_spike"]
        if vm.ndim == 1:
            vm = vm.unsqueeze(0).expand(b, -1)
            gs = gs.unsqueeze(0).expand(b, -1)

        k1, v1, g1, b1, q1 = self.layer1.project_slots(
            gdn_slots, vis_motion_rms=vm, grip_spike=gs, boundary_reset=boundary_reset
        )

        surprise = (
            self.layer1.compute_exact_write_surprise(
                k1.detach(),
                v1.detach(),
                g1.detach(),
                b1.detach(),
                num_write_slots=nw,
                initial_state_fp32=s1_0.detach(),
                force_pytorch=force_pytorch,
            )
            if compute_surprise
            else torch.zeros((b, w), device=dev, dtype=torch.float32)
        )

        raw_o1, s1_fin, o1_fin = run_dual_backend_gdn(
            k=k1,
            v=v1,
            g_fp32=g1,
            beta_fp32=b1,
            q=q1,
            initial_state_fp32=s1_0,
            initial_occ_fp32=o1_0,
            track_occ=True,
            force_pytorch=force_pytorch,
        )
        assert raw_o1 is not None and o1_fin is not None

        dt = self.layer1.k_proj.weight.dtype
        z_flat = gdn_slots.to(dt).reshape(b, w * nw, d)
        gate1 = F.silu(self.layer1.g_out_proj(z_flat)).view(b, w * nw, self.cfg.gdn_heads, self.cfg.gdn_d_v)
        l1_seq = self.layer1.o_proj(
            (gate1 * raw_o1.to(dt)).reshape(b, w * nw, self.cfg.gdn_heads * self.cfg.gdn_d_v)
        )
        if self.consolidator is not None:
            l1_seq = l1_seq + self.consolidator(z_flat, raw_o1)
        z_l2 = self.l2_norm(z_flat + l1_seq).view(b, w, nw, d)

        k2, v2, g2, b2, _ = self.layer2.project_slots(
            z_l2, vis_motion_rms=vm, grip_spike=gs, boundary_reset=boundary_reset
        )
        _, s2_fin, o2_fin = run_dual_backend_gdn(
            k=k2,
            v=v2,
            g_fp32=g2,
            beta_fp32=b2,
            q=None,
            initial_state_fp32=s2_0,
            initial_occ_fp32=o2_0,
            track_occ=True,
            force_pytorch=force_pytorch,
        )
        assert o2_fin is not None
        return {
            "S1": s1_fin,
            "occ1": o1_fin,
            "S2": s2_fin,
            "occ2": o2_fin,
            "surprise": surprise,
        }

    def forward_interleaved_episode_readout(
        self,
        gdn_slots: torch.Tensor,                     # (W, N_w, d_mem)
        kin: Dict[str, torch.Tensor],                # 1D kinematics of length W
        boundary_reset: Optional[torch.Tensor],      # (W,) in [0, 1] or None
        u_ends: torch.Tensor,                        # (M,) int64 causal written-frame cutoffs in [0, W]
        decay_factors: torch.Tensor,                 # (M,) fp32 pending-boundary decay multipliers in (0, 1]
        ctx_tokens: torch.Tensor,                    # (M, L_ctx, d_mem)
        extra_query_bias: Optional[torch.Tensor] = None,  # Optional (M, N_q, d_mem) for Stage-C Hop-2!
        force_pytorch: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Single-pass hardware-parallel causal readout across ALL `M` execution chunks (`2` GDN calls total!).

        Interleaves the `T_w = W_eff * N_w` write slots (`key = 2*s + 1`) with the `M * N_q` zero-write,
        zero-decay readout queries (`k=0, v=0, g=0, beta=0, q=Q[c, j]`, `key = 2 * (u_ends[c] * N_w)`).
        Because `g=0` (`exp(0)=1`) and `beta=0` at read tokens, GDN passes `(S, occ)` through untouched
        with `0.0` perturbation while returning exact `S(u_ends[c]) @ Q[c, j]` and `occ(u_ends[c]) @ Q[c, j]`
        for every chunk `c` in ONE parallel `chunk_gated_delta_rule` pass per layer!
        """
        dev = ctx_tokens.device
        m_chunks = int(ctx_tokens.shape[0])
        h, v_dim, k_dim = self.cfg.gdn_heads, self.cfg.gdn_d_v, self.cfg.gdn_d_k
        nw = int(self.num_write_slots)
        n_q = int(self.readout.num_tokens)
        inc_persist = extra_query_bias is None

        q_tok, q1_read, q2_read = self.readout.prepare_queries(ctx_tokens, extra_query_bias=extra_query_bias)
        w_eff = int(u_ends.max().item()) if (m_chunks > 0 and gdn_slots.shape[0] > 0) else 0
        if w_eff <= 0:
            z1 = torch.zeros((m_chunks, n_q, h, v_dim), device=dev, dtype=torch.float32)
            zo = torch.zeros((m_chunks, n_q, h), device=dev, dtype=torch.float32)
            return self.readout.finalize_from_raw_reads(q_tok, z1, zo, z1, zo, include_persistent=inc_persist)

        slots_w = gdn_slots[:w_eff].unsqueeze(0)  # (1, w_eff, nw, d)
        vm_w = kin["vis_motion_rms"][:w_eff].unsqueeze(0)
        gs_w = kin["grip_spike"][:w_eff].unsqueeze(0)
        br_w = boundary_reset[:w_eff].unsqueeze(0) if boundary_reset is not None else None

        t_w = w_eff * nw
        n_reads = m_chunks * n_q
        write_keys = torch.arange(t_w, device=dev, dtype=torch.long) * 2 + 1
        read_keys = (u_ends.long() * nw * 2).unsqueeze(1).expand(m_chunks, n_q).reshape(n_reads)
        all_keys = torch.cat([write_keys, read_keys], dim=0)
        perm = torch.argsort(all_keys, stable=True)
        inv_perm = torch.argsort(perm)
        write_pos = inv_perm[:t_w]
        read_pos = inv_perm[t_w:]

        # ---- Layer 1 Single-Pass Interleaved Write + Read-Before-Write + Chunk Readout ----
        k1_w, v1_w, g1_w, b1_w, q1_w = self.layer1.project_slots(
            slots_w, vis_motion_rms=vm_w, grip_spike=gs_w, boundary_reset=br_w
        )
        assert q1_w is not None
        ones_occ = torch.ones((1, t_w, h, 1), device=dev, dtype=torch.float32)
        v1_w_occ = torch.cat([v1_w.float(), ones_occ], dim=-1)

        zk = torch.zeros((1, n_reads, h, k_dim), device=dev, dtype=torch.float32)
        zv = torch.zeros((1, n_reads, h, v_dim + 1), device=dev, dtype=torch.float32)
        zg = torch.zeros((1, n_reads, h), device=dev, dtype=torch.float32)
        q1_r_flat = q1_read.reshape(1, n_reads, h, k_dim)

        k1_int = torch.cat([k1_w.float(), zk], dim=1)[:, perm].contiguous()
        v1_int = torch.cat([v1_w_occ, zv], dim=1)[:, perm].contiguous()
        g1_int = torch.cat([g1_w.float(), zg], dim=1)[:, perm].contiguous()
        b1_int = torch.cat([b1_w.float(), zg], dim=1)[:, perm].contiguous()
        q1_int = torch.cat([q1_w.float(), q1_r_flat], dim=1)[:, perm].contiguous()

        o1_int, _, _ = run_dual_backend_gdn(
            k=k1_int,
            v=v1_int,
            g_fp32=g1_int,
            beta_fp32=b1_int,
            q=q1_int,
            track_occ=False,
            force_pytorch=force_pytorch,
        )
        assert o1_int is not None
        raw_o1_w = o1_int[:, write_pos, :, :v_dim]  # (1, t_w, h, v_dim)
        o1_r = o1_int[0, read_pos].view(m_chunks, n_q, h, v_dim + 1)

        # ---- Cross-Scale Consolidation Gate + Layer 2 Single-Pass Interleaved Write + Readout ----
        dt = self.layer1.k_proj.weight.dtype
        z_flat = slots_w.to(dt).reshape(1, t_w, self.cfg.d_mem)
        gate1 = F.silu(self.layer1.g_out_proj(z_flat)).view(1, t_w, h, v_dim)
        l1_seq = self.layer1.o_proj((gate1 * raw_o1_w.to(dt)).reshape(1, t_w, h * v_dim))
        if self.consolidator is not None:
            l1_seq = l1_seq + self.consolidator(z_flat, raw_o1_w)
        z_l2 = self.l2_norm(z_flat + l1_seq).view(1, w_eff, nw, self.cfg.d_mem)

        k2_w, v2_w, g2_w, b2_w, _ = self.layer2.project_slots(
            z_l2, vis_motion_rms=vm_w, grip_spike=gs_w, boundary_reset=br_w
        )
        v2_w_occ = torch.cat([v2_w.float(), ones_occ], dim=-1)
        zq_w = torch.zeros((1, t_w, h, k_dim), device=dev, dtype=torch.float32)
        q2_r_flat = q2_read.reshape(1, n_reads, h, k_dim)

        k2_int = torch.cat([k2_w.float(), zk], dim=1)[:, perm].contiguous()
        v2_int = torch.cat([v2_w_occ, zv], dim=1)[:, perm].contiguous()
        g2_int = torch.cat([g2_w.float(), zg], dim=1)[:, perm].contiguous()
        b2_int = torch.cat([b2_w.float(), zg], dim=1)[:, perm].contiguous()
        q2_int = torch.cat([zq_w, q2_r_flat], dim=1)[:, perm].contiguous()

        o2_int, _, _ = run_dual_backend_gdn(
            k=k2_int,
            v=v2_int,
            g_fp32=g2_int,
            beta_fp32=b2_int,
            q=q2_int,
            track_occ=False,
            force_pytorch=force_pytorch,
        )
        assert o2_int is not None
        o2_r = o2_int[0, read_pos].view(m_chunks, n_q, h, v_dim + 1)

        dec4 = decay_factors.float().view(m_chunks, 1, 1, 1)
        dec3 = decay_factors.float().view(m_chunks, 1, 1)
        raw1 = o1_r[..., :v_dim] * dec4
        occ1_dot = o1_r[..., v_dim] * dec3
        raw2 = o2_r[..., :v_dim] * dec4
        occ2_dot = o2_r[..., v_dim] * dec3

        return self.readout.finalize_from_raw_reads(
            q_tok, raw1, occ1_dot, raw2, occ2_dot, include_persistent=inc_persist
        )


# =============================================================================
# 4. Non-Privileged Dual-Memory Landmark Bank (`GDN-Surprise` + Gripper Events)
# =============================================================================


class SurpriseEventLandmarkBank(nn.Module):
    """Selects `K` sparse visual/state landmark frames online with ZERO privileged labels.

    Salience score on written frame `m`:
      `score_m = w_surp * zscore(surprise_m) + w_grip * grip_spike_m + w_mot * vis_motion_rms_m`
    Quota & Anchor policy (strictly causal `< cutoff_frame` & flushed after any boundary reset `r_m > 0.5`):
      - Conditioning Demo (`is_demo == True`): always anchors `demo_first` and `demo_last`, plus up to
        `demo_landmark_quota - 2` highest-salience demo keyframes.
      - Execution (`is_demo == False`): fills remaining `K - K_demo` slots with highest-salience
        execution keyframes (gripper contact events, highlight flashes, swap completions).
      - Selected frames are sorted chronologically and tagged with relative age `step_t - step_lm`!
    """

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_landmarks = int(cfg.num_landmarks)
        self.tokens_per_lm = 16 if cfg.landmark_encoder == "multiscale_pyramid" else int(cfg.tokens_per_landmark)
        self.demo_quota = int(cfg.demo_landmark_quota)
        self.dim = int(cfg.mem_token_dim)

        self.rel_time_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.demo_tag_emb = nn.Embedding(2, self.dim)
        self.salience_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        nn.init.normal_(self.rel_time_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.demo_tag_emb.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def select_landmark_indices(
        self,
        write_frame_indices: torch.Tensor,   # (W,) int64 episode frame indices
        is_demo_w: torch.Tensor,             # (W,) bool
        salience_w: torch.Tensor,            # (W,) float32
        boundary_reset_w: Optional[torch.Tensor],  # Optional (W,) float32 in [0, 1]
        cutoff_excl_frame: int,
        pure_vis_salience_w: Optional[torch.Tensor] = None,  # Optional (W,) pure visual surprise without grip_spike
    ) -> torch.Tensor:
        """Return chronologically sorted indices into `[0 .. W-1]` of length `<= K`."""
        w = int(write_frame_indices.numel())
        dev = write_frame_indices.device
        if w == 0:
            return torch.empty((0,), dtype=torch.long, device=dev)

        valid_mask = write_frame_indices < int(cutoff_excl_frame)
        if not bool(valid_mask.any()):
            return torch.empty((0,), dtype=torch.long, device=dev)

        # Flush any frames prior to the most recent episode boundary reset `r_m > threshold`!
        start_pos = 0
        if boundary_reset_w is not None:
            valid_resets = torch.nonzero(
                valid_mask & (boundary_reset_w > float(self.cfg.boundary_flush_threshold)),
                as_tuple=False,
            ).flatten()
            if valid_resets.numel() > 0:
                start_pos = int(valid_resets[-1].item())

        active_mask = valid_mask.clone()
        if start_pos > 0:
            active_mask[:start_pos] = False

        cand_idx = torch.nonzero(active_mask, as_tuple=False).flatten()
        if cand_idx.numel() <= self.num_landmarks:
            return cand_idx

        demo_cands = cand_idx[is_demo_w[cand_idx]]
        exec_cands = cand_idx[~is_demo_w[cand_idx]]

        chosen: List[torch.Tensor] = []
        # `#315` Fix C: scale demo quota to `8` (leaving `4` exec slots) on long multi-stage demos (`VideoPlaceButton` ~760f)!
        eff_demo_quota = self.demo_quota
        if int(demo_cands.numel()) >= 96 and self.num_landmarks >= 10:
            eff_demo_quota = max(eff_demo_quota, min(8, (2 * self.num_landmarks) // 3))
        demo_budget = min(int(demo_cands.numel()), eff_demo_quota)
        if exec_cands.numel() == 0:
            demo_budget = min(int(demo_cands.numel()), self.num_landmarks)

        if demo_budget > 0:
            if demo_cands.numel() <= demo_budget:
                chosen.append(demo_cands)
            else:
                anchors = [demo_cands[:1]]
                if demo_budget >= 2:
                    anchors.append(demo_cands[-1:])
                rem = demo_budget - len(anchors)
                if rem > 0 and demo_cands.numel() > 2:
                    mid = demo_cands[1:-1]
                    topk_mid = torch.topk(salience_w[mid], k=min(rem, int(mid.numel()))).indices
                    anchors.append(mid[topk_mid])
                chosen.append(torch.cat(anchors, dim=0))

        used_n = sum(int(t.numel()) for t in chosen)
        exec_budget = self.num_landmarks - used_n
        if exec_budget > 0 and exec_cands.numel() > 0:
            if exec_cands.numel() <= exec_budget:
                chosen.append(exec_cands)
            else:
                if exec_budget == 1:
                    top_ex = torch.topk(salience_w[exec_cands], k=1).indices
                    chosen.append(exec_cands[top_ex])
                elif int(demo_cands.numel()) == 0 and exec_budget >= 6 and int(exec_cands.numel()) > exec_budget:
                    # `#315` Fix B: On no-demo tasks (`PickHighlight`, `ButtonUnmaskSwap`), protect `early_quota = 4`
                    # pure visual-surprise frames (`grip_spike == 0` flashes/reveals) from eviction by later grasps!
                    early_quota = min(4, exec_budget // 3)
                    early_len = min(120, max(early_quota, int(exec_cands.numel()) // 2))
                    early_pool = exec_cands[:early_len]
                    vis_src = pure_vis_salience_w if pure_vis_salience_w is not None else salience_w
                    top_early = early_pool[torch.topk(vis_src[early_pool], k=min(early_quota, int(early_pool.numel()))).indices]
                    recent_anchor = exec_cands[-1:]
                    keep_set = set(top_early.tolist()) | set(recent_anchor.tolist())
                    rem_pool = torch.tensor(
                        [int(x) for x in exec_cands[:-1].tolist() if int(x) not in keep_set],
                        device=dev,
                        dtype=torch.long,
                    )
                    rem_k = max(0, exec_budget - len(keep_set))
                    if rem_k > 0 and rem_pool.numel() > 0:
                        top_rem = rem_pool[torch.topk(salience_w[rem_pool], k=min(rem_k, int(rem_pool.numel()))).indices]
                        chosen.append(torch.cat([top_early, top_rem, recent_anchor], dim=0))
                    else:
                        chosen.append(torch.cat([top_early, recent_anchor], dim=0))
                else:
                    recent_anchor = exec_cands[-1:]
                    pool = exec_cands[:-1]
                    top_ex = torch.topk(salience_w[pool], k=min(exec_budget - 1, int(pool.numel()))).indices
                    chosen.append(torch.cat([pool[top_ex], recent_anchor], dim=0))

        merged = torch.unique(torch.cat(chosen, dim=0), sorted=True)
        if merged.numel() > self.num_landmarks:
            top_final = torch.topk(salience_w[merged], k=self.num_landmarks).indices
            merged = torch.sort(merged[top_final]).values
        return merged

    def gather_padded_landmarks(
        self,
        lm_slots_w: torch.Tensor,            # (W, M_lm, 1024)
        write_frame_indices: torch.Tensor,   # (W,)
        is_demo_w: torch.Tensor,             # (W,) bool
        salience_w: torch.Tensor,            # (W,) float32
        selected_pos: torch.Tensor,          # (K_sel,) indices into [0..W-1]
        query_step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gather `(K * M_lm, 1024)` landmark tokens + `(K * M_lm,)` bool mask for one query step."""
        dev = lm_slots_w.device
        dt = self.rel_time_proj.weight.dtype
        max_tok = self.num_landmarks * self.tokens_per_lm
        out_tok = torch.zeros((max_tok, self.dim), device=dev, dtype=dt)
        out_mask = torch.zeros((max_tok,), device=dev, dtype=torch.bool)

        k_sel = int(selected_pos.numel())
        if k_sel == 0:
            return out_tok, out_mask

        sel_slots = lm_slots_w[selected_pos].to(dt)  # (K_sel, M_lm, 1024)
        sel_steps = write_frame_indices[selected_pos].float()
        rel_age = (float(query_step_idx) - sel_steps).clamp(min=0.0) / 8.0
        age_emb = self.rel_time_proj(build_1d_sincos_embed(rel_age, self.dim).to(dt))  # (K_sel, 1024)
        demo_emb = self.demo_tag_emb(is_demo_w[selected_pos].long().clamp(0, 1)).to(dt)

        sal_norm = torch.tanh(salience_w[selected_pos].float() / 3.0).view(k_sel, 1, 1)
        scale = (1.0 + torch.tanh(self.salience_gate) * sal_norm).to(dt)
        enriched = sel_slots * scale + (age_emb + demo_emb).unsqueeze(1)

        n_valid = k_sel * self.tokens_per_lm
        out_tok[:n_valid] = enriched.reshape(n_valid, self.dim)
        out_mask[:n_valid] = True
        return out_tok, out_mask


# =============================================================================
# 5. Clean `T²Mem` Baseline (`P2-B1`) & `FrameSamp` Baseline (`P2-B2`)
# =============================================================================


class T2MemFastWeightMemory(nn.Module):
    """`T²Mem`-style (`arXiv:2609.36720` / LaCT-adapted) 1-Layer Online Fast-Weight TTT Baseline.

    Upgrades upstream `MME-VLA` static-slice `TTTLayerLinear` to `T²Mem`'s **observation-grounded
    query readout** over online fast weights `(W_t, b_t)` updated via residual reconstruction MSE:
      `L_TTT(W, b) = 0.5 * || LayerNorm(X_k @ W + b) - (X_v - X_k) ||_2^2`
    with input-dependent step size `eta_t` and Frobenius gradient-norm clipping.
    """

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = int(cfg.t2mem_heads)
        self.head_dim = int(cfg.t2mem_head_dim)
        self.hidden_dim = self.num_heads * self.head_dim
        self.mb_frames = max(1, int(cfg.t2mem_mini_batch_frames))
        self.max_lr = float(cfg.t2mem_max_lr)
        self.max_grad_norm = float(cfg.t2mem_max_grad_norm)
        self.num_queries = int(cfg.t2mem_num_query_tokens)
        self.reset_strength = float(cfg.boundary_reset_strength)

        self.in_proj = nn.Linear(cfg.d_mem, self.hidden_dim)
        self.wk = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.wv = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.v_norm = nn.LayerNorm(self.head_dim)

        base_lr = max(1e-4, float(cfg.t2mem_base_lr))
        inv_sp = math.log(math.expm1(base_lr))
        self.lr_bias = nn.Parameter(torch.full((self.num_heads,), inv_sp, dtype=torch.float32))
        self.lr_proj = nn.Linear(self.hidden_dim, self.num_heads)
        nn.init.zeros_(self.lr_proj.weight)
        nn.init.zeros_(self.lr_proj.bias)

        # Learned initial fast weights `(W0, b0)` + per-head LayerNorm affine `(gamma, beta)`
        self.W0 = nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.head_dim) / math.sqrt(self.head_dim))
        self.b0 = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.ln_gamma = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.ln_beta = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))

        # Observation-grounded query generator (`T²Mem` core upgrade over static tail slicing!)
        self.query_seeds = nn.Parameter(torch.randn(self.num_queries, cfg.d_mem) * 0.02)
        self.q_ctx_attn = MultiHeadAttn(cfg.d_mem, cfg.mem_enc_heads)
        self.wq_obs = nn.Linear(cfg.d_mem, self.hidden_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, cfg.mem_token_dim),
            nn.RMSNorm(cfg.mem_token_dim),
        )
        self.boundary_gate: Optional[EpisodeBoundaryForgetGate] = (
            EpisodeBoundaryForgetGate(cfg.d_mem) if cfg.use_boundary_reset_gate else None
        )

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.W0.float().unsqueeze(0).expand(batch_size, -1, -1, -1).clone()
        b = self.b0.float().unsqueeze(0).expand(batch_size, -1, -1).clone()
        return w.to(device), b.to(device)

    def _ln_and_vjp(
        self, z: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact analytical LayerNorm forward + VJP `d(0.5 ||LN(z) - target||^2) / dz` in `fp32`."""
        mean = z.mean(dim=-1, keepdim=True)
        xc = z - mean
        var = (xc * xc).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + 1e-6)
        x_hat = xc * rstd
        gam = self.ln_gamma.float().view(1, self.num_heads, 1, self.head_dim)
        bet = self.ln_beta.float().view(1, self.num_heads, 1, self.head_dim)
        y = x_hat * gam + bet
        err = y - target
        dy = err * gam
        d = float(self.head_dim)
        dz = rstd * (dy - dy.mean(dim=-1, keepdim=True) - x_hat * (dy * x_hat).mean(dim=-1, keepdim=True))
        return y, dz

    def update_segment(
        self,
        gdn_slots: torch.Tensor,                    # (B, W, N_w, d_mem)
        boundary_reset: Optional[torch.Tensor],     # Optional (B, W) in [0, 1]
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update fast weights `(W_t, b_t)` over `W` written frames in mini-batches of `mb_frames`."""
        bsz, w_frames, nw, _ = gdn_slots.shape
        dev = gdn_slots.device
        w_cur, b_cur = self.init_state(bsz, dev) if state is None else state
        if w_frames == 0:
            return w_cur, b_cur

        dt = self.in_proj.weight.dtype
        with torch.autocast(device_type=dev.type, enabled=False):
            h_in = self.in_proj(gdn_slots.to(dt))  # (B, W, Nw, hidden_dim)
            xk = self.k_norm(self.wk(h_in).view(bsz, w_frames, nw, self.num_heads, self.head_dim)).float()
            xv = self.v_norm(self.wv(h_in).view(bsz, w_frames, nw, self.num_heads, self.head_dim)).float()
            eta = F.softplus(self.lr_proj(h_in).float() + self.lr_bias.float()).clamp(0.0, self.max_lr)
            # Rearrange -> (B, H, W, Nw, D_h)
            xk = xk.permute(0, 3, 1, 2, 4).contiguous()
            xv = xv.permute(0, 3, 1, 2, 4).contiguous()
            eta = eta.permute(0, 3, 1, 2).contiguous()

            w0_ref = self.W0.float().unsqueeze(0)
            b0_ref = self.b0.float().unsqueeze(0)

            for f0 in range(0, w_frames, self.mb_frames):
                f1 = min(f0 + self.mb_frames, w_frames)
                if boundary_reset is not None:
                    r_blk = boundary_reset[:, f0:f1].max(dim=1).values.view(bsz, 1, 1, 1)
                    keep = torch.exp(-self.reset_strength * r_blk)
                    w_cur = keep * w_cur + (1.0 - keep) * w0_ref
                    b_cur = keep.squeeze(-1) * b_cur + (1.0 - keep.squeeze(-1)) * b0_ref

                n_tok = (f1 - f0) * nw
                k_mb = xk[:, :, f0:f1].reshape(bsz, self.num_heads, n_tok, self.head_dim)
                v_mb = xv[:, :, f0:f1].reshape(bsz, self.num_heads, n_tok, self.head_dim)
                eta_mb = (eta[:, :, f0:f1].reshape(bsz, self.num_heads, n_tok, 1)) / float(max(n_tok, 1))

                z = torch.matmul(k_mb, w_cur) + b_cur.unsqueeze(-2)
                _, dz = self._ln_and_vjp(z, v_mb - k_mb)

                dw = torch.matmul(k_mb.transpose(-1, -2), eta_mb * dz)  # (B, H, D_h, D_h)
                db = (eta_mb * dz).sum(dim=-2)                          # (B, H, D_h)

                gn = torch.linalg.vector_norm(dw, ord=2, dim=(-2, -1), keepdim=True)
                scale_w = torch.where(gn > self.max_grad_norm, self.max_grad_norm / (gn + 1e-8), torch.ones_like(gn))
                w_cur = w_cur - dw * scale_w
                b_cur = b_cur - db

        return w_cur, b_cur

    def query_readout(
        self,
        ctx_tokens: torch.Tensor,                # (B, L_ctx, d_mem)
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Observation-grounded readout `LN(Q_obs @ W_t + b_t) + Q_obs -> [B, N_q, 1024]`."""
        bsz = int(ctx_tokens.shape[0])
        dt = self.wq_obs.weight.dtype
        w_cur, b_cur = state
        q_seed = self.query_seeds.to(dt).unsqueeze(0).expand(bsz, -1, -1)
        q_ctx = q_seed + self.q_ctx_attn(q_seed, ctx_tokens.to(dt))
        xq = self.q_norm(self.wq_obs(q_ctx).view(bsz, self.num_queries, self.num_heads, self.head_dim))
        xq_h = xq.float().permute(0, 2, 1, 3)  # (B, H, N_q, D_h)

        with torch.autocast(device_type=ctx_tokens.device.type, enabled=False):
            z = torch.matmul(xq_h, w_cur.float()) + b_cur.float().unsqueeze(-2)
            mean = z.mean(dim=-1, keepdim=True)
            xc = z - mean
            rstd = torch.rsqrt((xc * xc).mean(dim=-1, keepdim=True) + 1e-6)
            gam = self.ln_gamma.float().view(1, self.num_heads, 1, self.head_dim)
            bet = self.ln_beta.float().view(1, self.num_heads, 1, self.head_dim)
            y = (xc * rstd) * gam + bet + xq_h

        y_flat = y.permute(0, 2, 1, 3).reshape(bsz, self.num_queries, self.hidden_dim).to(dt)
        return self.out_proj(y_flat)


class PerceptualFrameSampMemory(nn.Module):
    """Strided Perceptual Frame-Sampling buffer (`P2-B2`, `FrameSamp+Modul` baseline)."""

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_frames = int(cfg.framesamp_num_frames)
        self.tok_per_frame = int(cfg.framesamp_tokens_per_frame)
        self.dim = int(cfg.mem_token_dim)
        self.proj = nn.Sequential(
            nn.RMSNorm(cfg.d_mem),
            nn.Linear(cfg.d_mem, self.dim),
        )
        self.rel_time_proj = nn.Linear(self.dim, self.dim, bias=False)
        nn.init.normal_(self.rel_time_proj.weight, mean=0.0, std=0.02)

    def gather_frames(
        self,
        gdn_slots_w: torch.Tensor,          # (W, N_w, d_mem)
        write_frame_indices: torch.Tensor,  # (W,)
        cutoff_excl_frame: int,
        query_step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = gdn_slots_w.device
        dt = self.rel_time_proj.weight.dtype
        max_tok = self.num_frames * self.tok_per_frame
        out_tok = torch.zeros((max_tok, self.dim), device=dev, dtype=dt)
        out_mask = torch.zeros((max_tok,), device=dev, dtype=torch.bool)

        valid = torch.nonzero(write_frame_indices < int(cutoff_excl_frame), as_tuple=False).flatten()
        if valid.numel() == 0:
            return out_tok, out_mask

        if valid.numel() > self.num_frames:
            lin = torch.linspace(0, valid.numel() - 1, steps=self.num_frames, device=dev).round().long()
            sel = valid[lin]
        else:
            sel = valid

        slots = gdn_slots_w[sel, : self.tok_per_frame, :].to(dt)  # (F_sel, tok_per_frame, d_mem)
        proj_slots = self.proj(slots)
        rel_age = (float(query_step_idx) - write_frame_indices[sel].float()).clamp(min=0.0) / 8.0
        t_emb = self.rel_time_proj(build_1d_sincos_embed(rel_age, self.dim).to(dt)).unsqueeze(1)
        toks = (proj_slots + t_emb).reshape(-1, self.dim)
        n = int(toks.shape[0])
        out_tok[:n] = toks
        out_mask[:n] = True
        return out_tok, out_mask


# =============================================================================
# 6. Dense Training-Time Grounded Supervision & Self-Predicted Subgoal Injector
# =============================================================================


class GroundedAuxiliarySuite(nn.Module):
    """Dense training-time grounded supervision + optional zero-oracle test-time pointer injection!

    Heads supervised during training (`0` oracle inputs at test time!):
      1. `xy_norm` (`[-1, 1]^2` primary & secondary `<x, y>` via `SpatialSoftmax` over `front` grid +
         Gaussian KL spatial heatmap supervision — exact `align_corners=False` match!).
      2. `completed_count` (`0..11`), `remaining_count` (`0..11`), `ordinal_index` (`0..11`).
      3. `stage_id` (`0..14`), `color_id` (`0..8`), & `subgoal_latent` (`subgoal_latent_dim`).
    Optional Test-Time Self-Predicted Subgoal Token Injection (`use_self_subgoal_tokens=True`):
      Projects `[xy_pred, sec_xy_pred, count_probs, rem_probs, stage_probs, sg_latent]` into `2`
      `1024-D` grounded pointer tokens with **zero-initialized output projection** (`0.0` diff at step 0!).
    """

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d_mem = int(cfg.d_mem)
        d_tok = int(cfg.mem_token_dim)
        self.front_n = int(cfg.front_tokens_hist)
        self.grid_side = int(round(self.front_n**0.5))

        self.mem_summary_proj = nn.Linear(d_tok, d_mem)
        self.spatial_q_proj = nn.Linear(d_mem * 2, d_mem)
        self.spatial_k_proj = nn.Linear(d_mem, d_mem)
        # 2 spatial heatmap channels: ch 0 = primary `<x1, y1>`, ch 1 = secondary `<x2, y2>`
        self.heatmap_head = nn.Sequential(
            nn.RMSNorm(d_mem),
            nn.Linear(d_mem, 2),
        )

        fusion_dim = d_mem * 2
        self.trunk = nn.Sequential(
            nn.RMSNorm(fusion_dim),
            nn.Linear(fusion_dim, d_mem),
            nn.SiLU(),
        )
        self.completed_count_head = nn.Linear(d_mem, cfg.max_count_classes)
        self.remaining_count_head = nn.Linear(d_mem, cfg.max_count_classes)
        self.ordinal_head = nn.Linear(d_mem, cfg.max_count_classes)
        self.stage_head = nn.Linear(d_mem, cfg.num_stage_classes)
        self.color_head = nn.Linear(d_mem, cfg.num_color_classes)
        self.subgoal_latent_head = nn.Linear(d_mem, cfg.subgoal_latent_dim)

        self.use_self_tokens = bool(cfg.use_self_subgoal_tokens)
        self.num_self_tokens = int(cfg.num_self_subgoal_tokens)
        self.fourier_xy_dim = 32
        if self.use_self_tokens and self.num_self_tokens > 0:
            in_sg = (
                4
                + 2 * self.fourier_xy_dim
                + 3 * cfg.max_count_classes
                + cfg.num_stage_classes
                + cfg.num_color_classes
                + cfg.subgoal_latent_dim
            )
            self.self_token_mlp = nn.Sequential(
                nn.Linear(in_sg, d_mem),
                nn.SiLU(),
                nn.Linear(d_mem, self.num_self_tokens * d_tok),
            )
            # Zero-init final layer so Step-0 self-predicted tokens contribute exact `0.0`!
            nn.init.zeros_(self.self_token_mlp[-1].weight)  # type: ignore[union-attr]
            nn.init.zeros_(self.self_token_mlp[-1].bias)    # type: ignore[union-attr]

    def forward(
        self,
        front_spatial_ctx: torch.Tensor,   # (B, P_f, d_mem)
        ctx_tokens: torch.Tensor,          # (B, L_ctx, d_mem)
        mem_tokens: Optional[torch.Tensor],  # Optional (B, S_base, 1024)
        mem_mask: Optional[torch.Tensor],    # Optional (B, S_base) bool
    ) -> Dict[str, torch.Tensor]:
        bsz = int(front_spatial_ctx.shape[0])
        dt = self.spatial_q_proj.weight.dtype
        ctx_pool = ctx_tokens.to(dt).mean(dim=1)  # (B, d_mem)

        if mem_tokens is not None and mem_tokens.shape[1] > 0:
            if mem_mask is not None:
                mf = mem_mask.to(dt).unsqueeze(-1)
                has_m = mem_mask.any(dim=-1, keepdim=True).to(dt)
                mp = ((mem_tokens.to(dt) * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)) * has_m
            else:
                mp = mem_tokens.to(dt).mean(dim=1)
            mem_pool = self.mem_summary_proj(mp)
        else:
            mem_pool = torch.zeros_like(ctx_pool)

        joint = torch.cat([ctx_pool, mem_pool], dim=-1)  # (B, 2 * d_mem)
        q_sp = F.silu(self.spatial_q_proj(joint)).unsqueeze(1)  # (B, 1, d_mem)
        k_sp = self.spatial_k_proj(front_spatial_ctx.to(dt))     # (B, P_f, d_mem)
        hm_logits = self.heatmap_head(k_sp * q_sp).permute(0, 2, 1).contiguous()  # (B, 2, P_f)

        with torch.autocast(device_type=front_spatial_ctx.device.type, enabled=False):
            xy_pri, probs_pri = spatial_softmax_2d(
                hm_logits[:, 0, :].float(), grid_hw=(self.grid_side, self.grid_side), align_corners=False
            )
            xy_sec, probs_sec = spatial_softmax_2d(
                hm_logits[:, 1, :].float(), grid_hw=(self.grid_side, self.grid_side), align_corners=False
            )

        h = self.trunk(joint)
        comp_logits = self.completed_count_head(h).float()
        rem_logits = self.remaining_count_head(h).float()
        ord_logits = self.ordinal_head(h).float()
        stage_logits = self.stage_head(h).float()
        color_logits = self.color_head(h).float()
        sg_latent = self.subgoal_latent_head(h).float()

        out: Dict[str, torch.Tensor] = {
            "xy_pred": xy_pri,
            "sec_xy_pred": xy_sec,
            "hm_logits": hm_logits.float(),
            "hm_probs_pri": probs_pri,
            "hm_probs_sec": probs_sec,
            "completed_count_logits": comp_logits,
            "remaining_count_logits": rem_logits,
            "ordinal_logits": ord_logits,
            "stage_logits": stage_logits,
            "color_logits": color_logits,
            "subgoal_latent": sg_latent,
        }

        if self.use_self_tokens and self.num_self_tokens > 0:
            xy_in = xy_pri
            if self.training and self.cfg.train_stage == "stage_b" and self.cfg.self_xy_jitter_std > 0.0:
                # MME-VLA `dataset.py:171` ±8/112 truncated Gaussian jitter prevents rigid pointer overfitting
                jit = (torch.randn_like(xy_in) * float(self.cfg.self_xy_jitter_std)).clamp(-0.12, 0.12)
                xy_in = (xy_in + jit).clamp(-1.0, 1.0)
            fx = build_1d_sincos_embed(xy_in[:, 0] * 16.0, self.fourier_xy_dim, max_period=64.0)
            fy = build_1d_sincos_embed(xy_in[:, 1] * 16.0, self.fourier_xy_dim, max_period=64.0)
            sg_vec = torch.cat(
                [
                    xy_in,
                    xy_sec,
                    fx,
                    fy,
                    F.softmax(comp_logits, dim=-1),
                    F.softmax(rem_logits, dim=-1),
                    F.softmax(ord_logits, dim=-1),
                    F.softmax(stage_logits, dim=-1),
                    F.softmax(color_logits, dim=-1),
                    F.normalize(sg_latent, dim=-1),
                ],
                dim=-1,
            ).to(dt)
            self_tok = self.self_token_mlp(sg_vec).view(bsz, self.num_self_tokens, self.cfg.mem_token_dim)
            out["self_subgoal_tokens"] = self_tok
        return out

    def compute_aux_losses(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute masked `<x,y>` SmoothL1 + 2D Gaussian KL + Count CE + Stage/Color/Latent losses."""
        dev = preds["xy_pred"].device
        zero = torch.tensor(0.0, device=dev, dtype=torch.float32)

        # 1. 2D `<x, y>` regression + Gaussian heatmap KL loss on `has_xy` samples
        xy_loss = zero
        if "target_xy" in targets and "has_xy" in targets:
            has_xy = targets["has_xy"].bool()
            if bool(has_xy.any()):
                tgt_xy = targets["target_xy"].float()[has_xy]
                pred_xy = preds["xy_pred"][has_xy]
                reg = F.smooth_l1_loss(pred_xy, tgt_xy, beta=0.05)

                # Smooth Gaussian target over `(grid_side x grid_side)` patch centers (`align_corners=False`)
                g_s = self.grid_side
                centers = ((torch.arange(g_s, device=dev, dtype=torch.float32) + 0.5) / float(g_s)) * 2.0 - 1.0
                yy, xx = torch.meshgrid(centers, centers, indexing="ij")
                grid_xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1).unsqueeze(0)  # (1, P_f, 2)
                sigma2 = (1.5 / float(g_s)) ** 2
                d2 = ((grid_xy - tgt_xy.unsqueeze(1)) ** 2).sum(dim=-1)
                tgt_prob = F.softmax(-0.5 * d2 / sigma2, dim=-1)
                log_p = F.log_softmax(preds["hm_logits"][has_xy, 0, :], dim=-1)
                kl = F.kl_div(log_p, tgt_prob, reduction="batchmean")
                xy_loss = reg + 0.25 * kl

        if "secondary_target_xy" in targets and "has_secondary_xy" in targets:
            has_sec = targets["has_secondary_xy"].bool()
            if bool(has_sec.any()):
                xy_loss = xy_loss + 0.5 * F.smooth_l1_loss(
                    preds["sec_xy_pred"][has_sec], targets["secondary_target_xy"].float()[has_sec], beta=0.05
                )

        # 2. Action pass / pick / swing completed & remaining count losses
        cnt_loss = zero
        cnt_terms = 0
        max_c = self.cfg.max_count_classes - 1
        for pred_k, tgt_k in (
            ("completed_count_logits", "completed_count"),
            ("remaining_count_logits", "remaining_count"),
            ("ordinal_logits", "ordinal_index"),
        ):
            if tgt_k in targets:
                t_c = targets[tgt_k].long().clamp(0, max_c)
                cnt_loss = cnt_loss + F.cross_entropy(preds[pred_k], t_c)
                cnt_terms += 1
        if cnt_terms > 0:
            cnt_loss = cnt_loss / float(cnt_terms)

        # 3. Grounded subgoal stage, color, and continuous latent distillation
        sg_loss = zero
        if "stage_id" in targets:
            sg_loss = sg_loss + F.cross_entropy(
                preds["stage_logits"], targets["stage_id"].long().clamp(0, self.cfg.num_stage_classes - 1)
            )
        if "color_id" in targets:
            sg_loss = sg_loss + 0.5 * F.cross_entropy(
                preds["color_logits"], targets["color_id"].long().clamp(0, self.cfg.num_color_classes - 1)
            )
        if "target_subgoal_latent" in targets and "has_subgoal_latent" in targets:
            has_lat = targets["has_subgoal_latent"].bool()
            if bool(has_lat.any()):
                p_lat = F.normalize(preds["subgoal_latent"][has_lat], dim=-1)
                t_lat = F.normalize(targets["target_subgoal_latent"].float()[has_lat], dim=-1)
                sg_loss = sg_loss + (1.0 - (p_lat * t_lat).sum(dim=-1)).mean()

        total_aux = (
            self.cfg.xy_loss_weight * xy_loss
            + self.cfg.count_loss_weight * cnt_loss
            + self.cfg.subgoal_loss_weight * sg_loss
        )
        return {
            "aux_total": total_aux,
            "xy_loss": xy_loss.detach(),
            "count_loss": cnt_loss.detach(),
            "subgoal_loss": sg_loss.detach(),
        }

    @staticmethod
    def format_predicted_subgoal_strings(preds: Dict[str, torch.Tensor]) -> List[str]:
        """Format zero-oracle self-predicted `<row_y, col_x>` strings matching RoboMME `<y, x>` syntax!"""
        xy_px = norm_xy_to_pixel_xy(preds["xy_pred"].detach().cpu(), image_hw=(256, 256), align_corners=False)
        st_ids = preds["stage_logits"].detach().argmax(dim=-1).cpu().tolist()
        rem_ids = preds["remaining_count_logits"].detach().argmax(dim=-1).cpu().tolist()
        out: List[str] = []
        for i, st_i in enumerate(st_ids):
            col_x = int(round(float(xy_px[i, 0].clamp(0.0, 255.0).item())))
            row_y = int(round(float(xy_px[i, 1].clamp(0.0, 255.0).item())))
            st_name = STAGE_VOCAB[st_i].replace("_", " ")
            out.append(f"{st_name} at <{row_y}, {col_x}> (remaining: {rem_ids[i]})")
        return out


# =============================================================================
# 7. Stage-C Zero-Init Residual Multi-Hop Refiner + `M_CRN=8` Halting Gate
# =============================================================================


class ResidualMultiHopRefiner(nn.Module):
    """Stage-C Zero-Init Residual Multi-Hop Query Refinement (`M^(2) = M^(1) + tanh(gamma) * ΔM^(2)`).

    Guarantees `gamma(0) == 0.0 => M^(2) == M^(1)` bit-for-bit at Stage-C initialization (`SR_L2 >= SR_L1`)!
    """

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        d_tok = int(cfg.mem_token_dim)
        d_mem = int(cfg.d_mem)
        self.num_gdn = int(cfg.num_gdn_readout_tokens)

        # Refine GDN query bias from `(E_t, M^(1))` for Hop-2 re-querying of `(S1_past, S2_past)`
        self.hop1_to_qbias = nn.Sequential(
            nn.RMSNorm(d_tok),
            nn.Linear(d_tok, d_mem),
            nn.SiLU(),
            nn.Linear(d_mem, d_mem),
        )
        # Cross-attend from `M^(1)` over `[M_re_queried || E_t]` to refine all memory tokens
        self.ctx_to_tok = nn.Linear(d_mem, d_tok, bias=False)
        self.q_norm = nn.RMSNorm(d_tok)
        self.kv_norm = nn.RMSNorm(d_tok)
        self.hop2_attn = MultiHeadAttn(d_tok, num_heads=4)
        self.ffn_norm = nn.RMSNorm(d_tok)
        self.hop2_ffn = SwiGLU(d_tok, d_tok * 2)
        self.out_norm = nn.RMSNorm(d_tok)

        # Strict `gamma_0 = 0.0` scalar residual gate!
        self.gamma_hop2 = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def compute_query_bias(self, mem_hop1_gdn: torch.Tensor) -> torch.Tensor:
        return self.hop1_to_qbias(mem_hop1_gdn)

    def forward(
        self,
        mem_hop1: torch.Tensor,          # (B, S_mem, 1024)
        mem_mask: torch.Tensor,          # (B, S_mem) bool
        gdn_hop2_tokens: torch.Tensor,   # (B, N_gdn, 1024)
        ctx_tokens: torch.Tensor,        # (B, L_ctx, d_mem)
    ) -> torch.Tensor:
        dt = mem_hop1.dtype
        ctx_hi = self.ctx_to_tok(ctx_tokens.to(self.ctx_to_tok.weight.dtype)).to(dt)
        kv = self.kv_norm(torch.cat([gdn_hop2_tokens.to(dt), ctx_hi], dim=1))
        q = self.q_norm(mem_hop1)
        # `#260` Fix: route FFN strictly through `d1` (no `q` self-amplification shortcut) + RMS-matched unit output!
        d1 = self.hop2_attn(q, kv)
        delta_unit = self.out_norm(d1 + self.hop2_ffn(self.ffn_norm(d1)))
        m1_rms = mem_hop1.detach().float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().to(dt)
        gate = torch.tanh(self.gamma_hop2).to(dt)
        refined = mem_hop1 + gate * (delta_unit * m1_rms)
        return torch.where(mem_mask.unsqueeze(-1), refined, torch.zeros_like(refined))


class StageCCRNGate(nn.Module):
    """Invariant B4: Relative-quantile Common-Random-Number (`M_CRN=8`) adaptive Hop-2 gate."""

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        d_in = int(cfg.d_mem) + int(cfg.mem_token_dim) + 2
        self.net = nn.Sequential(
            nn.Linear(d_in, cfg.d_mem // 2),
            nn.SiLU(),
            nn.Linear(cfg.d_mem // 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)  # type: ignore[union-attr]
        nn.init.constant_(self.net[-1].bias, -0.5)
        self.register_buffer("calibrated_delta", torch.tensor(1e-4, dtype=torch.float32))

    def forward(
        self,
        ctx_tokens: torch.Tensor,
        mem_hop1: torch.Tensor,
        mem_mask: torch.Tensor,
        cov_stats: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return Hop-2 activation probability `p_hop2: (B,)` in `(0, 1)` (`logits` returned via `.logit` if needed)."""
        dt = self.net[0].weight.dtype  # type: ignore[index]
        c_pool = ctx_tokens.to(dt).mean(dim=1)
        mf = mem_mask.to(dt).unsqueeze(-1)
        has_m = mem_mask.any(dim=-1, keepdim=True).to(dt)
        m_pool = ((mem_hop1.to(dt) * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)) * has_m
        cov = torch.stack([cov_stats["cov1"], cov_stats["cov2"]], dim=-1).to(dt)
        logits = self.net(torch.cat([c_pool, m_pool, cov], dim=-1)).float().squeeze(-1)
        return logits


# =============================================================================
# 8. Track-2 Hybrid Surprise-Keyframe + Multi-Timescale GDN Bridge (`P2-M5`)
# =============================================================================


class TorchPosEmb3D(nn.Module):
    """Pure GPU PyTorch exact port of `src.pi05.perceptual_mem.PosEmb3D` (`< 1e-6` diff, zero CPU sync)."""

    temporal_omega: torch.Tensor
    spatial_pe_4: torch.Tensor

    def __init__(self, dim: int = 768, temporal_base: float = 10_000.0, spatial_base: float = 1_000.0) -> None:
        super().__init__()
        if dim % 6 != 0:
            raise ValueError(f"dim={dim} must be divisible by 6")
        self.dim = int(dim)
        width = self.dim // 6
        omega = torch.arange(width, dtype=torch.float32) / float(width - 1)
        t_omega = 1.0 / (temporal_base**omega)
        s_omega = 1.0 / (spatial_base**omega)

        yy, xx = torch.meshgrid(
            torch.arange(4, dtype=torch.float32),
            torch.arange(4, dtype=torch.float32),
            indexing="ij",
        )
        yf = (4.0 * yy + 2.0).reshape(-1, 1)
        xf = (4.0 * xx + 2.0).reshape(-1, 1)
        ye = yf * s_omega.unsqueeze(0)
        xe = xf * s_omega.unsqueeze(0)
        sp4 = torch.cat([torch.sin(ye), torch.cos(ye), torch.sin(xe), torch.cos(xe)], dim=-1)

        self.register_buffer("temporal_omega", t_omega, persistent=False)
        self.register_buffer("spatial_pe_4", sp4, persistent=False)

    def forward(self, pos_indices: torch.Tensor) -> torch.Tensor:
        """Map `(L,)` `int64` local frame positions -> `(L, 16, 768)` `float32` 3D sin/cos embeddings."""
        l_len = int(pos_indices.numel())
        if l_len == 0:
            return torch.empty((0, 16, self.dim), device=pos_indices.device, dtype=torch.float32)
        te = pos_indices.float().unsqueeze(-1) * self.temporal_omega.unsqueeze(0)
        tp = torch.cat([torch.sin(te), torch.cos(te)], dim=-1).unsqueeze(1).expand(l_len, 16, -1)
        sp = self.spatial_pe_4.unsqueeze(0).expand(l_len, 16, -1)
        return torch.cat([tp, sp], dim=-1)


class HybridSurpriseGDNBridge(nn.Module):
    """Step-0 Exact `PerceptualMemoryEncoder` Graft + Zero-Init `8x8` Detail & `DeltaMem` Residual (`P2-M5`).

    Guarantees at initialization (`gamma_detail == 0`, `gamma_hybrid == 0`):
      1. `100%` exact `PerceptualMemoryEncoder` token manifold & `q_positions = 512..531` RoPE alignment,
      2. Immediate `k=3` cross-session interference immunity via `EpisodeBoundaryForgetGate` reset of both
         keyframe selection AND `PosEmb3D` local session frame coordinates,
      3. Non-zero Step-0 gradients into `gamma_hybrid`, `gamma_detail`, and `DeltaMem` grounded `<x,y>`/count heads!
    """

    def __init__(self, cfg: Pi05MemConfig, pos_dim: int = 768) -> None:
        super().__init__()
        self.cfg = cfg
        self.vis_dim = int(cfg.vis_dim)
        self.d_tok = int(cfg.mem_token_dim)
        self.d_mem = int(cfg.d_mem)
        self.canvas_tokens = int(cfg.hybrid_canvas_tokens)
        self.max_keyframes = min(int(cfg.hybrid_num_keyframes), self.canvas_tokens // 16)

        self.pos_emb_3d = TorchPosEmb3D(dim=pos_dim)

        # Pathway 1: High-res `8x8` (`4` sub-cells per `4x4` patch) spatial detail residual (`gamma_detail = 0`)
        d_sub = max(64, self.vis_dim // 8)
        self.detail_norm = nn.RMSNorm(self.vis_dim)
        self.detail_down = nn.Linear(self.vis_dim, d_sub, bias=False)
        self.detail_up = nn.Linear(4 * d_sub, self.vis_dim, bias=False)
        self.detail_out_norm = nn.RMSNorm(self.vis_dim)
        self.gamma_detail = nn.Parameter(torch.zeros((1, 1, self.vis_dim), dtype=torch.float32))

        # Pathway 2: Cross-attention + SwiGLU injecting `2L-MultiScale-GDN + Wrist + Self-<x,y>/Count` (`gamma_hybrid = 0`)
        self.ctx_to_tok = nn.Linear(self.d_mem, self.d_tok, bias=False)
        self.q_norm = nn.RMSNorm(self.d_tok)
        self.kv_norm = nn.RMSNorm(self.d_tok)
        self.cross_attn = MultiHeadAttn(self.d_tok, num_heads=int(cfg.hybrid_bridge_heads))
        self.ffn_norm = nn.RMSNorm(self.d_tok)
        self.ffn = SwiGLU(self.d_tok, self.d_tok * 2)
        self.out_norm = nn.RMSNorm(self.d_tok)
        self.gamma_hybrid = nn.Parameter(torch.zeros((1, 1, self.d_tok), dtype=torch.float32))

    def select_hybrid_keyframes(
        self,
        write_idx: torch.Tensor,                # (W,) int64
        is_demo: torch.Tensor,                  # (W,) bool
        salience_w: torch.Tensor,               # (W,) fp32
        b_reset_w: Optional[torch.Tensor],      # Optional (W,) fp32
        *,
        cutoff_excl_frame: int,
        query_step_idx: int,
        anchor_ratio_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return `(sel_u_pos, local_frame_steps, sess_start_step)` strictly within current session (`#259` Fixed!)."""
        dev = write_idx.device
        w_len = int(write_idx.numel())
        if w_len == 0:
            empty = torch.empty((0,), device=dev, dtype=torch.long)
            return empty, empty, 0

        u_end = int(torch.searchsorted(write_idx, int(cutoff_excl_frame), right=False).item())
        u_query = int(torch.searchsorted(write_idx, int(query_step_idx), right=True).item())

        u_post_start = 0
        if b_reset_w is not None and u_query > 0:
            br_hits = torch.nonzero(
                b_reset_w[:u_query] > float(self.cfg.boundary_flush_threshold), as_tuple=False
            ).flatten()
            if br_hits.numel() > 0:
                u_post_start = int(br_hits[-1].item())

        sess_start_step = int(write_idx[u_post_start].item())
        if u_end <= u_post_start:
            sel_u = torch.tensor([u_post_start], device=dev, dtype=torch.long)
            loc_steps = torch.zeros((1,), device=dev, dtype=torch.long)
            return sel_u, loc_steps, sess_start_step

        cand = torch.arange(u_post_start, u_end, device=dev, dtype=torch.long)
        n_cand = int(cand.numel())
        k_max = max(1, self.max_keyframes)

        if n_cand <= k_max:
            sel_u = cand
        else:
            demo_rel = torch.nonzero(is_demo[cand], as_tuple=False).flatten()
            has_demo = demo_rel.numel() > 0
            eff_anc_ratio = (
                float(self.cfg.hybrid_anchor_ratio)
                if anchor_ratio_override is None
                else float(anchor_ratio_override)
            )
            k_anc = max(2, min(k_max, int(round(k_max * eff_anc_ratio))))
            n_anc_pts = (k_max - 1) if (has_demo and k_anc >= k_max and k_max >= 3) else k_anc
            # `#265`+`#266` Fix: space anchors uniformly in physical timestamp `t` (eliminates dense_prefix bunching)!
            cand_t = (write_idx[cand] - sess_start_step).clamp(min=0)
            t_max = int(cand_t[-1].item())
            target_t = (torch.arange(n_anc_pts, device=dev, dtype=torch.long) * t_max) // max(1, n_anc_pts - 1)
            anc_rel = torch.searchsorted(cand_t, target_t, right=False).clamp(0, n_cand - 1).unique()

            # Preserve final conditioning-demo frame WITHOUT ever dropping index 0 or `n_cand - 1` (`#259` Bug #3)!
            if has_demo:
                last_d = demo_rel[-1:]
                anc_rel = torch.cat([anc_rel, last_d], dim=0).unique()
                while int(anc_rel.numel()) > k_max:
                    mid = max(1, min(int(anc_rel.numel()) - 2, int(anc_rel.numel()) // 2))
                    anc_rel = torch.cat([anc_rel[:mid], anc_rel[mid + 1 :]], dim=0)

            rem = k_max - int(anc_rel.numel())
            if rem > 0:
                scores = salience_w[cand].float().clone()
                scores[anc_rel] = -1e9
                picked_rel: List[int] = []
                for _ in range(rem):
                    best_i = int(scores.argmax().item())
                    if float(scores[best_i].item()) <= -1e8:
                        break
                    picked_rel.append(best_i)
                    scores[best_i] = -1e9
                    if best_i > 0 and float(scores[best_i - 1].item()) > -1e8:
                        scores[best_i - 1] = scores[best_i - 1] - 0.5
                    if best_i + 1 < n_cand and float(scores[best_i + 1].item()) > -1e8:
                        scores[best_i + 1] = scores[best_i + 1] - 0.5
                if picked_rel:
                    sal_rel = torch.tensor(picked_rel, device=dev, dtype=torch.long)
                    all_rel = torch.sort(torch.cat([anc_rel, sal_rel], dim=0)).values
                else:
                    all_rel = torch.sort(anc_rel).values
            else:
                all_rel = torch.sort(anc_rel).values
            sel_u = cand[all_rel]

        loc_steps = (write_idx[sel_u] - sess_start_step).clamp(min=0)
        return sel_u, loc_steps, sess_start_step

    def encode_perceptual_canvas(
        self,
        f_8x8_sel: torch.Tensor,                # (L_sel, 64, vis_dim)
        local_frame_steps: torch.Tensor,        # (L_sel,) int64
        perceptual_mem_encoder: nn.Module,
        *,
        fallback_curr_front: Optional[torch.Tensor] = None,  # (P_curr, vis_dim)
        query_local_step: Optional[int] = None,              # Exact current session step `t`
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode selected keyframes into `[canvas_tokens, d_tok]` + `bool[canvas_tokens]` mask."""
        dev = f_8x8_sel.device
        if int(f_8x8_sel.shape[0]) == 0:
            assert fallback_curr_front is not None
            f_8x8_sel = pool_spatial_tokens_to_square(fallback_curr_front.unsqueeze(0), 64)
            local_frame_steps = torch.tensor(
                [max(0, int(query_local_step)) if query_local_step is not None else 0],
                device=dev,
                dtype=torch.long,
            )
        elif (
            fallback_curr_front is not None
            and query_local_step is not None
            and int(query_local_step) > int(local_frame_steps[-1].item())
        ):
            # Guarantee exact current-step `t` endpoint even when `t >= 32` falls on an odd stride step!
            curr_8x8 = pool_spatial_tokens_to_square(fallback_curr_front.unsqueeze(0), 64).to(f_8x8_sel.dtype)
            q_pos_t = torch.tensor([int(query_local_step)], device=dev, dtype=torch.long)
            if int(f_8x8_sel.shape[0]) < self.max_keyframes:
                f_8x8_sel = torch.cat([f_8x8_sel, curr_8x8], dim=0)
                local_frame_steps = torch.cat([local_frame_steps, q_pos_t], dim=0)
            else:
                f_8x8_sel = torch.cat([f_8x8_sel[:-1], curr_8x8], dim=0)
                local_frame_steps = torch.cat([local_frame_steps[:-1], q_pos_t], dim=0)

        l_sel = min(int(f_8x8_sel.shape[0]), self.canvas_tokens // 16)
        f_8x8 = f_8x8_sel[:l_sel]
        loc_pos = local_frame_steps[:l_sel]

        f_4x4 = pool_spatial_tokens_to_square(f_8x8, 16)  # (L_sel, 16, vis_dim)
        dt_d = self.detail_down.weight.dtype
        f_sub = (
            f_8x8.to(dt_d)
            .reshape(l_sel, 4, 2, 4, 2, self.vis_dim)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(l_sel, 16, 4, self.vis_dim)
        )
        det_h = F.silu(self.detail_down(self.detail_norm(f_sub))).reshape(l_sel, 16, -1)
        det_unit = self.detail_out_norm(self.detail_up(det_h)).to(f_4x4.dtype)
        f4_rms = f_4x4.detach().float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().to(f_4x4.dtype)
        g_det = torch.tanh(self.gamma_detail).to(f_4x4.dtype)
        img_in = f_4x4 + g_det * (det_unit * f4_rms)

        pos_in = self.pos_emb_3d(loc_pos)  # (L_sel, 16, 768)
        enc_tok = perceptual_mem_encoder(
            img_in.reshape(l_sel * 16, self.vis_dim),
            pos_in.reshape(l_sel * 16, -1),
            None,
        )  # (L_sel * 16, d_tok)

        n_act = int(enc_tok.shape[0])
        canvas = torch.zeros((self.canvas_tokens, self.d_tok), device=dev, dtype=enc_tok.dtype)
        mask = torch.zeros((self.canvas_tokens,), device=dev, dtype=torch.bool)
        canvas[:n_act] = enc_tok
        mask[:n_act] = True
        return canvas, mask

    def forward(
        self,
        perc_canvas: torch.Tensor,      # (B, S_canvas, d_tok)
        perc_mask: torch.Tensor,        # (B, S_canvas) bool
        deltamem_tokens: torch.Tensor,  # (B, S_dm, d_tok)
        deltamem_mask: torch.Tensor,    # (B, S_dm) bool
        ctx_tokens: torch.Tensor,       # (B, L_ctx, d_mem)
    ) -> torch.Tensor:
        dt = perc_canvas.dtype
        ctx_hi = self.ctx_to_tok(ctx_tokens.to(self.ctx_to_tok.weight.dtype)).to(dt)
        ctx_m = torch.ones((ctx_hi.shape[0], ctx_hi.shape[1]), device=perc_canvas.device, dtype=torch.bool)
        kv = self.kv_norm(torch.cat([deltamem_tokens.to(dt), ctx_hi], dim=1))
        kv_mask = torch.cat([deltamem_mask, ctx_m], dim=1)

        q = self.q_norm(perc_canvas)
        # `#260` Fix: route FFN strictly through `d1` (zero `perc_canvas` FFN shortcut!) + RMS-matched unit residual!
        d1 = self.cross_attn(q, kv, kv_valid_mask=kv_mask)
        delta_unit = self.out_norm(d1 + self.ffn(self.ffn_norm(d1)))
        canvas_rms = perc_canvas.detach().float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().to(dt)

        gate = torch.tanh(self.gamma_hybrid).to(dt)
        fused = perc_canvas + gate * (delta_unit * canvas_rms)
        return torch.where(perc_mask.unsqueeze(-1), fused, torch.zeros_like(perc_canvas))


class GraftedMemoryRMSNorm(nn.Module):
    """Zero-shock `fp32`-gated residual `MemoryRMSNorm` for grafting `HistoryBlock` onto `symbolic_grounded_subgoal_pt` (`P2-M7A`).

    Preserves the exact `self.dense = nn.Linear(dim, 2 * dim, bias=True)` parameter names of `MemoryRMSNorm(adaptive=True)`
    so pretrained `perceptual_framesamp_modul` / `Hybrid` FiLM weights load `1:1`, while gating `(xf * scale + rms * shift)`
    with a learnable per-layer `fp32` scalar `tanh(self.graft_gate)`:
      - At `graft_gate == 0.0`, output equals `x` bit-for-bit (`0.0` Step-0 shock to `symbolic_grounded_subgoal_pt`!).
      - Because `(mem_attn, dense)` are pretrained (`delta != 0`), `d(out)/d(graft_gate) = delta != 0` on Step 1!
    """

    def __init__(self, dim: int = 1024, eps: float = 1e-6, init_gate: float = 0.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.adaptive = True
        self.residual_zero_init = True
        self.weight: Optional[nn.Parameter] = None
        self.dense = nn.Linear(self.dim, 2 * self.dim, bias=True)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)
        self.graft_gate = nn.Parameter(torch.full((1,), float(init_gate), dtype=torch.float32))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is None:
            return x
        dtype = x.dtype
        w_dtype = dtype if self.dense.weight.dtype == torch.bfloat16 else torch.float32
        modulation = F.linear(
            cond.to(w_dtype),
            self.dense.weight.to(w_dtype),
            self.dense.bias.to(w_dtype) if self.dense.bias is not None else None,
        )
        scale, shift = modulation.chunk(2, dim=-1)
        xf = x.float()
        rms = torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + self.eps)
        delta = xf * scale.float() + rms * shift.float()
        g = torch.tanh(self.graft_gate.float())
        return (xf + g * delta).to(dtype)


class VLMPrefixSurpriseSelector(nn.Module):
    """`P2-M8` (`#329` & `#331`) Causal `GDN-Surprise` Reveal Keyframe Selector for `PaliGemma-2B` VLM Prefix.

    Selects `K_vlm = 4` chronological reveal/highlight/demo frames per session (`start_pos`-isolated via
    `EpisodeBoundaryForgetGate` for `0.00 pp` `k=3` cross-session interference!) with temporal NMS so
    multi-object reveals (`VideoUnmask*`, `VideoPlaceOrder`, `ButtonUnmask*`, `PickHighlight`) capture
    distinct physical events rather than adjacent frames of a single peak, adds a zero-init `[K_vlm, 1, D_vlm]`
    chronological frame-order embedding `vlm_hist_slot_emb`, and appends `[K_vlm * P_vlm, D_vlm]` native
    `SigLIP` tokens at the end of `PaliGemma-2B`'s bidirectional (`prefix_ar_mask == 0`) prefix.
    """

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_keyframes = max(1, int(cfg.vlm_prefix_surprise_keyframes))
        self.tokens_per_frame = max(1, int(cfg.vlm_prefix_tokens_per_frame))
        self.vlm_dim = int(cfg.vlm_width)
        self.demo_quota = max(0, int(cfg.vlm_prefix_demo_quota))
        self.early_quota = max(0, int(cfg.vlm_prefix_early_quota))
        self.flush_thresh = float(cfg.boundary_flush_threshold)
        self.vlm_hist_slot_emb = nn.Parameter(torch.zeros(self.num_keyframes, 1, self.vlm_dim))

    @staticmethod
    def _greedy_nms_topk(
        cand_indices: List[int],
        scores_by_idx: Dict[int, float],
        k: int,
        min_sep: int,
    ) -> List[int]:
        if k <= 0 or not cand_indices:
            return []
        ranked = sorted(cand_indices, key=lambda idx: (-scores_by_idx[idx], idx))
        chosen: List[int] = []
        eff_sep = max(1, int(min_sep))
        for idx in ranked:
            if all(abs(idx - prev) >= eff_sep for prev in chosen):
                chosen.append(idx)
                if len(chosen) >= k:
                    return chosen
        chosen_set = set(chosen)
        for idx in ranked:
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)
                if len(chosen) >= k:
                    break
        return chosen

    @torch.no_grad()
    def select_vlm_surprise_indices(
        self,
        w_steps: torch.Tensor,                      # (W,) int64 written frame indices
        w_is_demo: torch.Tensor,                    # (W,) bool
        salience_w: torch.Tensor,                   # (W,) fp32 full surprise + grip salience
        pure_vis_salience_w: torch.Tensor,          # (W,) fp32 pure visual surprise (no grip / no demo boost)
        boundary_reset_w: Optional[torch.Tensor],   # Optional (W,) fp32 EpisodeBoundaryForgetGate
        *,
        cutoff_excl_frame: int,
    ) -> torch.Tensor:
        dev = w_steps.device
        w_tot = int(w_steps.numel())
        if w_tot == 0:
            return torch.empty((0,), device=dev, dtype=torch.long)

        u_cut = int(torch.searchsorted(w_steps, int(cutoff_excl_frame), right=False).item())
        if u_cut <= 0:
            return torch.empty((0,), device=dev, dtype=torch.long)

        start_pos = 0
        if boundary_reset_w is not None:
            resets = torch.nonzero(boundary_reset_w[:u_cut] > self.flush_thresh, as_tuple=False).flatten()
            if resets.numel() > 0:
                start_pos = int(resets[-1].item())

        n_sess = u_cut - start_pos
        if n_sess <= 0:
            return torch.empty((0,), device=dev, dtype=torch.long)
        if n_sess <= self.num_keyframes:
            return torch.arange(start_pos, u_cut, device=dev, dtype=torch.long)

        d_list = w_is_demo[start_pos:u_cut].detach().cpu().tolist()
        sal_list = salience_w[start_pos:u_cut].detach().float().cpu().tolist()
        pvis_list = pure_vis_salience_w[start_pos:u_cut].detach().float().cpu().tolist()

        sal_map = {start_pos + i: float(sal_list[i]) for i in range(n_sess)}
        pvis_map = {start_pos + i: float(pvis_list[i]) for i in range(n_sess)}
        demo_cands = [start_pos + i for i in range(n_sess) if d_list[i]]
        exec_cands = [start_pos + i for i in range(n_sess) if not d_list[i]]

        k_tot = self.num_keyframes
        chosen: List[int] = []

        if demo_cands:
            q_demo = k_tot if not exec_cands else min(self.demo_quota, k_tot, len(demo_cands))
            sep_demo = max(2, len(demo_cands) // (q_demo + 1))
            chosen.extend(self._greedy_nms_topk(demo_cands, pvis_map, q_demo, sep_demo))
            rem = k_tot - len(chosen)
            if rem > 0 and exec_cands:
                sep_exec = max(2, len(exec_cands) // (rem + 1))
                chosen.extend(self._greedy_nms_topk(exec_cands, sal_map, rem, sep_exec))
            rem = k_tot - len(chosen)
            if rem > 0:
                chosen_set = set(chosen)
                leftover = [i for i in (demo_cands + exec_cands) if i not in chosen_set]
                chosen.extend(self._greedy_nms_topk(leftover, pvis_map, rem, 1))
        else:
            early_len = max(4, min(24, (len(exec_cands) * 2 + 4) // 5))
            early_cands = exec_cands[:early_len]
            q_early = min(self.early_quota, k_tot, len(early_cands))
            sep_early = max(2, len(early_cands) // (q_early + 1))
            chosen.extend(self._greedy_nms_topk(early_cands, pvis_map, q_early, sep_early))
            rem = k_tot - len(chosen)
            if rem > 0:
                chosen_set = set(chosen)
                rest_exec = [i for i in exec_cands if i not in chosen_set]
                sep_rest = max(2, len(exec_cands) // (k_tot + 1))
                chosen.extend(self._greedy_nms_topk(rest_exec, sal_map, rem, sep_rest))

        chosen_sorted = sorted(chosen[:k_tot])
        return torch.tensor(chosen_sorted, device=dev, dtype=torch.long)

    def encode_vlm_surprise_tokens(
        self,
        f_sel_hist: torch.Tensor,  # (K_sel, P_hist, D_vlm), 0 <= K_sel <= K_vlm
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(vlm_tokens [K_vlm * P_vlm, D_vlm], vlm_mask [K_vlm * P_vlm] bool)`."""
        dev = f_sel_hist.device
        dt = f_sel_hist.dtype
        k_sel = min(int(f_sel_hist.shape[0]), self.num_keyframes)
        tot_tok = self.num_keyframes * self.tokens_per_frame
        out_tok = torch.zeros((self.num_keyframes, self.tokens_per_frame, self.vlm_dim), device=dev, dtype=dt)
        out_mask = torch.zeros((self.num_keyframes, self.tokens_per_frame), device=dev, dtype=torch.bool)
        if k_sel > 0:
            pooled = pool_spatial_tokens_to_square(f_sel_hist[:k_sel], self.tokens_per_frame)
            slot_bias = self.vlm_hist_slot_emb[:k_sel].to(device=dev, dtype=dt)
            out_tok[:k_sel] = pooled + slot_bias
            out_mask[:k_sel] = True
        return out_tok.reshape(tot_tok, self.vlm_dim), out_mask.reshape(tot_tok)


class ContinuousVLMGroundingBridge(nn.Module):
    """`P2-M11` (`#395(2)`) Zero-ASCII Continuous VLM-Grounded Spatial Bridge in `HistoryBlock` KV (`max_token_len = 64`!).

    Eliminates the `"<row_y, col_x>"` ASCII discretization & `64 -> 128` RoPE-shift bottleneck on
    `perceptual_framesamp_modul_pt` by projecting frozen `P2-M9-Hyb-v3-HR16`'s continuous grounding packet:
      - `64` spatial `8x8` scene+heatmap tokens (`[front_vlm_8x8 || avg_pool(p1,p2) || max_pool(p1,p2) || 2D_grid_xy]`),
      - `7` soft-ROI & VLM context visual tokens (`[p1_curr@F, p1_latched@F, p2_curr@F, p2_latched@F, lang, gdn, rev]`),
      - `9` stage/color/count/continuous-`(x, y)` intent tokens,
    into `80` `1024-D` grounding tokens `Z_vlm_ground`, fused into `perc_canvas` (`512t`) via `fp32` zero-init
    RMS-matched cross-attention (`gamma_vlm`) + tail-slot KV injection (`gamma_vlm_direct`).
    On the 10 base tasks (`active_mask == False`), returns `perc_canvas` untouched (`0.000000e+00` `v3_s250` identity!).
    """

    def __init__(self, cfg: Pi05MemConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vlm_dim = int(cfg.vlm_width)
        self.d_tok = int(cfg.mem_token_dim)
        self.jitter_cells = float(cfg.cont_vlm_jitter_cells)

        centers_8 = ((torch.arange(8, dtype=torch.float32) + 0.5) / 8.0) * 2.0 - 1.0
        yy8, xx8 = torch.meshgrid(centers_8, centers_8, indexing="ij")
        xy8 = torch.stack([xx8.reshape(-1), yy8.reshape(-1)], dim=-1)
        pos8 = torch.cat(
            [
                xy8,
                build_1d_sincos_embed(xy8[:, 0] * 4.0, 8, max_period=16.0),
                build_1d_sincos_embed(xy8[:, 1] * 4.0, 8, max_period=16.0),
            ],
            dim=-1,
        )  # (64, 18) smooth monotonic 2D grid basis
        self.register_buffer("grid_pos_8x8", pos8, persistent=False)

        self.feat_drop = nn.Dropout(0.10)
        self.scene_proj = nn.Sequential(
            nn.RMSNorm(self.vlm_dim + 8 + 18),
            nn.Linear(self.vlm_dim + 8 + 18, self.d_tok),
        )
        self.roi_proj = nn.Sequential(
            nn.RMSNorm(self.vlm_dim),
            nn.Linear(self.vlm_dim, self.d_tok),
        )
        self.roi_role_emb = nn.Parameter(torch.zeros(1, 7, self.d_tok))
        # `sem_probs` (48) + `xy_coords` (8) + `8 * 8` low-freq smooth sincos = 120-D -> 9 * d_tok
        self.intent_mlp = nn.Sequential(
            nn.Linear(48 + 8 + 64, self.d_tok * 2),
            nn.RMSNorm(self.d_tok * 2),
            nn.SiLU(),
            nn.Linear(self.d_tok * 2, 9 * self.d_tok),
        )

        self.q_norm = nn.RMSNorm(self.d_tok)
        self.kv_norm = nn.RMSNorm(self.d_tok)
        self.cross_attn = MultiHeadAttn(self.d_tok, num_heads=int(cfg.hybrid_bridge_heads))
        self.ffn_norm = nn.RMSNorm(self.d_tok)
        self.ffn = SwiGLU(self.d_tok, self.d_tok * 2)
        self.out_norm = nn.RMSNorm(self.d_tok)
        self.direct_norm = nn.RMSNorm(self.d_tok)

        self.gamma_vlm = nn.Parameter(torch.zeros((1, 1, self.d_tok), dtype=torch.float32))
        self.gamma_vlm_direct = nn.Parameter(torch.zeros((1, 1, self.d_tok), dtype=torch.float32))

    def _jitter_heatmaps_and_xy(
        self,
        hm_maps: torch.Tensor,    # (B, 4, 256) fp32
        xy_coords: torch.Tensor,  # (B, 8) fp32 in [-1, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not (self.training and self.jitter_cells > 0.0):
            return hm_maps, xy_coords
        bsz = int(hm_maps.shape[0])
        dev = hm_maps.device
        max_shift = float(self.jitter_cells) * (2.0 / 16.0)
        shift_xy = (torch.rand((bsz, 2), device=dev, dtype=torch.float32) * 2.0 - 1.0) * max_shift
        # Shift 16x16 probability maps by `shift_xy` via inverse sampling grid (`x_src = x_dst - shift_xy`)
        c16 = ((torch.arange(16, device=dev, dtype=torch.float32) + 0.5) / 16.0) * 2.0 - 1.0
        yy, xx = torch.meshgrid(c16, c16, indexing="ij")
        base_g = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(bsz, -1, -1, -1)
        src_g = base_g - shift_xy[:, None, None, :]
        hm_4d = hm_maps.view(bsz, 4, 16, 16)
        hm_sh = F.grid_sample(hm_4d, src_g, mode="bilinear", padding_mode="zeros", align_corners=False)
        hm_flat = hm_sh.view(bsz, 4, 256)
        hm_sum = hm_flat.sum(dim=-1, keepdim=True)
        hm_out = torch.where(hm_sum > 1e-5, hm_flat / hm_sum.clamp(min=1e-5), hm_maps)
        xy_sh = (xy_coords.view(bsz, 4, 2) + shift_xy.unsqueeze(1)).clamp(-1.0, 1.0).view(bsz, 8)
        return hm_out, xy_sh

    def encode_grounding_tokens(self, cont_ground: Dict[str, torch.Tensor]) -> torch.Tensor:
        dt = self.scene_proj[1].weight.dtype  # type: ignore[index]
        dev = self.scene_proj[1].weight.device  # type: ignore[index]
        f8 = self.feat_drop(cont_ground["front_vlm_8x8"].to(device=dev, dtype=dt))
        rois = self.feat_drop(cont_ground["roi_vecs"].to(device=dev, dtype=dt))
        hm_raw = cont_ground["hm_maps"].to(device=dev, dtype=torch.float32)
        xy_raw = cont_ground["xy_coords"].to(device=dev, dtype=torch.float32)
        sem_p = cont_ground["sem_probs"].to(device=dev, dtype=torch.float32)

        hm_j, xy_j = self._jitter_heatmaps_and_xy(hm_raw, xy_raw)
        bsz = int(f8.shape[0])
        hm_4d = hm_j.view(bsz, 4, 16, 16)
        hm_avg = (F.avg_pool2d(hm_4d, kernel_size=2, stride=2) * 4.0).view(bsz, 4, 64).transpose(1, 2)
        hm_max = (F.max_pool2d(hm_4d, kernel_size=2, stride=2) * 4.0).view(bsz, 4, 64).transpose(1, 2)
        pos8 = self.grid_pos_8x8.to(device=dev, dtype=dt).unsqueeze(0).expand(bsz, -1, -1)
        scene_in = torch.cat([f8, hm_avg.to(dt), hm_max.to(dt), pos8], dim=-1)
        scene_64 = self.scene_proj(scene_in)

        roi_7 = self.roi_proj(rois) + self.roi_role_emb.to(device=dev, dtype=dt)

        xy_sin_parts = [build_1d_sincos_embed(xy_j[:, ci] * 4.0, 8, max_period=16.0) for ci in range(8)]
        intent_in = torch.cat([sem_p, xy_j, *xy_sin_parts], dim=-1).to(dt)
        intent_9 = self.intent_mlp(intent_in).view(bsz, 9, self.d_tok)
        return torch.cat([scene_64, roi_7, intent_9], dim=1)  # (B, 80, d_tok)

    def forward(
        self,
        perc_canvas: torch.Tensor,                  # (B, L_c, d_tok)
        perc_mask: torch.Tensor,                    # (B, L_c) bool
        cont_ground: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if cont_ground is None:
            return perc_canvas
        act_m = cont_ground.get("active_mask")
        if act_m is not None and not bool(act_m.any().item()):
            return perc_canvas

        dt = perc_canvas.dtype
        z_vlm = self.encode_grounding_tokens(cont_ground).to(dt)  # (B, 80, d_tok)
        q = self.q_norm(perc_canvas.to(self.q_norm.weight.dtype))
        kv = self.kv_norm(z_vlm.to(self.kv_norm.weight.dtype))
        d1 = self.cross_attn(q, kv)
        delta_unit = self.out_norm(d1 + self.ffn(self.ffn_norm(d1))).to(dt)
        canvas_rms = perc_canvas.detach().float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().to(dt)

        g_ca = torch.tanh(self.gamma_vlm).to(dt)
        fused = perc_canvas + g_ca * (delta_unit * canvas_rms)

        k_tail = min(int(z_vlm.shape[1]), int(fused.shape[1]))
        if k_tail > 0:
            z_unit = self.direct_norm(z_vlm[:, -k_tail:, :].to(self.direct_norm.weight.dtype)).to(dt)
            tail_rms = fused[:, -k_tail:, :].detach().float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt().to(dt)
            g_dir = torch.tanh(self.gamma_vlm_direct).to(dt)
            fused_tail = fused[:, -k_tail:, :] + g_dir * (z_unit * tail_rms)
            fused = torch.cat([fused[:, :-k_tail, :], fused_tail], dim=1)

        fused = torch.where(perc_mask.unsqueeze(-1), fused, torch.zeros_like(perc_canvas))
        if act_m is not None:
            act_b = act_m.to(device=perc_canvas.device, dtype=torch.bool).view(-1, 1, 1)
            fused = torch.where(act_b, fused, perc_canvas)
        return fused
