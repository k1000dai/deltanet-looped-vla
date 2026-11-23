#!/usr/bin/env python3
"""Mechanistic State & Multi-Hop Loop Probe Suite for Pretrained `SmolVLA-DeltaLoop` (Cycle 3B / `U6`).

Evaluates frozen representations, `512 KB` 2-layer Negative-Eigenvalue GDN memory states `(S1_past, S2_past)`,
`U4` auxiliary grounding predictions, and `CRNHaltingGate` loop allocation of trained `SmolVLADeltaLoopPolicy`
checkpoints (`smolvla_m0_base`, `smolvla_m1_rdvla_loop4`, `smolvla_m2_framesamp`, `smolvla_m5_gdn2l_l1`,
`smolvla_m7_deltaloop`) on cached `robomme_smolvla_v1/{TaskName}.pt` shards with **100% identical JSON schema**
to `analysis/probe_robomme_state.py` (`mechanistic_probes.json`) so `analysis/plot_robomme.py` plots Cycle 3B
`SmolVLA` out-of-the-box:

  - **Probe 1 (Hop-by-Hop `<x, y>` Target Dereferencing across `l = 1..4`):**
    Evaluates **BOTH** zero-shot `U4` `model.predict_grounding_xy(ctx_tokens, C_t^(l), mem_mask=...)`
    **AND** episode-disjoint standardized Woodbury dual Ridge + 2-layer MLP probes on `C_t^(l)` and
    fused `[ctx_pool, C_t^(l)]` (plus `ctx_pool` Markov baseline).

  - **Probe 2 (Frozen `512 KB` `(S1_past, S2_past)` Count & Post-Swap Retention vs Horizon):**
    Decodes cumulative pick/stage count (`Early <32`, `Mid 32..95`, `Late >=96` steps) and post-occlusion
    `3x3` target spatial bin + continuous `<x, y>` + action stage from frozen `512 KB` `[S1, S2, occ1, occ2]`
    (`133,120` floats via exact Woodbury dual Ridge) vs `24 KB` `FrameSamp-32` buffer vs `0 B` `ctx_tokens`.

  - **Probe 3 (`k in {0, 1, 3}` Un-Reset Cross-Session Interference & Clutter-Triggered Halting):**
    Streams `k in {0, 1, 3}` prior distractor episodes through persistent memory WITHOUT reset, measuring
    Coda cosine invariance `cos(C_t(k), C_t(0))`, `<x, y>` localization drift `+\Delta px`, state retention
    drop, and `CRNHaltingGate` clutter-triggered loop allocation `E[L_t | k]` alongside event-triggered
    `E[L_t | post_demo_onset / subgoal_transition / steady_execution]`.

Fast Backboneless Probing:
  Uses cached `(64, 960)` VLM vision tokens + `embed_language_tokens` directly without invoking the
  16-layer VLM prefix KV prefill or 16-layer Action Expert ODE loop, running in seconds per checkpoint!

CPU Smoke-Test & Miyabi CUDA Usage:
  CUDA_VISIBLE_DEVICES="" python analysis/probe_smolvla_state.py --smoke-test
  python analysis/probe_smolvla_state.py \\
      --device cuda \\
      --checkpoints checkpoints/robomme/cycle3b_smolvla/smolvla_m0_base.pt \\
                    checkpoints/robomme/cycle3b_smolvla/smolvla_m1_rdvla_loop4.pt \\
                    checkpoints/robomme/cycle3b_smolvla/smolvla_m2_framesamp.pt \\
                    checkpoints/robomme/cycle3b_smolvla/smolvla_m5_gdn2l_l1.pt \\
                    checkpoints/robomme/cycle3b_smolvla/smolvla_m7_deltaloop.pt \\
      --shard-dir /work/SSR/share/data/robomme_smolvla_v1/val \\
      --output-json results/robomme/cycle3b/mechanistic_probes.json
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.probe_robomme_state import (
    StandardizedDualRidgeRegressor,
    StandardizedRidgeClassifier,
    _concat_ep_records,
    compute_cls_metrics,
    compute_xy_metrics,
    fit_and_predict_mlp_xy,
    print_probe_summary_table,
    xy_norm_to_3x3_bin,
)
from analysis.subgoal_grounding import (
    extract_subgoal_metadata,
    parse_grounded_subgoals,
)
from robomme_eval.interface import MME_VLA_NORM_STATS, ROBOMME_TASKS
from src.models.deltaloop_vla import run_dual_backend_gdn
from src.models.smolvla_deltaloop import (
    ALL_SMOLVLA_ARCHS,
    MeanStdDeltaNormalizer,
    SmolVLADeltaLoopConfig,
    SmolVLADeltaLoopPolicy,
    cast_smolvla_precision,
    pool_vlm_tokens_8x8_to_4x4,
)


# =============================================================================
# 1. Checkpoint Loader & Causal `robomme_smolvla_v1` Session Streamer
# =============================================================================


def load_smolvla_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[SmolVLADeltaLoopPolicy, MeanStdDeltaNormalizer, Dict[str, Any]]:
    """Load any `SmolVLADeltaLoopPolicy` `.pt` checkpoint onto `device` (`float32` on CPU, `bf16` on CUDA)."""
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw_cfg = dict(payload["model_config"])
    valid_fields = {f.name for f in dataclasses.fields(SmolVLADeltaLoopConfig)}
    filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_fields}
    cfg = SmolVLADeltaLoopConfig.from_dict(filtered_cfg)

    model = SmolVLADeltaLoopPolicy(cfg)
    state_dict = payload["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    cast_smolvla_precision(model, device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)

    norm_stats = payload.get("norm_stats", MME_VLA_NORM_STATS)
    normalizer = MeanStdDeltaNormalizer(stats=norm_stats)

    step_val = payload.get("step", None)
    if step_val is None and isinstance(payload.get("metrics"), dict):
        step_val = payload["metrics"].get("steps", None)

    meta = {
        "ckpt_path": str(ckpt_path),
        "arch": cfg.arch,
        "step": step_val,
        "state_bytes": model.state_size_bytes(),
        "max_loops": cfg.max_loops if model.is_multi_depth else 1,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, normalizer, meta


class SmolVLAShardSessionStreamer:
    """Streams cached `robomme_smolvla_v1/{TaskName}.pt` episodes through `SmolVLADeltaLoopPolicy`.

    Mirrors `SmolVLARoboMMEWrapper` (`src/models/smolvla_policy_wrapper.py`) step-for-step:
      - Exact A3 Video-Demo Cutoff (`cutoff_excl = max(exec_start_idx, step_idx - 15)`),
      - Exact `U5` adaptive demo striding (`long_demo_stride=8` when `exec_start > adaptive_demo_stride_threshold=256`,
        else `write_stride=4`; execution anchored at `exec_start` with `write_stride=4`),
      - Exact `U5` per-written-frame visual motion RMS $\\beta$-gating (`_compute_next_motion_rms`),
      - Exact `U2` `obs_horizon2` (`prev_idx = max(exec_start, t_step - 1)`),
      - Clean (`k=0`) and un-reset multi-episode distractor streaming (`k in {1, 3}`) matching `scripts/eval_smolvla.py`.
    """

    def __init__(
        self,
        model: SmolVLADeltaLoopPolicy,
        normalizer: MeanStdDeltaNormalizer,
        device: torch.device,
        *,
        exec_horizon: Optional[int] = None,
        force_pytorch_gdn: bool = True,
    ) -> None:
        self.model = model
        self.cfg = model.cfg
        self.normalizer = normalizer
        self.device = device
        self.dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
        self.exec_horizon = int(exec_horizon if exec_horizon is not None else self.cfg.exec_horizon)
        self.force_pytorch_gdn = bool(force_pytorch_gdn or device.type == "cpu")

        self._frame_cursor: int = 0
        self._demo_frames_count: int = 0
        self._pending_strided: List[Tuple[int, torch.Tensor, bool, torch.Tensor]] = []
        self._conv_ring: List[torch.Tensor] = []
        self._prev_written_front_tok: Optional[torch.Tensor] = None

        self._S1: Optional[torch.Tensor] = None
        self._occ1: Optional[torch.Tensor] = None
        self._S2: Optional[torch.Tensor] = None
        self._occ2: Optional[torch.Tensor] = None
        self._framesamp_committed: List[torch.Tensor] = []
        self._framesamp_demo_count: int = 0

        self.reset_session()

    def _clear_intra_episode_short_caches(self) -> None:
        """Clear ONLY short-term intra-episode caches while preserving persistent recurrent memory."""
        self._pending_strided.clear()
        self._conv_ring.clear()
        self._prev_written_front_tok = None

    def reset_session(self) -> None:
        self._frame_cursor = 0
        self._demo_frames_count = 0
        self._clear_intra_episode_short_caches()
        self._framesamp_committed.clear()
        self._framesamp_demo_count = 0

        if self.model.uses_gdn:
            H = self.cfg.gdn_heads
            K = self.cfg.gdn_d_k
            Vh = self.cfg.gdn_d_v // 2
            self._S1 = torch.zeros((1, H, Vh, K), device=self.device, dtype=torch.float32)
            self._occ1 = torch.zeros((1, H, K), device=self.device, dtype=torch.float32)
            self._S2 = torch.zeros((1, H, Vh, K), device=self.device, dtype=torch.float32)
            self._occ2 = torch.zeros((1, H, K), device=self.device, dtype=torch.float32)
        else:
            self._S1 = self._occ1 = self._S2 = self._occ2 = None

    def _stream_slot_conv_single(self, raw_slot_nw_d: torch.Tensor) -> torch.Tensor:
        resampler = self.model.slot_resampler
        assert resampler is not None
        dt = resampler.slot_queries.dtype
        x_cur = raw_slot_nw_d.to(dt)
        Nw, D = x_cur.shape

        if resampler.temporal_conv is not None and resampler.conv_width > 1:
            cw = resampler.conv_width
            pad_needed = (cw - 1) - len(self._conv_ring)
            hist_list: List[torch.Tensor] = []
            if pad_needed > 0:
                zero_f = torch.zeros((Nw, D), device=self.device, dtype=dt)
                hist_list.extend([zero_f] * pad_needed)
            hist_list.extend(self._conv_ring[-(cw - 1) :])
            hist_list.append(x_cur)
            win = torch.stack(hist_list, dim=0).permute(1, 2, 0).contiguous()
            y = F.silu(resampler.temporal_conv(win)).squeeze(-1)
            self._conv_ring.append(x_cur)
            if len(self._conv_ring) > cw - 1:
                self._conv_ring.pop(0)
        else:
            y = x_cur
        return resampler.out_norm(y).view(1, 1, Nw, D)

    def _compute_next_motion_rms(self, f_tok_single: torch.Tensor) -> torch.Tensor:
        """Compute causal per-written-frame visual motion RMS `(1,)` matching `SmolVLARoboMMEWrapper`."""
        f32 = f_tok_single.float().reshape(f_tok_single.shape[-2], f_tok_single.shape[-1])
        if self._prev_written_front_tok is None:
            m_val = torch.sqrt(torch.mean(f32 * f32) + 1e-8).view(1)
        else:
            diff = f32 - self._prev_written_front_tok
            m_val = torch.sqrt(torch.mean(diff * diff) + 1e-8).view(1)
        self._prev_written_front_tok = f32.detach().clone()
        return m_val

    def _queue_frame_range(
        self,
        front_vlm: torch.Tensor,
        wrist_vlm: torch.Tensor,
        state_norm: torch.Tensor,
        is_demo: torch.Tensor,
        *,
        is_demo_segment: bool = False,
    ) -> None:
        n = int(front_vlm.shape[0])
        if n == 0:
            return
        if not self.model.uses_history:
            self._frame_cursor += n
            if is_demo_segment:
                self._demo_frames_count += n
            return

        if is_demo_segment:
            total_demo = self._demo_frames_count + n
            stride = (
                int(self.cfg.long_demo_stride)
                if total_demo > int(self.cfg.adaptive_demo_stride_threshold)
                else int(self.cfg.write_stride)
            )
        else:
            stride = int(self.cfg.write_stride)

        sel_local: List[int] = []
        sel_global: List[int] = []
        for i in range(n):
            g_idx = self._frame_cursor + i
            if not is_demo_segment:
                is_strided = ((g_idx - self._demo_frames_count) % stride) == 0
            else:
                is_strided = (g_idx % stride) == 0
            if is_strided:
                sel_local.append(i)
                sel_global.append(g_idx)

        self._frame_cursor += n
        if is_demo_segment:
            self._demo_frames_count += n
        if not sel_local:
            return

        idx_t = torch.tensor(sel_local, dtype=torch.long)
        f_tok = front_vlm[idx_t].to(device=self.device, dtype=self.dtype)
        w_tok = wrist_vlm[idx_t].to(device=self.device, dtype=self.dtype)
        s_tok = state_norm[idx_t].to(device=self.device, dtype=torch.float32)
        d_tok = is_demo[idx_t].to(device=self.device, dtype=torch.long)
        step_t = torch.tensor(sel_global, device=self.device, dtype=torch.long)

        raw_slots = self.model.encode_history_frames_raw(f_tok, w_tok, s_tok, step_t, d_tok)
        d_list = is_demo[idx_t].tolist()
        for k_i, g_idx in enumerate(sel_global):
            m_val = self._compute_next_motion_rms(f_tok[k_i])
            self._pending_strided.append((g_idx, raw_slots[k_i], bool(d_list[k_i]), m_val))

    def _flush_eligible(self, cutoff_global_excl: int) -> None:
        if not self.model.uses_history:
            self._pending_strided.clear()
            return

        to_commit: List[Tuple[int, torch.Tensor, bool, torch.Tensor]] = []
        keep_p: List[Tuple[int, torch.Tensor, bool, torch.Tensor]] = []
        for item in self._pending_strided:
            if item[0] < cutoff_global_excl:
                to_commit.append(item)
            else:
                keep_p.append(item)
        self._pending_strided = keep_p

        if not to_commit:
            return

        if self.model.arch == "smolvla_m2_framesamp":
            for _, rs, is_d, _ in to_commit:
                cs = self._stream_slot_conv_single(rs)
                self._framesamp_committed.append(cs.view(-1))
                if is_d:
                    self._framesamp_demo_count += 1
            return

        assert (
            self.model.gdn1 is not None
            and self.model.gdn2 is not None
            and self.model.mid_norm is not None
            and self.model.mid_ffn is not None
        )
        z_frames = torch.cat([self._stream_slot_conv_single(rs) for _, rs, _, _ in to_commit], dim=1)
        m_frames = torch.stack([mv for _, _, _, mv in to_commit], dim=0).unsqueeze(0)
        W_seg, Nw = z_frames.shape[1], z_frames.shape[2]

        k1, v1, g1, b1, q1 = self.model.gdn1.project_frame_slots(z_frames, motion_feat=m_frames)
        assert q1 is not None
        o1, self._S1, self._occ1 = run_dual_backend_gdn(
            k=k1,
            v=v1,
            g_fp32=g1,
            beta_fp32=b1,
            q=q1,
            initial_state_fp32=self._S1,
            initial_occ_fp32=self._occ1,
            track_occ=True,
            force_pytorch=self.force_pytorch_gdn,
        )
        assert o1 is not None
        z_flat = z_frames.view(1, W_seg * Nw, -1)
        seq1 = self.model.gdn1.format_seq_readout(z_flat, o1)
        z_mid = z_flat.to(seq1.dtype) + seq1
        z_mid = z_mid + self.model.mid_ffn(self.model.mid_norm(z_mid))
        k2, v2, g2, b2, _ = self.model.gdn2.project_frame_slots(
            z_mid.view(1, W_seg, Nw, -1), motion_feat=m_frames
        )
        _, self._S2, self._occ2 = run_dual_backend_gdn(
            k=k2,
            v=v2,
            g_fp32=g2,
            beta_fp32=b2,
            initial_state_fp32=self._S2,
            initial_occ_fp32=self._occ2,
            track_occ=True,
            force_pytorch=self.force_pytorch_gdn,
        )

    def _current_mem_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.model.uses_history:
            return None
        if self.model.arch == "smolvla_m2_framesamp":
            B_sz = self.cfg.framesamp_buffer_size
            D = self.cfg.d_mem
            buf_dt = self.model.coda.proj.weight.dtype if self.model.coda is not None else self.dtype
            buf_tok = torch.zeros((1, B_sz, D), device=self.device, dtype=buf_dt)
            buf_mask = torch.zeros((1, B_sz), device=self.device, dtype=torch.bool)
            n_avail = len(self._framesamp_committed)
            if n_avail > 0:
                sel_idx, n_sel = self.model.select_framesamp_slot_indices(
                    n_avail, self._framesamp_demo_count, budget=B_sz, device=self.device
                )
                stacked = torch.stack(self._framesamp_committed, dim=0).to(buf_dt)
                buf_tok[0, :n_sel] = stacked[sel_idx]
                buf_mask[0, :n_sel] = True
            return {"buf_tokens": buf_tok, "buf_mask": buf_mask}

        assert self._S1 is not None and self._S2 is not None
        return {"S1": self._S1, "occ1": self._occ1, "S2": self._S2, "occ2": self._occ2}  # type: ignore[dict-item]

    @staticmethod
    def _flatten_state_vector(mem_state: Optional[Dict[str, torch.Tensor]], ctx_pool: torch.Tensor) -> torch.Tensor:
        """Extract exact flattened `(S1_past, S2_past, occ1, occ2)` (`532,480 B`) or `buf_tokens` or `ctx_pool`."""
        if mem_state is None:
            return ctx_pool.detach().cpu().float().reshape(-1)
        if "S1" in mem_state:
            parts = [
                mem_state["S1"].detach().cpu().float().reshape(-1),
                mem_state["S2"].detach().cpu().float().reshape(-1),
            ]
            if mem_state.get("occ1") is not None and mem_state.get("occ2") is not None:
                parts.append(mem_state["occ1"].detach().cpu().float().reshape(-1))
                parts.append(mem_state["occ2"].detach().cpu().float().reshape(-1))
            return torch.cat(parts, dim=0)
        if "buf_tokens" in mem_state:
            return mem_state["buf_tokens"].detach().cpu().float().reshape(-1)
        return ctx_pool.detach().cpu().float().reshape(-1)

    @torch.no_grad()
    def stream_episode(
        self,
        shard: Dict[str, Any],
        ep_local: int,
        *,
        reset_before: bool = True,
        collect_features: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Stream episode `ep_local` through persistent state, optionally collecting per-chunk probe features."""
        if reset_before:
            self.reset_session()
        else:
            self._flush_eligible(cutoff_global_excl=self._frame_cursor)
            self._clear_intra_episode_short_caches()

        ep_base_global = self._frame_cursor
        self._demo_frames_count = 0
        s = int(shard["ep_offsets"][ep_local].item())
        e = int(shard["ep_offsets"][ep_local + 1].item())
        ep_len = e - s
        exec_start = int(shard["exec_start_idx"][ep_local].item())

        front_key = "front_vlm" if "front_vlm" in shard else "front_emb"
        wrist_key = "wrist_vlm" if "wrist_vlm" in shard else "wrist_emb"
        front_ep = shard[front_key][s:e]
        wrist_ep = shard[wrist_key][s:e]
        raw_state_np = shard["state"][s:e].float().numpy()
        state_norm_ep = torch.from_numpy(self.normalizer.normalize_state(raw_state_np)).float()
        is_demo_ep = shard["is_demo"][s:e].bool()

        # Fast path for distractor-only episodes (`collect_features=False`)
        if not collect_features:
            if exec_start > 0:
                self._queue_frame_range(
                    front_ep[:exec_start],
                    wrist_ep[:exec_start],
                    state_norm_ep[:exec_start],
                    is_demo_ep[:exec_start],
                    is_demo_segment=True,
                )
            self._demo_frames_count = ep_base_global + exec_start
            if exec_start < ep_len:
                self._queue_frame_range(
                    front_ep[exec_start:ep_len],
                    wrist_ep[exec_start:ep_len],
                    state_norm_ep[exec_start:ep_len],
                    is_demo_ep[exec_start:ep_len],
                    is_demo_segment=False,
                )
            self._flush_eligible(cutoff_global_excl=self._frame_cursor)
            return None

        chunk_steps = np.arange(exec_start, ep_len, self.exec_horizon, dtype=np.int64)
        if chunk_steps.size == 0:
            chunk_steps = np.array([min(exec_start, max(0, ep_len - 1))], dtype=np.int64)
        M_ch = int(chunk_steps.size)
        c_idx_cpu = torch.from_numpy(chunk_steps).long()

        f_batch = front_ep[c_idx_cpu].to(device=self.device, dtype=self.dtype)
        w_batch = wrist_ep[c_idx_cpu].to(device=self.device, dtype=self.dtype)
        s_batch = state_norm_ep[c_idx_cpu].to(device=self.device, dtype=torch.float32)

        if self.model.readout_ctx_encoder is not None:
            lang_ids = shard["lang_tokens"][ep_local].unsqueeze(0).to(device=self.device, dtype=torch.long)
            lang_mask = shard["lang_masks"][ep_local].unsqueeze(0).to(device=self.device, dtype=torch.bool)
            lang_raw_1 = self.model.vlm_with_expert.embed_language_tokens(lang_ids)  # (1, 48, 960)

            if self.cfg.obs_horizon2:
                prev_steps = np.maximum(exec_start, chunk_steps - 1)
                p_idx_cpu = torch.from_numpy(prev_steps).long()
                pf_batch = front_ep[p_idx_cpu].to(device=self.device, dtype=self.dtype)
                pw_batch = wrist_ep[p_idx_cpu].to(device=self.device, dtype=self.dtype)
                ps_batch = state_norm_ep[p_idx_cpu].to(device=self.device, dtype=torch.float32)
            else:
                pf_batch = pw_batch = ps_batch = None

            ctx_tokens_all, ctx_mask_all = self.model.encode_readout_context(
                f_batch,
                w_batch,
                s_batch,
                lang_raw_1.expand(M_ch, -1, -1),
                lang_mask.expand(M_ch, -1),
                prev_front_vlm=pf_batch,
                prev_wrist_vlm=pw_batch,
                prev_state_norm=ps_batch,
            )
            assert ctx_tokens_all is not None and ctx_mask_all is not None
            m_f_all = ctx_mask_all.float().unsqueeze(-1)
            ctx_pool_all = (ctx_tokens_all.float() * m_f_all).sum(dim=1) / m_f_all.sum(dim=1).clamp(min=1.0)
        else:
            # Fallback if `use_grounding_head=False` on `smolvla_m0_base` or `smolvla_m2_framesamp`
            ctx_tokens_all = None
            ctx_mask_all = None
            w16_batch = pool_vlm_tokens_8x8_to_4x4(w_batch)
            ctx_pool_all = torch.cat(
                [f_batch.float().mean(dim=1), w16_batch.float().mean(dim=1), s_batch.float()], dim=-1
            )

        # Ingest conditioning video demo `[0 : exec_start]` first
        if exec_start > 0:
            self._queue_frame_range(
                front_ep[:exec_start],
                wrist_ep[:exec_start],
                state_norm_ep[:exec_start],
                is_demo_ep[:exec_start],
                is_demo_segment=True,
            )
            self._flush_eligible(cutoff_global_excl=ep_base_global + exec_start)
        self._demo_frames_count = ep_base_global + exec_start

        local_ingested = exec_start
        L_eval = self.cfg.max_loops if self.model.is_multi_depth else 1

        ctx_pools: List[torch.Tensor] = [ctx_pool_all[i].detach().cpu().float() for i in range(M_ch)]
        state_vecs: List[torch.Tensor] = []
        collected_mem_states: List[Optional[Dict[str, torch.Tensor]]] = []

        for m_idx, t_step_np in enumerate(chunk_steps):
            t_step = int(t_step_np)
            if t_step > local_ingested:
                self._queue_frame_range(
                    front_ep[local_ingested:t_step],
                    wrist_ep[local_ingested:t_step],
                    state_norm_ep[local_ingested:t_step],
                    is_demo_ep[local_ingested:t_step],
                    is_demo_segment=False,
                )
                local_ingested = t_step

            cutoff_local = self.model.compute_a3_cutoff_excl(
                step_idx=t_step,
                exec_start_idx=exec_start,
                exclusion_window=self.cfg.exclusion_window,
            )
            self._flush_eligible(cutoff_global_excl=ep_base_global + cutoff_local)

            mem_state = self._current_mem_state_dict()
            state_vecs.append(self._flatten_state_vector(mem_state, ctx_pools[m_idx]))
            if mem_state is None:
                collected_mem_states.append(None)
            else:
                collected_mem_states.append({k: v.clone() for k, v in mem_state.items() if v is not None})

            # Ingest current frame `t_step` before advancing
            self._queue_frame_range(
                front_ep[t_step : t_step + 1],
                wrist_ep[t_step : t_step + 1],
                state_norm_ep[t_step : t_step + 1],
                is_demo_ep[t_step : t_step + 1],
                is_demo_segment=False,
            )
            local_ingested = t_step + 1

        # Batched multi-hop Coda readout across all `M_ch` chunks of the episode
        if collected_mem_states[0] is None:
            batched_mem = None
        else:
            keys = list(collected_mem_states[0].keys())
            batched_mem = {
                k: torch.cat([ms[k] for ms in collected_mem_states if ms is not None], dim=0)
                for k in keys
            }

        ro_all = self.model.run_coda_readout(
            ctx_tokens_all,
            ctx_mask_all,
            mem_state=batched_mem,
            batch_size=M_ch,
            device=self.device,
            num_loops=L_eval,
            halt_mode="fixed",
        )
        coda_list_all = ro_all["coda_per_depth"]
        ro_mem_mask_all = ro_all.get("mem_mask")

        coda_by_depth: List[List[torch.Tensor]] = [[] for _ in range(L_eval)]
        zero_shot_xy_by_depth: List[List[torch.Tensor]] = [[] for _ in range(L_eval)]
        has_zero_shot_u4 = False

        for l_i in range(L_eval):
            c_tok_m = coda_list_all[l_i]
            if c_tok_m is None:
                coda_by_depth[l_i] = list(ctx_pools)
            else:
                c_cpu = c_tok_m.detach().cpu().float().reshape(M_ch, -1)
                coda_by_depth[l_i] = [c_cpu[i] for i in range(M_ch)]

            pxy_all = self.model.predict_grounding_xy(
                ctx_tokens_all, c_tok_m, mem_mask=ro_mem_mask_all
            )
            if pxy_all is not None:
                has_zero_shot_u4 = True
                pxy = pxy_all.detach().cpu().float()
                zero_shot_xy_by_depth[l_i] = [pxy[i] for i in range(M_ch)]

        halt_dict = ro_all.get("halt")
        if halt_dict is not None and "p_tilde" in halt_dict and L_eval > 1:
            probs_m = halt_dict["p_tilde"].detach().cpu().float()  # (M_ch, L_eval)
            depths = torch.arange(1, L_eval + 1, dtype=torch.float32).unsqueeze(0)
            exp_m = (probs_m * depths).sum(dim=1)
            cdf_m = torch.cumsum(probs_m, dim=1)
            hit_m = cdf_m >= 0.5
            hit_m[:, -1] = True
            gate_m = (torch.argmax(hit_m.int(), dim=1) + 1).float()
            halt_expected_L = exp_m.tolist()
            halt_gate_L = gate_m.tolist()
        else:
            halt_expected_L = [float(L_eval)] * M_ch
            halt_gate_L = [float(L_eval)] * M_ch

        coda_conv_L: List[float] = []
        if L_eval > 1 and coda_list_all[0] is not None and coda_list_all[-1] is not None:
            c_stack = torch.stack([c.detach().cpu().float() for c in coda_list_all if c is not None], dim=1)
            drift_m = torch.linalg.vector_norm(c_stack[:, -1] - c_stack[:, 0], ord=2, dim=-1).mean(dim=-1) / (
                torch.linalg.vector_norm(c_stack[:, 0], ord=2, dim=-1).mean(dim=-1) + 1e-5
            )
            coda_rel_drift = drift_m.tolist()
            for i in range(M_ch):
                cd = L_eval
                for l_i in range(1, L_eval):
                    rd = float(
                        (
                            torch.linalg.vector_norm(c_stack[i, l_i] - c_stack[i, l_i - 1], ord=2, dim=-1).mean()
                            / (torch.linalg.vector_norm(c_stack[i, l_i - 1], ord=2, dim=-1).mean() + 1e-5)
                        ).item()
                    )
                    if rd < 0.08:
                        cd = l_i + 1
                        break
                coda_conv_L.append(float(cd))
        else:
            coda_rel_drift = [0.0] * M_ch
            coda_conv_L = [1.0] * M_ch

        # Ingest remaining tail frames `[local_ingested : ep_len]` so cross-session state is complete
        if local_ingested < ep_len:
            self._queue_frame_range(
                front_ep[local_ingested:ep_len],
                wrist_ep[local_ingested:ep_len],
                state_norm_ep[local_ingested:ep_len],
                is_demo_ep[local_ingested:ep_len],
                is_demo_segment=False,
            )
        self._flush_eligible(cutoff_global_excl=self._frame_cursor)

        # Extract per-chunk ground-truth subgoal labels at `chunk_steps`
        grounded_ep = shard["grounded_subgoals"][ep_local]
        simple_ep = shard["simple_subgoals"][ep_local]
        g_parsed = parse_grounded_subgoals(grounded_ep, resolve_complete=False)
        s_meta = extract_subgoal_metadata([simple_ep], exec_start_idx=[exec_start])

        c_idx_t = torch.from_numpy(chunk_steps).long()
        trans_frame_mask = s_meta["is_transition"]
        chunk_has_trans = torch.zeros((len(chunk_steps),), dtype=torch.bool)
        for m_i, t_s in enumerate(chunk_steps.tolist()):
            lo = max(exec_start, int(t_s) - 4)
            hi = min(ep_len, int(t_s) + self.exec_horizon)
            chunk_has_trans[m_i] = bool(trans_frame_mask[lo:hi].any().item())

        out: Dict[str, Any] = {
            "ep_local": ep_local,
            "exec_start": exec_start,
            "chunk_steps": c_idx_t,
            "elapsed_exec_steps": c_idx_t - exec_start,
            "ctx_pool": torch.stack(ctx_pools, dim=0),
            "state_vec": torch.stack(state_vecs, dim=0),
            "coda_by_depth": [torch.stack(coda_by_depth[l], dim=0) for l in range(L_eval)],
            "xy_norm": g_parsed["xy_norm"][c_idx_t],
            "has_xy": g_parsed["has_xy"][c_idx_t],
            "spatial_bin_3x3": xy_norm_to_3x3_bin(g_parsed["xy_norm"][c_idx_t]),
            "ordinal_index": s_meta["ordinal_index"][c_idx_t],
            "completed_picks": s_meta["completed_picks"][c_idx_t],
            "completed_transitions": s_meta["completed_transitions"][c_idx_t],
            "stage_id": s_meta["stage_id"][c_idx_t],
            "color_id": s_meta["color_id"][c_idx_t],
            "chunk_has_transition": chunk_has_trans,
            "halt_expected_L": torch.tensor(halt_expected_L, dtype=torch.float32),
            "halt_gate_L": torch.tensor(halt_gate_L, dtype=torch.float32),
            "coda_conv_L": torch.tensor(coda_conv_L, dtype=torch.float32),
            "coda_rel_drift": torch.tensor(coda_rel_drift, dtype=torch.float32),
        }
        if has_zero_shot_u4:
            out["zero_shot_xy_by_depth"] = [torch.stack(zero_shot_xy_by_depth[l], dim=0) for l in range(L_eval)]
        return out


# =============================================================================
# 2. Probes 1, 2, & 3 Runner (100% Schema Parity with `probe_robomme_state.py`)
# =============================================================================


def evaluate_smolvla_probes_on_shard(
    model: SmolVLADeltaLoopPolicy,
    normalizer: MeanStdDeltaNormalizer,
    shard: Dict[str, Any],
    device: torch.device,
    *,
    train_frac: float = 0.8,
    max_episodes: Optional[int] = None,
    distractor_ks: Sequence[int] = (0, 1, 3),
    run_mlp_probe1: bool = True,
) -> Dict[str, Any]:
    """Execute Probes 1, 2, and 3 on a single `robomme_smolvla_v1/{TaskName}.pt` shard."""
    total_eps = int(shard["ep_offsets"].numel() - 1)
    n_use = min(total_eps, int(max_episodes)) if max_episodes is not None else total_eps
    if n_use < 2:
        raise ValueError(f"Need at least 2 episodes for episode-disjoint probing, got {n_use}")

    n_train = max(1, min(n_use - 1, int(round(n_use * float(train_frac)))))
    train_ep_indices = list(range(0, n_train))
    eval_ep_indices = list(range(n_train, n_use))

    streamer = SmolVLAShardSessionStreamer(model, normalizer, device=device)
    L_eval = model.cfg.max_loops if model.is_multi_depth else 1

    train_records: List[Dict[str, Any]] = []
    for e_i in train_ep_indices:
        rec = streamer.stream_episode(shard, e_i, reset_before=True, collect_features=True)
        assert rec is not None
        train_records.append(rec)

    eval_records_k0: List[Dict[str, Any]] = []
    for e_i in eval_ep_indices:
        rec = streamer.stream_episode(shard, e_i, reset_before=True, collect_features=True)
        assert rec is not None
        eval_records_k0.append(rec)

    tr = _concat_ep_records(train_records, L_eval)
    te = _concat_ep_records(eval_records_k0, L_eval)

    # -------------------------------------------------------------------------
    # PROBE 1: Hop-by-Hop `<x, y>` Target Dereferencing (`l = 1 .. L_eval`)
    # -------------------------------------------------------------------------
    tr_xy_mask = tr["has_xy"]
    te_xy_mask = te["has_xy"]
    probe1_out: Dict[str, Any] = {
        "n_train_xy_chunks": int(tr_xy_mask.sum().item()),
        "n_eval_xy_chunks": int(te_xy_mask.sum().item()),
        "ctx_only_ridge": {},
        "coda_only_by_depth": {},
        "fused_ctx_coda_by_depth": {},
    }
    fitted_fused_L_reg: Optional[StandardizedDualRidgeRegressor] = None

    if int(tr_xy_mask.sum().item()) >= 4 and int(te_xy_mask.sum().item()) >= 2:
        Y_tr_xy = tr["xy_norm"][tr_xy_mask]
        Y_te_xy = te["xy_norm"][te_xy_mask]
        ep_tr_xy = tr["ep_ids"][tr_xy_mask]

        ctx_reg = StandardizedDualRidgeRegressor().fit(
            tr["ctx_pool"][tr_xy_mask], Y_tr_xy, ep_ids=ep_tr_xy
        )
        probe1_out["ctx_only_ridge"] = compute_xy_metrics(
            ctx_reg.predict(te["ctx_pool"][te_xy_mask]), Y_te_xy
        )

        if run_mlp_probe1:
            probe1_out["mlp_fused_by_depth"] = {}

        for l_idx in range(L_eval):
            depth_key = f"l={l_idx + 1}"
            c_tr = tr["coda_by_depth"][l_idx][tr_xy_mask]
            c_te = te["coda_by_depth"][l_idx][te_xy_mask]
            c_reg = StandardizedDualRidgeRegressor().fit(c_tr, Y_tr_xy, ep_ids=ep_tr_xy)
            probe1_out["coda_only_by_depth"][depth_key] = compute_xy_metrics(
                c_reg.predict(c_te), Y_te_xy
            )

            f_tr = torch.cat([tr["ctx_pool"][tr_xy_mask], c_tr], dim=-1)
            f_te = torch.cat([te["ctx_pool"][te_xy_mask], c_te], dim=-1)
            f_reg = StandardizedDualRidgeRegressor().fit(f_tr, Y_tr_xy, ep_ids=ep_tr_xy)
            probe1_out["fused_ctx_coda_by_depth"][depth_key] = compute_xy_metrics(
                f_reg.predict(f_te), Y_te_xy
            )
            if l_idx == L_eval - 1:
                fitted_fused_L_reg = f_reg

            if run_mlp_probe1:
                mlp_pred = fit_and_predict_mlp_xy(f_tr, Y_tr_xy, f_te, seed=42 + l_idx)
                probe1_out["mlp_fused_by_depth"][depth_key] = compute_xy_metrics(mlp_pred, Y_te_xy)

        if "zero_shot_xy_by_depth" in te:
            probe1_out["zero_shot_u4_by_depth"] = {
                f"l={l_idx + 1}": compute_xy_metrics(
                    te["zero_shot_xy_by_depth"][l_idx][te_xy_mask], Y_te_xy
                )
                for l_idx in range(L_eval)
            }
            if L_eval > 1:
                zs_l1 = probe1_out["zero_shot_u4_by_depth"]["l=1"]["mean_px_err"]
                zs_lmax = probe1_out["zero_shot_u4_by_depth"][f"l={L_eval}"]["mean_px_err"]
                probe1_out["zero_shot_u4_hop_px_improvement_l1_to_lmax"] = float(zs_l1 - zs_lmax)

        if L_eval > 1:
            px_l1 = probe1_out["coda_only_by_depth"]["l=1"]["mean_px_err"]
            px_lmax = probe1_out["coda_only_by_depth"][f"l={L_eval}"]["mean_px_err"]
            probe1_out["coda_hop_px_improvement_l1_to_lmax"] = float(px_l1 - px_lmax)

    # -------------------------------------------------------------------------
    # PROBE 2: Frozen `512 KB` `(S1_past, S2_past)` Count & Post-Swap Retention
    # -------------------------------------------------------------------------
    probe2_out: Dict[str, Any] = {
        "state_feature_dim": int(tr["state_vec"].shape[1]),
    }
    fitted_state_count_clf: Optional[StandardizedRidgeClassifier] = None
    fitted_state_bin_clf: Optional[StandardizedRidgeClassifier] = None

    count_label_key = (
        "completed_picks"
        if torch.unique(tr["completed_picks"]).numel() >= 2
        else "completed_transitions"
    )
    probe2_out["count_target_name"] = count_label_key

    y_cnt_tr = tr[count_label_key]
    y_cnt_te = te[count_label_key]
    if torch.unique(y_cnt_tr).numel() >= 2 and y_cnt_te.numel() >= 2:
        fitted_state_count_clf = StandardizedRidgeClassifier().fit(
            tr["state_vec"], y_cnt_tr, ep_ids=tr["ep_ids"]
        )
        pred_cnt_state = fitted_state_count_clf.predict(te["state_vec"])
        probe2_out["count_from_S_past"] = compute_cls_metrics(pred_cnt_state, y_cnt_te)

        coda_cnt_clf = StandardizedRidgeClassifier().fit(
            tr["coda_by_depth"][-1], y_cnt_tr, ep_ids=tr["ep_ids"]
        )
        probe2_out["count_from_coda_L"] = compute_cls_metrics(
            coda_cnt_clf.predict(te["coda_by_depth"][-1]), y_cnt_te
        )

        el = te["elapsed_exec_steps"]
        by_horizon: Dict[str, Any] = {}
        for h_name, h_mask in (
            ("early_lt32", el < 32),
            ("mid_32_95", (el >= 32) & (el < 96)),
            ("late_ge96", el >= 96),
        ):
            if int(h_mask.sum().item()) > 0:
                by_horizon[h_name] = compute_cls_metrics(pred_cnt_state[h_mask], y_cnt_te[h_mask])
        probe2_out["count_from_S_past_by_horizon"] = by_horizon

    if int(tr_xy_mask.sum().item()) >= 4 and int(te_xy_mask.sum().item()) >= 2:
        y_bin_tr = tr["spatial_bin_3x3"][tr_xy_mask]
        y_bin_te = te["spatial_bin_3x3"][te_xy_mask]
        if torch.unique(y_bin_tr).numel() >= 2:
            fitted_state_bin_clf = StandardizedRidgeClassifier().fit(
                tr["state_vec"][tr_xy_mask], y_bin_tr, ep_ids=tr["ep_ids"][tr_xy_mask]
            )
            probe2_out["target_3x3_bin_from_S_past"] = compute_cls_metrics(
                fitted_state_bin_clf.predict(te["state_vec"][te_xy_mask]), y_bin_te
            )
        s_xy_reg = StandardizedDualRidgeRegressor().fit(
            tr["state_vec"][tr_xy_mask], tr["xy_norm"][tr_xy_mask], ep_ids=tr["ep_ids"][tr_xy_mask]
        )
        probe2_out["target_xy_from_S_past"] = compute_xy_metrics(
            s_xy_reg.predict(te["state_vec"][te_xy_mask]), te["xy_norm"][te_xy_mask]
        )

    if torch.unique(tr["stage_id"]).numel() >= 2:
        st_clf = StandardizedRidgeClassifier().fit(tr["state_vec"], tr["stage_id"], ep_ids=tr["ep_ids"])
        probe2_out["stage_from_S_past"] = compute_cls_metrics(
            st_clf.predict(te["state_vec"]), te["stage_id"]
        )

    # -------------------------------------------------------------------------
    # PROBE 3: Cross-Session Interference (`k in {0, 1, 3}`) & CRN Halting `E[L_t | k]`
    # -------------------------------------------------------------------------
    trans_m = te["chunk_has_transition"]
    onset_m = te["elapsed_exec_steps"] < (2 * streamer.exec_horizon)
    steady_m = (~trans_m) & (~onset_m)

    def _profile_slice(mask: torch.Tensor) -> Dict[str, float]:
        if int(mask.sum().item()) == 0:
            return {
                "n_chunks": 0,
                "mean_expected_L": float("nan"),
                "mean_gate_L": float("nan"),
                "mean_coda_conv_L": float("nan"),
                "mean_coda_drift": float("nan"),
            }
        return {
            "n_chunks": int(mask.sum().item()),
            "mean_expected_L": float(te["halt_expected_L"][mask].mean().item()),
            "mean_gate_L": float(te["halt_gate_L"][mask].mean().item()),
            "mean_coda_conv_L": float(te["coda_conv_L"][mask].mean().item()),
            "mean_coda_drift": float(te["coda_rel_drift"][mask].mean().item()),
        }

    event_loop_profile = {
        "overall": _profile_slice(torch.ones_like(trans_m, dtype=torch.bool)),
        "post_demo_onset": _profile_slice(onset_m),
        "subgoal_transition": _profile_slice(trans_m),
        "steady_execution": _profile_slice(steady_m),
    }

    cross_session_by_k: Dict[str, Any] = {}
    coda_k0_ref = te["coda_by_depth"][-1]

    for k_val in distractor_ks:
        k_int = int(k_val)
        if k_int == 0 or (not model.uses_history):
            te_k = te
        else:
            eval_recs_k: List[Dict[str, Any]] = []
            for eval_pos, ep_target in enumerate(eval_ep_indices):
                streamer.reset_session()
                for d_step in range(k_int):
                    dist_ep = train_ep_indices[(eval_pos * 7 + d_step * 3) % len(train_ep_indices)]
                    streamer.stream_episode(
                        shard, dist_ep, reset_before=False, collect_features=False
                    )
                rec_k = streamer.stream_episode(
                    shard, ep_target, reset_before=False, collect_features=True
                )
                assert rec_k is not None
                eval_recs_k.append(rec_k)
            te_k = _concat_ep_records(eval_recs_k, L_eval)

        coda_k = te_k["coda_by_depth"][-1]
        if model.uses_history:
            cos_sim = float(
                F.cosine_similarity(coda_k.float(), coda_k0_ref.float(), dim=-1).mean().item()
            )
        else:
            cos_sim = 1.0

        k_entry: Dict[str, Any] = {
            "k_distractor_episodes": k_int,
            "coda_cos_sim_vs_k0": cos_sim,
            "mean_expected_L": float(te_k["halt_expected_L"].mean().item()),
            "mean_gate_L": float(te_k["halt_gate_L"].mean().item()),
            "mean_coda_conv_L": float(te_k["coda_conv_L"].mean().item()),
        }
        if fitted_fused_L_reg is not None and int(te_k["has_xy"].sum().item()) >= 2:
            m_xy = te_k["has_xy"]
            f_te_k = torch.cat([te_k["ctx_pool"][m_xy], coda_k[m_xy]], dim=-1)
            xy_m = compute_xy_metrics(fitted_fused_L_reg.predict(f_te_k), te_k["xy_norm"][m_xy])
            k_entry["probe1_fused_px_err"] = xy_m["mean_px_err"]
            k_entry["probe1_fused_r2"] = xy_m["r2"]
        if "zero_shot_xy_by_depth" in te_k and int(te_k["has_xy"].sum().item()) >= 2:
            m_xy = te_k["has_xy"]
            zs_m = compute_xy_metrics(te_k["zero_shot_xy_by_depth"][-1][m_xy], te_k["xy_norm"][m_xy])
            k_entry["probe1_zero_shot_px_err"] = zs_m["mean_px_err"]
        if fitted_state_count_clf is not None:
            cm = compute_cls_metrics(
                fitted_state_count_clf.predict(te_k["state_vec"]), te_k[count_label_key]
            )
            k_entry["probe2_count_acc_pct"] = cm["acc_pct"]
        if fitted_state_bin_clf is not None and int(te_k["has_xy"].sum().item()) >= 2:
            m_xy = te_k["has_xy"]
            bm = compute_cls_metrics(
                fitted_state_bin_clf.predict(te_k["state_vec"][m_xy]), te_k["spatial_bin_3x3"][m_xy]
            )
            k_entry["probe2_target_3x3_acc_pct"] = bm["acc_pct"]

        cross_session_by_k[f"k={k_int}"] = k_entry

    max_k = max(int(k) for k in distractor_ks)
    if "k=0" in cross_session_by_k and f"k={max_k}" in cross_session_by_k and max_k > 0:
        e0 = cross_session_by_k["k=0"]
        eM = cross_session_by_k[f"k={max_k}"]
        cross_session_summary = {
            f"coda_cos_sim_at_k{max_k}": eM["coda_cos_sim_vs_k0"],
            f"px_err_increase_k0_to_k{max_k}": (
                float(eM["probe1_fused_px_err"] - e0["probe1_fused_px_err"])
                if "probe1_fused_px_err" in e0 and "probe1_fused_px_err" in eM
                else float("nan")
            ),
            f"zero_shot_px_err_increase_k0_to_k{max_k}": (
                float(eM["probe1_zero_shot_px_err"] - e0["probe1_zero_shot_px_err"])
                if "probe1_zero_shot_px_err" in e0 and "probe1_zero_shot_px_err" in eM
                else float("nan")
            ),
            f"count_acc_drop_k0_to_k{max_k}": (
                float(e0["probe2_count_acc_pct"] - eM["probe2_count_acc_pct"])
                if "probe2_count_acc_pct" in e0 and "probe2_count_acc_pct" in eM
                else float("nan")
            ),
            f"expected_L_shift_k0_to_k{max_k}": float(
                eM.get("mean_expected_L", float("nan")) - e0.get("mean_expected_L", float("nan"))
            ),
        }
    else:
        cross_session_summary = {}

    return {
        "n_episodes_used": n_use,
        "n_train_episodes": len(train_ep_indices),
        "n_eval_episodes": len(eval_ep_indices),
        "L_eval": L_eval,
        "probe1_xy_localization": probe1_out,
        "probe2_state_retention": probe2_out,
        "probe3_cross_session_and_halting": {
            "event_loop_profile": event_loop_profile,
            "by_distractor_k": cross_session_by_k,
            "summary": cross_session_summary,
        },
    }


# =============================================================================
# 3. Synthetic `robomme_smolvla_v1` Shard Builder & CPU `--smoke-test`
# =============================================================================


def build_synthetic_smolvla_shard(
    task_name: str,
    *,
    num_episodes: int = 10,
    seed: int = 303,
) -> Dict[str, Any]:
    """Build an in-memory `robomme_smolvla_v1/{TaskName}.pt` shard (`front_vlm/wrist_vlm` `(N, 64, 960)` bf16)."""
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    basis = torch.randn((8, 960), generator=gen, dtype=torch.float32) * 0.25

    front_list: List[torch.Tensor] = []
    wrist_list: List[torch.Tensor] = []
    state_list: List[torch.Tensor] = []
    actions_list: List[torch.Tensor] = []
    is_demo_list: List[torch.Tensor] = []
    ep_offsets = [0]
    exec_starts: List[int] = []
    epis_idx: List[int] = []
    prompts: List[str] = []
    simple_sgs: List[List[str]] = []
    grounded_sgs: List[List[str]] = []

    ordinals = ["first", "second", "third"]
    colors = ["red", "green", "blue"]

    for ep in range(num_episodes):
        # Include >256-frame demo episodes (`264 > 256`) to test `U5` adaptive `long_demo_stride=8`!
        demo_len = 264 if (ep % 5 == 0) else 36
        exec_len = 112  # Covers `early <32`, `mid 32..95`, AND `late >=96` horizon bins!
        T = demo_len + exec_len
        exec_starts.append(demo_len)
        epis_idx.append(ep)

        col_x = int(rng.integers(32, 224))
        row_y = int(rng.integers(32, 224))
        x_norm = ((col_x + 0.5) / 256.0) * 2.0 - 1.0
        y_norm = ((row_y + 0.5) / 256.0) * 2.0 - 1.0
        color = colors[ep % len(colors)]

        f_ep = torch.randn((T, 64, 960), generator=gen, dtype=torch.float32) * 0.08
        w_ep = torch.randn((T, 64, 960), generator=gen, dtype=torch.float32) * 0.08
        xy_sig = x_norm * basis[0] + y_norm * basis[1]
        f_ep[:demo_len] += xy_sig.view(1, 1, 960) * 1.5
        f_ep[demo_len:] += xy_sig.view(1, 1, 960) * 0.35

        st_ep = torch.zeros((T, 8), dtype=torch.float32)
        act_ep = torch.zeros((T, 8), dtype=torch.float32)
        demo_mask = torch.zeros((T,), dtype=torch.bool)
        demo_mask[:demo_len] = True

        ep_simple: List[str] = []
        ep_grounded: List[str] = []
        for t in range(T):
            if t < demo_len:
                ep_simple.append("NO RECORD")
                ep_grounded.append("NO RECORD")
            else:
                rel = t - demo_len
                stage_idx = min(2, rel // 38)
                f_ep[t] += float(stage_idx + 1) * basis[2 + stage_idx].view(1, 960)
                st_ep[t, 0] = x_norm * 0.2
                st_ep[t, 1] = y_norm * 0.2
                st_ep[t, 7] = 1.0 if (rel % 38) < 20 else -1.0

                if task_name == "PickXtimes":
                    ord_word = ordinals[stage_idx]
                    if (rel % 38) < 26:
                        ep_simple.append(f"pick up the {color} cube for the {ord_word} time")
                        ep_grounded.append(
                            f"pick up the {color} cube at <{row_y}, {col_x}> for the {ord_word} time"
                        )
                    else:
                        ep_simple.append("put down")
                        ep_grounded.append("put down")
                else:
                    if rel < 24:
                        ep_simple.append("press the button to unmask")
                        ep_grounded.append("press the first button at <60, 128>")
                    elif rel < 96:
                        ep_simple.append(f"pick up the container that hides the {color} cube")
                        ep_grounded.append(
                            f"pick up the container at <{row_y}, {col_x}> that hides the {color} cube"
                        )
                    else:
                        ep_simple.append("complete")
                        ep_grounded.append("complete")

        front_list.append(f_ep.to(torch.bfloat16))
        wrist_list.append(w_ep.to(torch.bfloat16))
        state_list.append(st_ep)
        actions_list.append(act_ep)
        is_demo_list.append(demo_mask)
        ep_offsets.append(ep_offsets[-1] + T)
        prompts.append(f"solve {task_name} for the {color} cube\n")
        simple_sgs.append(ep_simple)
        grounded_sgs.append(ep_grounded)

    lang_tokens = torch.randint(1, 512, (num_episodes, 48), generator=gen, dtype=torch.long)
    lang_masks = torch.ones((num_episodes, 48), dtype=torch.bool)
    lang_masks[:, 14:] = False

    return {
        "task_name": task_name,
        "suite": "SyntheticSmolVLASmoke",
        "front_vlm": torch.cat(front_list, dim=0),
        "wrist_vlm": torch.cat(wrist_list, dim=0),
        "state": torch.cat(state_list, dim=0),
        "actions": torch.cat(actions_list, dim=0),
        "is_demo": torch.cat(is_demo_list, dim=0),
        "ep_offsets": torch.tensor(ep_offsets, dtype=torch.long),
        "exec_start_idx": torch.tensor(exec_starts, dtype=torch.long),
        "epis_idx": torch.tensor(epis_idx, dtype=torch.long),
        "prompts": prompts,
        "lang_tokens": lang_tokens,
        "lang_masks": lang_masks,
        "simple_subgoals": simple_sgs,
        "grounded_subgoals": grounded_sgs,
    }


@torch.no_grad()
def _verify_smolvla_streaming_vs_packed_parity(
    model: SmolVLADeltaLoopPolicy,
    normalizer: MeanStdDeltaNormalizer,
    shard: Dict[str, Any],
    device: torch.device,
) -> float:
    """Verify `< 1e-5` `(S1, S2)` parity between `SmolVLAShardSessionStreamer` and `compute_packed_gdn_states`."""
    streamer = SmolVLAShardSessionStreamer(model, normalizer, device=device, force_pytorch_gdn=True)
    streamer.reset_session()

    # Episode 0 has `exec_start == 264 > 256`, exercising `long_demo_stride=8` + `write_stride=4` + motion RMS!
    s = int(shard["ep_offsets"][0].item())
    e = int(shard["ep_offsets"][1].item())
    exec_start = int(shard["exec_start_idx"][0].item())
    f_ep = shard["front_vlm"][s:e]
    w_ep = shard["wrist_vlm"][s:e]
    st_norm_ep = torch.from_numpy(normalizer.normalize_state(shard["state"][s:e].float().numpy())).float()
    is_demo_ep = shard["is_demo"][s:e].bool()

    # Stream up to `t_step = exec_start + 32` (`cutoff = exec_start + 17`)
    t_target = exec_start + 32
    streamer._queue_frame_range(
        f_ep[:exec_start], w_ep[:exec_start], st_norm_ep[:exec_start], is_demo_ep[:exec_start], is_demo_segment=True
    )
    streamer._flush_eligible(cutoff_global_excl=exec_start)
    streamer._demo_frames_count = exec_start
    streamer._queue_frame_range(
        f_ep[exec_start:t_target],
        w_ep[exec_start:t_target],
        st_norm_ep[exec_start:t_target],
        is_demo_ep[exec_start:t_target],
        is_demo_segment=False,
    )
    cutoff = model.compute_a3_cutoff_excl(t_target, exec_start, exclusion_window=model.cfg.exclusion_window)
    streamer._flush_eligible(cutoff_global_excl=cutoff)
    assert streamer._S1 is not None and streamer._S2 is not None

    w_idx, _ = model.compute_written_frame_indices(cutoff, exec_start, device=device)
    f_ref = f_ep[w_idx.cpu()].to(device=device, dtype=torch.float32)
    w_ref = w_ep[w_idx.cpu()].to(device=device, dtype=torch.float32)
    s_ref = st_norm_ep[w_idx.cpu()].to(device=device, dtype=torch.float32)
    d_ref = (w_idx < exec_start).long()
    raw_ref = model.encode_history_frames_raw(f_ref, w_ref, s_ref, w_idx, d_ref)
    assert model.slot_resampler is not None
    conv_ref = model.slot_resampler.apply_episode_slot_conv(raw_ref)
    mot_ref = model.compute_visual_motion_rms(f_ref)
    packed = model.compute_packed_gdn_states(
        [conv_ref],
        torch.tensor([0], device=device, dtype=torch.long),
        torch.tensor([int(w_idx.numel())], device=device, dtype=torch.long),
        episode_motion_seqs=[mot_ref],
        force_pytorch=True,
    )
    max_diff = max(
        float((streamer._S1[0] - packed["S1"][0]).abs().max().item()),
        float((streamer._S2[0] - packed["S2"][0]).abs().max().item()),
    )
    assert max_diff < 1e-5, f"SmolVLA streamer vs packed GDN mismatch: {max_diff}"
    return max_diff


def run_smoke_test(smoke_ckpt_dir: Optional[Path], output_json: Path) -> None:
    """Smoke-test Probes 1–3 on CPU (`CUDA_VISIBLE_DEVICES=""`) across all 5 `SmolVLA` architectures."""
    device = torch.device("cpu")
    t0 = time.time()

    shards = {
        "ButtonUnmaskSwap": build_synthetic_smolvla_shard("ButtonUnmaskSwap", num_episodes=10, seed=101),
        "PickXtimes": build_synthetic_smolvla_shard("PickXtimes", num_episodes=10, seed=202),
    }

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        ckpt_paths: List[Path] = []
        if smoke_ckpt_dir is not None and smoke_ckpt_dir.is_dir():
            for arch in ALL_SMOLVLA_ARCHS:
                cand = smoke_ckpt_dir / f"{arch}.pt"
                if cand.is_file():
                    ckpt_paths.append(cand)

        if not ckpt_paths:
            # Build lightweight 2-layer VLM/Expert smoke `.pt` checkpoints for all 5 controlled architectures
            # (`512 KB` exact GDN state on `smolvla_m5_gdn2l_l1` and `smolvla_m7_deltaloop`!)
            torch.manual_seed(2026)
            base_cfg = SmolVLADeltaLoopConfig(
                arch="smolvla_m0_base",
                num_vlm_layers=2,
                num_expert_layers=2,
                d_mem=384,
                mem_heads=6,
                mem_ff=1024,
                ctx_layers=1,
                num_write_slots=8,
                gdn_heads=8,
                gdn_d_k=128,
                gdn_d_v=128,
            )
            norm_ms = MeanStdDeltaNormalizer(stats=MME_VLA_NORM_STATS)
            for arch in ALL_SMOLVLA_ARCHS:
                cfg_a = copy.deepcopy(base_cfg)
                cfg_a.arch = arch
                pol = cast_smolvla_precision(
                    SmolVLADeltaLoopPolicy(cfg_a), device=device, dtype=torch.float32
                ).eval()
                # Perturb output heads slightly from zero-init so hop-by-hop `l=1..4` & `U4` produce distinct predictions
                with torch.no_grad():
                    if pol.coda is not None:
                        torch.nn.init.normal_(pol.coda.proj.weight, std=0.02)
                    if pol.grounding_xy_head is not None:
                        torch.nn.init.normal_(pol.grounding_xy_head[-1].weight, std=0.02)  # type: ignore[arg-type]
                cp = tmp_dir / f"{arch}.pt"
                torch.save(
                    {
                        "completed": True,
                        "arch": arch,
                        "step": 100,
                        "model_config": cfg_a.to_dict(),
                        "model_state_dict": {k: v.cpu() for k, v in pol.state_dict().items()},
                        "action_mode": "mean_std_delta",
                        "norm_stats": norm_ms.to_stats_dict(),
                    },
                    cp,
                )
                ckpt_paths.append(cp)

        all_results: Dict[str, Any] = {
            "mode": "cpu_smoke_test",
            "device": "cpu",
            "train_frac": 0.7,
            "distractor_ks": [0, 1, 3],
            "models": {},
        }

        max_stream_diff = 0.0
        for cp in ckpt_paths:
            model, normalizer, meta = load_smolvla_checkpoint(cp, device=device)
            assert len(meta["missing_keys"]) == 0, f"Unexpected missing keys in {cp.name}: {meta['missing_keys']}"
            if model.uses_gdn:
                assert meta["state_bytes"] == 532480, f"Expected 532,480 B (512 KB + occ), got {meta['state_bytes']}"
                diff_s = _verify_smolvla_streaming_vs_packed_parity(
                    model, normalizer, shards["ButtonUnmaskSwap"], device=device
                )
                max_stream_diff = max(max_stream_diff, diff_s)

            key = cp.stem
            task_out: Dict[str, Any] = {}
            for task_name, shard in shards.items():
                res = evaluate_smolvla_probes_on_shard(
                    model,
                    normalizer,
                    shard,
                    device=device,
                    train_frac=0.7,
                    distractor_ks=(0, 1, 3),
                    run_mlp_probe1=True,
                )
                p1 = res["probe1_xy_localization"]
                p1_coda = p1["coda_only_by_depth"]
                p1_zs = p1.get("zero_shot_u4_by_depth", {})
                assert len(p1_coda) == meta["max_loops"], f"Expected {meta['max_loops']} depths for {key}"
                assert len(p1_zs) == meta["max_loops"], f"Expected {meta['max_loops']} zero-shot U4 depths for {key}"
                for d_metrics in p1_coda.values():
                    assert math.isfinite(d_metrics["mean_px_err"]), f"Non-finite Ridge px err for {key}"
                for zs_metrics in p1_zs.values():
                    assert math.isfinite(zs_metrics["mean_px_err"]), f"Non-finite U4 px err for {key}"

                p2 = res["probe2_state_retention"]
                assert "late_ge96" in p2.get("count_from_S_past_by_horizon", {}), f"Missing late_ge96 bin for {key}"
                if model.uses_gdn:
                    assert p2["state_feature_dim"] == 133120, f"Expected 133,120 GDN state floats, got {p2['state_feature_dim']}"

                p3_k3 = res["probe3_cross_session_and_halting"]["by_distractor_k"]["k=3"]
                assert math.isfinite(p3_k3["coda_cos_sim_vs_k0"]), f"Non-finite k=3 cos sim for {key}"
                assert math.isfinite(p3_k3["mean_expected_L"]), f"Non-finite E[L_t | k=3] for {key}"
                task_out[task_name] = res

            all_results["models"][key] = {"meta": meta, "tasks": task_out}

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(all_results, indent=2))
    elapsed = time.time() - t0
    print_probe_summary_table(all_results)
    print(
        f"[PASS] Smoke-tested {len(all_results['models'])} SmolVLA architectures "
        f"({list(all_results['models'].keys())}, max_stream_diff={max_stream_diff:.2e}) "
        f"across {list(shards.keys())} on CPU in {elapsed:.2f}s -> {output_json}"
    )


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SmolVLA-DeltaLoop Mechanistic State & Multi-Hop Loop Probe Suite")
    p.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        default=[],
        help="One or more `.pt` SmolVLA checkpoints (`smolvla_m0_base .. smolvla_m7_deltaloop`)",
    )
    p.add_argument(
        "--shard-dir",
        type=str,
        default="",
        help="Directory containing cached `robomme_smolvla_v1/{TaskName}.pt` shards",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "ButtonUnmaskSwap",
            "VideoUnmaskSwap",
            "PickXtimes",
            "StopCube",
            "PickHighlight",
            "VideoPlaceOrder",
            "PatternLock",
            "MoveCube",
            "SwingXtimes",
            "VideoRepick",
        ],
        help="Task names inside `--shard-dir` (pass `all` for all 16 RoboMME tasks)",
    )
    p.add_argument("--train-frac", type=float, default=0.8, help="Episode fraction used to fit probes")
    p.add_argument("--max-episodes", type=int, default=None, help="Optional cap on episodes per shard")
    p.add_argument("--distractor-ks", nargs="+", type=int, default=[0, 1, 3], help="Prior distractor counts `k`")
    p.add_argument("--no-mlp", action="store_true", help="Skip 2-layer MLP check in Probe 1 (Ridge + Zero-Shot U4 only)")
    p.add_argument("--device", type=str, default="cpu", help="Compute device (`cpu` or `cuda`)")
    p.add_argument("--cpu-threads", type=int, default=4, help="Max PyTorch CPU threads when `--device cpu`")
    p.add_argument("--output-json", type=str, default="", help="Path to write structured JSON results")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run self-contained CPU smoke test across all 5 `SmolVLA` architectures",
    )
    p.add_argument(
        "--smoke-ckpt-dir",
        type=str,
        default="",
        help="Optional directory of real `smolvla_*.pt` checkpoints for `--smoke-test`",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()
    if args.device == "cpu" or args.smoke_test:
        torch.set_num_threads(max(1, min(int(args.cpu_threads), os.cpu_count() or 4)))

    if args.smoke_test:
        out_path = Path(args.output_json) if args.output_json else Path("/tmp/probe_smolvla_smoke.json")
        smoke_dir = Path(args.smoke_ckpt_dir) if args.smoke_ckpt_dir else None
        run_smoke_test(smoke_dir, out_path)
        if not args.output_json and out_path.exists():
            out_path.unlink(missing_ok=True)
        return

    if not args.checkpoints or not args.shard_dir:
        raise SystemExit("Provide `--checkpoints ... --shard-dir ...` or run with `--smoke-test`.")

    device = torch.device(args.device)
    shard_dir = Path(args.shard_dir)
    task_names = list(ROBOMME_TASKS) if args.tasks == ["all"] else list(args.tasks)

    loaded_shards: Dict[str, Dict[str, Any]] = {}
    for task_name in task_names:
        sp = shard_dir / f"{task_name}.pt"
        if sp.is_file():
            loaded_shards[task_name] = torch.load(str(sp), map_location="cpu", weights_only=False)
        else:
            print(f"[WARN] Shard not found, skipping: {sp}")

    if not loaded_shards:
        raise FileNotFoundError(f"None of requested tasks {task_names} found in {shard_dir}")

    results: Dict[str, Any] = {
        "mode": "smolvla_cached_shards",
        "shard_dir": str(shard_dir),
        "device": str(device),
        "train_frac": args.train_frac,
        "distractor_ks": list(args.distractor_ks),
        "models": {},
    }

    for cp_str in args.checkpoints:
        cp = Path(cp_str)
        t_m = time.time()
        model, normalizer, meta = load_smolvla_checkpoint(cp, device=device)
        key = cp.stem
        task_map: Dict[str, Any] = {}
        for task_name, shard in loaded_shards.items():
            task_map[task_name] = evaluate_smolvla_probes_on_shard(
                model,
                normalizer,
                shard,
                device=device,
                train_frac=args.train_frac,
                max_episodes=args.max_episodes,
                distractor_ks=args.distractor_ks,
                run_mlp_probe1=(not args.no_mlp),
            )
        meta["probe_wall_sec"] = round(time.time() - t_m, 2)
        results["models"][key] = {"meta": meta, "tasks": task_map}

    print_probe_summary_table(results)
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(results, indent=2))
        print(f"[DONE] Saved SmolVLA mechanistic probe report to {out_p}")


if __name__ == "__main__":
    main()
