#!/usr/bin/env python3
"""Cycle 2A Unified `DeltaLoop-VLA` Multi-Task Trainer & Self-Test Suite (Miyabi GH200).

Trains the controlled visuomotor Flow-Matching policy suite on the 8-Task Representative
RoboMME Benchmark (`2 tasks x 4 suites = 800 episodes`) using pre-extracted multi-camera
spatial SigLIP features (`/work/gj26/b20098/datasets/robomme_siglip_v1`):
  - Permanence: `ButtonUnmaskSwap`, `VideoUnmaskSwap`
  - Counting:   `PickXtimes`, `StopCube`
  - Reference:  `PickHighlight`, `VideoPlaceOrder`
  - Imitation:  `PatternLock`, `MoveCube`

Controlled Models (`seed = 42`, matched shared backbone, EMA=0.999, separate Write vs Loop grad clip):
  - `m0_memoryless`:      Markov Flow-Matching baseline (no history memory, L=1)
  - `m1_rdvla_loop4`:     Memoryless Weight-Tied Looped Core (L=4, no GDN — RD-VLA equivalent)
  - `m2_framesamp_modul`: Perceptual Frame-Sampling buffer (32 past frame slots + Modulator/Cross-Attn)
  - `m3_gdn_b1_l1`:       Standard Single-Pass GDN (beta in (0, 1), L=1)
  - `m4_gdn_b2_l1`:       Negative-Eigenvalue Single-Pass GDN (beta in (0, 2), L=1)
  - `m5_gdn2layer_b2_l1`: 2-Layer Stacked Neg-Eig GDN (beta in (0, 2), L=1, strictly iso-state!)
  - `m6_deltaloop_b2`:    DeltaLoop-VLA (Ours: beta in (0, 2) GDN + Dynamic Multi-Hop L=1..4 +
                          Stage-2 M_CRN=8 Frozen-Backbone Adaptive Halting Gate)
  - `m6b_deltaloop_wb`:   DeltaLoop-VLA + Chunk-Boundary Write-Back (`c_wb`, `g=0.0`) ablation
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.subgoal_grounding import (
    COLOR_VOCAB,
    STAGE_VOCAB,
    extract_subgoal_metadata,
    parse_grounded_subgoals,
    sample_subgrid_crop_affine,
    transform_coords_under_affine,
)
from models.deltaloop_vla import (
    ALL_VLA_ARCHS,
    ArchVLA,
    DeltaLoopVLAConfig,
    DeltaLoopVLAPolicy,
    HAS_FLA,
    run_dual_backend_gdn,
)
from models.robomme_policy_wrapper import RoboMMEPolicyWrapper
from robomme_data.cache_dataset import load_cache_normalizer, load_task_shards
from robomme_eval.interface import ActionStateNormalizer, DELTA_JOINT_MASK, MME_VLA_NORM_STATS

CYCLE2A_TASKS: Tuple[str, ...] = (
    "ButtonUnmaskSwap",
    "VideoUnmaskSwap",
    "PickXtimes",
    "StopCube",
    "PickHighlight",
    "VideoPlaceOrder",
    "PatternLock",
    "MoveCube",
)

PILOT_4_TASKS: Tuple[str, ...] = (
    "ButtonUnmaskSwap",
    "VideoUnmaskSwap",
    "StopCube",
    "MoveCube",
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. Comprehensive Mathematical & Streaming Parity Self-Tests
# =============================================================================


class _MockSiglipEncoder:
    """Deterministic zero-VRAM feature encoder used to verify streaming vs batch parity."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype

    def encode_views(
        self, front_rgb: np.ndarray, wrist_rgb: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f_arr = np.asarray(front_rgb)
        w_arr = np.asarray(wrist_rgb)
        if f_arr.ndim == 3:
            f_arr = f_arr[None, ...]
            w_arr = w_arr[None, ...]
        N = f_arr.shape[0]
        # Deterministic embedding seeded by pixel values encoded in synthetic test frames
        f_mean = torch.from_numpy(f_arr[:, :64, 0, 0].astype(np.float32) / 255.0).to(self.device)
        w_mean = torch.from_numpy(w_arr[:, :16, 0, 0].astype(np.float32) / 255.0).to(self.device)
        f_tok = f_mean.unsqueeze(-1).repeat(1, 1, 768).to(self.dtype)
        w_tok = w_mean.unsqueeze(-1).repeat(1, 1, 768).to(self.dtype)
        return f_tok, w_tok

    def encode_texts(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        B = len(prompts)
        tok = torch.ones((B, 64, 768), device=self.device, dtype=self.dtype) * 0.1
        pool = torch.ones((B, 768), device=self.device, dtype=self.dtype) * 0.1
        mask = torch.zeros((B, 64), device=self.device, dtype=torch.bool)
        mask[:, :12] = True
        return {"text_token_emb": tok, "text_pool_emb": pool, "text_attn_mask": mask}


@torch.no_grad()
def run_mathematical_self_tests(device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    """Verify Zero-Init step-0 parity, A1 fp32 g-mask, A3 cutoff, fla<->PyTorch, and Streaming parity."""
    print(f"[Self-Test] Running mathematical invariants & parity checks on device={device} dtype={dtype} (HAS_FLA={HAS_FLA})...")
    set_all_seeds(1234)
    B = 3
    base_cfg = DeltaLoopVLAConfig(arch="m0_memoryless", d_model=128, attn_heads=4, d_ff=256, ctx_layers=1, expert_layers=2, gdn_heads=4, gdn_d_k=64, gdn_d_v=64)
    m0 = DeltaLoopVLAPolicy(base_cfg).to(device=device, dtype=dtype).eval()

    front = torch.randn((B, 64, 768), device=device, dtype=dtype)
    wrist = torch.randn((B, 16, 768), device=device, dtype=dtype)
    state = torch.randn((B, 8), device=device, dtype=torch.float32)
    txt = torch.randn((B, 64, 768), device=device, dtype=dtype)
    txt_mask = torch.ones((B, 64), device=device, dtype=torch.bool)
    txt_mask[:, 15:] = False
    x_tau = torch.randn((B, 20, 8), device=device, dtype=torch.float32)
    tau = torch.tensor([0.15, 0.50, 0.85], device=device, dtype=torch.float32)

    ctx0, cmask0 = m0.obs_encoder(front, wrist, state, txt, txt_mask)
    v0 = m0.action_expert(x_tau, tau, ctx0, cmask0, mem_tokens=None)

    # 1. Verify exact step-0 zero-init equivalence across ALL models (`m1..m6b`)!
    max_zero_init_diff = 0.0
    for arch in ALL_VLA_ARCHS:
        if arch == "m0_memoryless":
            continue
        cfg_a = copy.deepcopy(base_cfg)
        cfg_a.arch = arch
        ma = DeltaLoopVLAPolicy(cfg_a).to(device=device, dtype=dtype).eval()
        ma.obs_encoder.load_state_dict(m0.obs_encoder.state_dict())
        ma.action_expert.load_state_dict(m0.action_expert.state_dict())

        ctx_a, cmask_a = ma.obs_encoder(front, wrist, state, txt, txt_mask)
        if arch == "m2_framesamp_modul":
            mem_st = {
                "buf_tokens": torch.randn((B, 32, 128), device=device, dtype=dtype),
                "buf_mask": torch.tensor(
                    [[False] * 32, [True] * 8 + [False] * 24, [True] * 32],
                    device=device,
                    dtype=torch.bool,
                ),
            }
        elif arch in ("m5_gdn2layer_b2_l1", "m7_deltaloop_2l_b2"):
            mem_st = {
                "S1": torch.randn((B, 4, 32, 64), device=device, dtype=torch.float32),
                "occ1": torch.randn((B, 4, 64), device=device, dtype=torch.float32),
                "S2": torch.randn((B, 4, 32, 64), device=device, dtype=torch.float32),
                "occ2": torch.randn((B, 4, 64), device=device, dtype=torch.float32),
            }
        elif ma.uses_gdn:
            mem_st = {
                "S": torch.randn((B, 4, 64, 64), device=device, dtype=torch.float32),
                "occ": torch.randn((B, 4, 64), device=device, dtype=torch.float32),
            }
        else:
            mem_st = None

        ro = ma.run_coda_readout(ctx_a, cmask_a, mem_state=mem_st)
        va = ma.action_expert(x_tau, tau, ctx_a, cmask_a, mem_tokens=ro["coda"], mem_mask=ro["mem_mask"])
        assert torch.isfinite(va).all(), f"Non-finite output on {arch} (including empty mask row)!"
        diff = float((va - v0).abs().max().item())
        max_zero_init_diff = max(max_zero_init_diff, diff)
        assert diff < 1e-6, f"Zero-init parity failed on {arch}: max_diff={diff}"

    # 2. Verify A1 frame-wise fp32 log-decay mask (`g[:, :, 1:, :] == 0.0`) & A3 video-demo cutoff
    cfg_m6 = copy.deepcopy(base_cfg)
    cfg_m6.arch = "m6_deltaloop_b2"
    m6 = DeltaLoopVLAPolicy(cfg_m6).to(device=device, dtype=dtype).eval()
    assert m6.gdn is not None
    dummy_slots = torch.randn((2, 7, cfg_m6.num_write_slots, 128), device=device, dtype=dtype)
    k_p, _, g_p, beta_p, _ = m6.gdn.project_frame_slots(dummy_slots)
    assert g_p.dtype == torch.float32 and beta_p.dtype == torch.float32
    g_reshaped = g_p.view(2, 7, cfg_m6.num_write_slots, cfg_m6.gdn_heads)
    assert (g_reshaped[:, :, 0, :] < 0.0).all(), "Slot s=0 must carry negative log-decay!"
    assert (g_reshaped[:, :, 1:, :] == 0.0).all(), "Slots s >= 1 MUST have exact g == 0.0!"
    k_norms = torch.linalg.vector_norm(k_p.float(), ord=2, dim=-1)
    assert float((k_norms - 1.0).abs().max().item()) < 1e-5, "Keys must be unit L2-norm in fp32!"

    assert m6.compute_a3_cutoff_excl(step_idx=168, exec_start_idx=168, exclusion_window=16) == 168
    assert m6.compute_a3_cutoff_excl(step_idx=184, exec_start_idx=168, exclusion_window=16) == 169
    assert m6.compute_a3_cutoff_excl(step_idx=0, exec_start_idx=0, exclusion_window=16) == 0
    assert m6.compute_a3_cutoff_excl(step_idx=15, exec_start_idx=0, exclusion_window=16) == 0
    assert m6.compute_a3_cutoff_excl(step_idx=16, exec_start_idx=0, exclusion_window=16) == 1

    # 3. Verify `fla` Hopper Triton vs Pure-PyTorch exact `< 1e-5` parity when CUDA + fla available
    fla_vs_pt_diff = 0.0
    if HAS_FLA and device.type == "cuda":
        ep_seqs = [
            torch.randn((25, cfg_m6.num_write_slots, 128), device=device, dtype=dtype),
            torch.randn((40, cfg_m6.num_write_slots, 128), device=device, dtype=dtype),
        ]
        s_ep = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
        s_wc = torch.tensor([0, 19, 12, 37], device=device, dtype=torch.long)
        st_fla = m6.compute_packed_gdn_states(ep_seqs, s_ep, s_wc, force_pytorch=False)
        st_pt = m6.compute_packed_gdn_states(ep_seqs, s_ep, s_wc, force_pytorch=True)
        d_s = float((st_fla["S"] - st_pt["S"]).abs().max().item())
        d_occ = float((st_fla["occ"] - st_pt["occ"]).abs().max().item())
        fla_vs_pt_diff = max(d_s, d_occ)
        assert fla_vs_pt_diff < 1e-5, f"fla vs PyTorch parity exceeded 1e-5: {fla_vs_pt_diff}"

        cfg_m5 = copy.deepcopy(base_cfg)
        cfg_m5.arch = "m5_gdn2layer_b2_l1"
        m5 = DeltaLoopVLAPolicy(cfg_m5).to(device=device, dtype=dtype).eval()
        st5_fla = m5.compute_packed_gdn_states(ep_seqs, s_ep, s_wc, force_pytorch=False)
        st5_pt = m5.compute_packed_gdn_states(ep_seqs, s_ep, s_wc, force_pytorch=True)
        d_s5 = max(
            float((st5_fla["S1"] - st5_pt["S1"]).abs().max().item()),
            float((st5_fla["S2"] - st5_pt["S2"]).abs().max().item()),
        )
        fla_vs_pt_diff = max(fla_vs_pt_diff, d_s5)
        assert d_s5 < 1e-5, f"2-layer fla vs PyTorch parity exceeded 1e-5: {d_s5}"

        # Also verify full `fla` backward pass with `cu_seqlens` + `fp32` produces finite non-zero grads!
        with torch.enable_grad():
            m6_grad = DeltaLoopVLAPolicy(cfg_m6).to(device=device, dtype=dtype).train()
            ep_seqs_g = [s.detach().clone().requires_grad_(True) for s in ep_seqs]
            st_g = m6_grad.compute_packed_gdn_states(ep_seqs_g, s_ep, s_wc, force_pytorch=False)
            loss_g = st_g["S"].square().mean() + st_g["occ"].square().mean()
            loss_g.backward()
            assert ep_seqs_g[0].grad is not None and torch.isfinite(ep_seqs_g[0].grad).all()
            assert m6_grad.gdn is not None and m6_grad.gdn.k_proj.weight.grad is not None
            assert torch.isfinite(m6_grad.gdn.k_proj.weight.grad).all()

    # 4. Verify Offline Episode Training vs Online `RoboMMEPolicyWrapper` Streaming Parity (`< 1e-5`)!
    mock_enc = _MockSiglipEncoder(device=device, dtype=dtype)
    norm = ActionStateNormalizer(mode="quantile_delta", stats=MME_VLA_NORM_STATS)
    wrapper = RoboMMEPolicyWrapper(
        m6,
        device=device,
        dtype=dtype,
        vision_encoder=mock_enc,  # type: ignore[arg-type]
        normalizer=norm,
    )
    rng_np = np.random.default_rng(99)
    T_ep, exec_start = 53, 20
    front_frames = rng_np.integers(0, 255, size=(T_ep, 256, 256, 3), dtype=np.uint8)
    wrist_frames = rng_np.integers(0, 255, size=(T_ep, 256, 256, 3), dtype=np.uint8)
    raw_states = rng_np.uniform(-0.2, 0.2, size=(T_ep, 8)).astype(np.float32)

    # Compute offline reference states at `step_idx = 20` (`cutoff=20 -> 5 demo frames: 0,4,8,12,16`)
    # and `step_idx = 36` (`cutoff=21 -> 6 frames: 0,4,8,12,16,20`)
    strided_idx = np.arange(0, 24, 4)  # 0, 4, 8, 12, 16, 20
    f_tok_off, w_tok_off = mock_enc.encode_views(front_frames[strided_idx], wrist_frames[strided_idx])
    s_norm_off = torch.from_numpy(norm.normalize_state(raw_states[strided_idx])).to(device=device, dtype=dtype)
    assert m6.slot_resampler is not None
    raw_slots_off = m6.slot_resampler.encode_frames_raw(
        f_tok_off,
        w_tok_off,
        s_norm_off,
        torch.tensor(strided_idx, device=device, dtype=torch.long),
        torch.tensor(strided_idx < exec_start, device=device, dtype=torch.long),
    )
    conv_slots_off = m6.slot_resampler.apply_episode_slot_conv(raw_slots_off)  # (6, N_w, d)
    ref_states = m6.compute_packed_gdn_states(
        [conv_slots_off],
        torch.tensor([0, 0], device=device, dtype=torch.long),
        torch.tensor([5, 6], device=device, dtype=torch.long),
    )

    wrapper.reset()
    wrapper.ingest_demo_frames(
        list(front_frames[:exec_start]),
        list(wrist_frames[:exec_start]),
        list(raw_states[:exec_start]),
        "pick target cube",
    )
    out_c0 = wrapper.predict_chunk(
        front_frames[exec_start],
        wrist_frames[exec_start],
        raw_states[exec_start],
        "pick target cube",
        step_idx=exec_start,
        exec_start_idx=exec_start,
    )
    assert out_c0.actions.shape == (20, 8) and np.isfinite(out_c0.actions).all()
    assert wrapper._S is not None
    stream_diff_c0 = float((wrapper._S[0] - ref_states["S"][0]).abs().max().item())

    # Simulate 15 intermediate steps `21..35`, then predict chunk 1 at `step_idx = 36`!
    wrapper.ingest_step_history(
        list(front_frames[exec_start + 1 : exec_start + 16]),
        list(wrist_frames[exec_start + 1 : exec_start + 16]),
        list(raw_states[exec_start + 1 : exec_start + 16]),
        "pick target cube",
    )
    out_c1 = wrapper.predict_chunk(
        front_frames[exec_start + 16],
        wrist_frames[exec_start + 16],
        raw_states[exec_start + 16],
        "pick target cube",
        step_idx=exec_start + 16,
        exec_start_idx=exec_start,
    )
    assert out_c1.actions.shape == (20, 8)
    stream_diff_c1 = float((wrapper._S[0] - ref_states["S"][1]).abs().max().item())
    max_stream_diff = max(stream_diff_c0, stream_diff_c1)
    assert max_stream_diff < 1e-5, f"Streaming vs Offline GDN state mismatch: {max_stream_diff}"

    # 5. Cycle 3 (`v3_mode=True`, U1–U5) Zero-Init, Long-Demo (`T_demo > 256`), `obs_horizon=2`, & `U3/U4` Parity!
    v3_base = DeltaLoopVLAConfig.v3_preset(
        arch="m0_memoryless",
        d_model=128,
        attn_heads=4,
        d_ff=256,
        ctx_layers=1,
        expert_layers=2,
        gdn_heads=4,
        gdn_d_k=64,
        gdn_d_v=64,
        num_write_slots=8,
    )
    m0_v3 = DeltaLoopVLAPolicy(v3_base).to(device=device, dtype=dtype).eval()
    p_front = torch.randn((B, 64, 768), device=device, dtype=dtype)
    p_wrist = torch.randn((B, 16, 768), device=device, dtype=dtype)
    p_state = torch.randn((B, 8), device=device, dtype=torch.float32)

    ctx0_v3, cmask0_v3 = m0_v3.obs_encoder(
        front, wrist, state, txt, txt_mask, prev_front_emb=p_front, prev_wrist_emb=p_wrist, prev_state_norm=p_state
    )
    v0_v3, g0_v3 = m0_v3.action_expert(
        x_tau, tau, ctx0_v3, cmask0_v3, mem_tokens=None, return_gripper_logits=True
    )
    assert g0_v3 is not None and g0_v3.shape == (B, 16)

    max_v3_zero_init_diff = 0.0
    for arch in ALL_VLA_ARCHS:
        if arch == "m0_memoryless":
            continue
        cfg_v3_a = copy.deepcopy(v3_base)
        cfg_v3_a.arch = arch
        ma_v3 = DeltaLoopVLAPolicy(cfg_v3_a).to(device=device, dtype=dtype).eval()
        ma_v3.obs_encoder.load_state_dict(m0_v3.obs_encoder.state_dict())
        ma_v3.action_expert.load_state_dict(m0_v3.action_expert.state_dict())
        ctx_va, cm_va = ma_v3.obs_encoder(
            front, wrist, state, txt, txt_mask, prev_front_emb=p_front, prev_wrist_emb=p_wrist, prev_state_norm=p_state
        )
        if arch == "m2_framesamp_modul":
            mem_st_v3 = {
                "buf_tokens": torch.randn((B, 32, 128), device=device, dtype=dtype),
                "buf_mask": torch.tensor(
                    [[False] * 32, [True] * 8 + [False] * 24, [True] * 32],
                    device=device,
                    dtype=torch.bool,
                ),
            }
        elif arch in ("m5_gdn2layer_b2_l1", "m7_deltaloop_2l_b2"):
            mem_st_v3 = {
                "S1": torch.randn((B, 4, 32, 64), device=device, dtype=torch.float32),
                "occ1": torch.randn((B, 4, 64), device=device, dtype=torch.float32),
                "S2": torch.randn((B, 4, 32, 64), device=device, dtype=torch.float32),
                "occ2": torch.randn((B, 4, 64), device=device, dtype=torch.float32),
            }
        elif ma_v3.uses_gdn:
            mem_st_v3 = {
                "S": torch.randn((B, 4, 64, 64), device=device, dtype=torch.float32),
                "occ": torch.randn((B, 4, 64), device=device, dtype=torch.float32),
            }
        else:
            mem_st_v3 = None
        ro_v3 = ma_v3.run_coda_readout(ctx_va, cm_va, mem_state=mem_st_v3)
        va_v3, ga_v3 = ma_v3.action_expert(
            x_tau, tau, ctx_va, cm_va, mem_tokens=ro_v3["coda"], mem_mask=ro_v3["mem_mask"], return_gripper_logits=True
        )
        assert ga_v3 is not None
        diff_v3 = max(
            float((va_v3 - v0_v3).abs().max().item()),
            float((ga_v3 - g0_v3).abs().max().item()),
        )
        max_v3_zero_init_diff = max(max_v3_zero_init_diff, diff_v3)
        assert diff_v3 < 1e-6, f"v3 zero-init parity failed on {arch}: {diff_v3}"

    # Verify v3 Long-Demo (`T_demo=280 > 256` -> `demo_stride=8`, `exec_stride=4`) + `obs_horizon=2` + `(20,8)` norm!
    cfg_m6_v3 = copy.deepcopy(v3_base)
    cfg_m6_v3.arch = "m6_deltaloop_b2"
    m6_v3 = DeltaLoopVLAPolicy(cfg_m6_v3).to(device=device, dtype=dtype).eval()
    q01_20x8 = np.tile(np.asarray(MME_VLA_NORM_STATS["actions"]["q01"], dtype=np.float32), (20, 1))
    q99_20x8 = np.tile(np.asarray(MME_VLA_NORM_STATS["actions"]["q99"], dtype=np.float32), (20, 1))
    stats_v3 = {
        "state": MME_VLA_NORM_STATS["state"],
        "actions": {"q01": q01_20x8.tolist(), "q99": q99_20x8.tolist()},
    }
    norm_v3 = ActionStateNormalizer(mode="quantile_delta", stats=stats_v3)
    wrapper_v3 = RoboMMEPolicyWrapper(
        m6_v3,
        device=device,
        dtype=dtype,
        vision_encoder=mock_enc,  # type: ignore[arg-type]
        normalizer=norm_v3,
    )
    T_long, exec_long = 305, 280
    f_long = rng_np.integers(0, 255, size=(T_long, 256, 256, 3), dtype=np.uint8)
    w_long = rng_np.integers(0, 255, size=(T_long, 256, 256, 3), dtype=np.uint8)
    s_long = rng_np.uniform(-0.2, 0.2, size=(T_long, 8)).astype(np.float32)

    w_idx_c1, n_demo_c1 = m6_v3.compute_written_frame_indices(281, exec_long, cfg_m6_v3)
    assert n_demo_c1 == 35 and int(w_idx_c1.numel()) == 36  # 0..272 step 8 (35) + exec 280 (1)
    f_tok_v3, w_tok_v3 = mock_enc.encode_views(f_long[w_idx_c1.numpy()], w_long[w_idx_c1.numpy()])
    s_norm_v3 = torch.from_numpy(norm_v3.normalize_state(s_long[w_idx_c1.numpy()])).to(device=device, dtype=dtype)
    assert m6_v3.slot_resampler is not None
    raw_slots_v3 = m6_v3.slot_resampler.encode_frames_raw(
        f_tok_v3,
        w_tok_v3,
        s_norm_v3,
        w_idx_c1.to(device),
        (w_idx_c1 < exec_long).long().to(device),
    )
    conv_slots_v3 = m6_v3.slot_resampler.apply_episode_slot_conv(raw_slots_v3)
    mot_rms_v3 = m6_v3.compute_visual_motion_rms(f_tok_v3)
    ref_v3 = m6_v3.compute_packed_gdn_states(
        [conv_slots_v3],
        torch.tensor([0, 0], device=device, dtype=torch.long),
        torch.tensor([35, 36], device=device, dtype=torch.long),
        episode_motion_seqs=[mot_rms_v3],
    )

    wrapper_v3.reset()
    wrapper_v3.ingest_demo_frames(
        list(f_long[:exec_long]), list(w_long[:exec_long]), list(s_long[:exec_long]), "place cubes in order"
    )
    out_v3_c0 = wrapper_v3.predict_chunk(
        f_long[exec_long],
        w_long[exec_long],
        s_long[exec_long],
        "place cubes in order",
        step_idx=exec_long,
        exec_start_idx=exec_long,
    )
    assert out_v3_c0.actions.shape == (20, 8) and np.isfinite(out_v3_c0.actions).all()
    assert wrapper_v3._S is not None
    v3_diff_c0 = float((wrapper_v3._S[0] - ref_v3["S"][0]).abs().max().item())

    wrapper_v3.ingest_step_history(
        list(f_long[exec_long + 1 : exec_long + 16]),
        list(w_long[exec_long + 1 : exec_long + 16]),
        list(s_long[exec_long + 1 : exec_long + 16]),
        "place cubes in order",
    )
    # Verify `obs_horizon=2` cached exact step `exec_long + 15`!
    f_exp_prev, _ = mock_enc.encode_views(f_long[exec_long + 15], w_long[exec_long + 15])
    assert wrapper_v3._prev_exec_obs is not None
    obs_h2_diff = float((wrapper_v3._prev_exec_obs[0] - f_exp_prev).abs().max().item())
    assert obs_h2_diff < 1e-6, f"obs_horizon=2 prev frame mismatch: {obs_h2_diff}"

    out_v3_c1 = wrapper_v3.predict_chunk(
        f_long[exec_long + 16],
        w_long[exec_long + 16],
        s_long[exec_long + 16],
        "place cubes in order",
        step_idx=exec_long + 16,
        exec_start_idx=exec_long,
    )
    assert out_v3_c1.actions.shape == (20, 8)
    v3_diff_c1 = float((wrapper_v3._S[0] - ref_v3["S"][1]).abs().max().item())
    v3_stream_max_diff = max(v3_diff_c0, v3_diff_c1)
    assert v3_stream_max_diff < 1e-5, f"v3 Long-Demo Streaming vs Offline mismatch: {v3_stream_max_diff}"

    # Verify U3 -> U4 sub-grid affine target transform & `predict_grounding_xy` grad flow
    with torch.enable_grad():
        m6_v3_tr = DeltaLoopVLAPolicy(cfg_m6_v3).to(device=device, dtype=dtype).train()
        th_aug = sample_subgrid_crop_affine(B, scale_range=(0.90, 1.0), max_extra_shift=0.04, device=device)
        xy_src = torch.tensor([[0.0, 0.0], [0.25, -0.35], [-0.4, 0.5]], device=device, dtype=torch.float32)
        has_xy = torch.tensor([True, True, False], device=device, dtype=torch.bool)
        xy_crop, valid_crop = transform_coords_under_affine(xy_src, has_xy, th_aug)
        ctx_tr, cm_tr = m6_v3_tr.obs_encoder(
            front, wrist, state, txt, txt_mask, prev_front_emb=p_front, prev_wrist_emb=p_wrist, prev_state_norm=p_state, aug_theta=th_aug
        )
        ro_tr = m6_v3_tr.run_coda_readout(
            ctx_tr,
            cm_tr,
            mem_state={
                "S": torch.zeros((B, 4, 64, 64), device=device, dtype=torch.float32),
                "occ": torch.zeros((B, 4, 64), device=device, dtype=torch.float32),
            },
        )
        pxy = m6_v3_tr.predict_grounding_xy(ctx_tr, ro_tr["coda"])
        assert pxy is not None and pxy.shape == (B, 2)
        sg_tgt = F.normalize(torch.randn((B, cfg_m6_v3.subgoal_dim), device=device, dtype=torch.float32), dim=-1)
        lo_tr = m6_v3_tr.compute_crn_flow_and_aux_loss(
            ctx_tr,
            cm_tr,
            ro_tr["coda_per_depth"],
            None,
            torch.randn((B, 20, 8), device=device, dtype=torch.float32),
            torch.randn((B, 49), device=device, dtype=torch.float32),
            grounding_xy_target=xy_crop,
            grounding_xy_mask=valid_crop,
            subgoal_emb_target=sg_tgt,
            subgoal_emb_mask=torch.tensor([True, True, False], device=device, dtype=torch.bool),
        )
        lo_tr["loss"].backward()
        assert torch.isfinite(lo_tr["loss"]) and float(lo_tr["ground_loss"].item()) > 0.0

    # 6. Verify `m7_deltaloop_2l_b2` iso-state (`528,384 B` at `K=128, V=128`), `L=4` readout,
    #    online `RoboMMEPolicyWrapper` 2-layer streaming parity, and `K=128` `V<=64` `fla` chunked backward!
    cfg_m7_128 = DeltaLoopVLAConfig.v3_preset(
        arch="m7_deltaloop_2l_b2",
        d_model=128,
        attn_heads=4,
        d_ff=256,
        ctx_layers=1,
        expert_layers=2,
        gdn_heads=8,
        gdn_d_k=128,
        gdn_d_v=128,
        num_write_slots=8,
    )
    cfg_m6_128 = copy.deepcopy(cfg_m7_128)
    cfg_m6_128.arch = "m6_deltaloop_b2"
    # Test exact mathematical 2-layer streaming parity in IEEE fp32 (eliminates cuBLAS bf16 tile-split ULP noise),
    # plus verify bf16 wrapper rollout stays within `< 2e-3` (`< 1 bf16 ULP`).
    # Disable both cuBLAS and cuDNN TF32 during exact IEEE fp32 mathematical parity checks
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32 if device.type == "cuda" else False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        m7_128 = DeltaLoopVLAPolicy(cfg_m7_128).to(device=device, dtype=torch.float32).eval()
        m6_128 = DeltaLoopVLAPolicy(cfg_m6_128).to(device=device, dtype=dtype).eval()
        assert m7_128.gdn1 is not None and m7_128.gdn2 is not None and m6_128.gdn is not None
        assert (
            m7_128.gdn1.state_numel + m7_128.gdn2.state_numel == m6_128.gdn.state_numel == 131072
        ), "S1 + S2 matrix state must equal 1-layer S state (131,072 fp32 = 512.0 KB)!"
        assert m7_128.state_size_bytes() == 532480, f"Expected 532,480 B (512KB S + 8KB occ) for m7, got {m7_128.state_size_bytes()}"
        assert m7_128.halting_gate is not None and m7_128.is_multi_depth

        mock_enc_f32 = _MockSiglipEncoder(device=device, dtype=torch.float32)
        f_tok_v3_f32 = f_tok_v3.float()
        w_tok_v3_f32 = w_tok_v3.float()
        s_norm_v3_f32 = s_norm_v3.float()
        mot_rms_v3_f32 = m7_128.compute_visual_motion_rms(f_tok_v3_f32)
        conv_m7_f32 = m7_128.slot_resampler.apply_episode_slot_conv(  # type: ignore[union-attr]
            m7_128.slot_resampler.encode_frames_raw(     # type: ignore[union-attr]
                f_tok_v3_f32, w_tok_v3_f32, s_norm_v3_f32, w_idx_c1.to(device), (w_idx_c1 < exec_long).long().to(device)
            )
        )
        ref_m7_pt = m7_128.compute_packed_gdn_states(
            [conv_m7_f32],
            torch.tensor([0, 0], device=device, dtype=torch.long),
            torch.tensor([35, 36], device=device, dtype=torch.long),
            episode_motion_seqs=[mot_rms_v3_f32],
            force_pytorch=True,
        )
        wrapper_m7 = RoboMMEPolicyWrapper(
            m7_128,
            device=device,
            dtype=torch.float32,
            vision_encoder=mock_enc_f32,  # type: ignore[arg-type]
            normalizer=norm_v3,
            ode_steps=2,
            force_pytorch_gdn=True,
        )
        wrapper_m7.reset()
        wrapper_m7.ingest_demo_frames(
            list(f_long[:exec_long]), list(w_long[:exec_long]), list(s_long[:exec_long]), "place cubes in order"
        )
        out_m7_c0 = wrapper_m7.predict_chunk(
            f_long[exec_long],
            w_long[exec_long],
            s_long[exec_long],
            "place cubes in order",
            step_idx=exec_long,
            exec_start_idx=exec_long,
        )
        assert out_m7_c0.actions.shape == (20, 8) and np.isfinite(out_m7_c0.actions).all()
        assert wrapper_m7._S1 is not None and wrapper_m7._S2 is not None
        m7_stream_diff = max(
            float((wrapper_m7._S1[0] - ref_m7_pt["S1"][0]).abs().max().item()),
            float((wrapper_m7._S2[0] - ref_m7_pt["S2"][0]).abs().max().item()),
        )
        assert m7_stream_diff < 2e-4, f"m7 2-layer exact streaming parity failed: {m7_stream_diff}"
        if HAS_FLA and device.type == "cuda":
            ref_m7_fla = m7_128.compute_packed_gdn_states(
                [conv_m7_f32],
                torch.tensor([0, 0], device=device, dtype=torch.long),
                torch.tensor([35, 36], device=device, dtype=torch.long),
                episode_motion_seqs=[mot_rms_v3_f32],
                force_pytorch=False,
            )
            m7_fla_vs_pt = max(
                float((ref_m7_fla["S1"][0] - ref_m7_pt["S1"][0]).abs().max().item()),
                float((ref_m7_fla["S2"][0] - ref_m7_pt["S2"][0]).abs().max().item()),
            )
            assert m7_fla_vs_pt < 5e-4, f"m7 2-layer fla WGMMA vs PyTorch exceeded 5e-4: {m7_fla_vs_pt}"
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    if HAS_FLA and device.type == "cuda":
        ep_128 = [
            torch.randn((18, 8, 128), device=device, dtype=dtype),
            torch.randn((22, 8, 128), device=device, dtype=dtype),
        ]
        mot_128 = [torch.rand((18, 1), device=device), torch.rand((22, 1), device=device)]
        s_ep128 = torch.tensor([0, 1], device=device, dtype=torch.long)
        s_wc128 = torch.tensor([15, 20], device=device, dtype=torch.long)
        st128_fla = m6_128.compute_packed_gdn_states(ep_128, s_ep128, s_wc128, episode_motion_seqs=mot_128, force_pytorch=False)
        st128_pt = m6_128.compute_packed_gdn_states(ep_128, s_ep128, s_wc128, episode_motion_seqs=mot_128, force_pytorch=True)
        d128 = max(
            float((st128_fla["S"] - st128_pt["S"]).abs().max().item()),
            float((st128_fla["occ"] - st128_pt["occ"]).abs().max().item()),
        )
        fla_vs_pt_diff = max(fla_vs_pt_diff, d128)
        assert d128 < 1e-5, f"K=128 V<=64 chunked fla vs PyTorch parity failed: {d128}"
        with torch.enable_grad():
            m6_128_g = DeltaLoopVLAPolicy(cfg_m6_128).to(device=device, dtype=dtype).train()
            m7_128_g = DeltaLoopVLAPolicy(cfg_m7_128).to(device=device, dtype=dtype).train()
            ep_128_g = [x.detach().clone().requires_grad_(True) for x in ep_128]
            st6_g = m6_128_g.compute_packed_gdn_states(ep_128_g, s_ep128, s_wc128, episode_motion_seqs=mot_128, force_pytorch=False)
            (st6_g["S"].square().mean() + st6_g["occ"].square().mean()).backward()
            assert ep_128_g[0].grad is not None and torch.isfinite(ep_128_g[0].grad).all()
            ep_128_g2 = [x.detach().clone().requires_grad_(True) for x in ep_128]
            st7_g = m7_128_g.compute_packed_gdn_states(ep_128_g2, s_ep128, s_wc128, episode_motion_seqs=mot_128, force_pytorch=False)
            (st7_g["S1"].square().mean() + st7_g["S2"].square().mean()).backward()
            assert ep_128_g2[0].grad is not None and torch.isfinite(ep_128_g2[0].grad).all()

    report = {
        "zero_init_max_diff": max_zero_init_diff,
        "v3_zero_init_max_diff": max_v3_zero_init_diff,
        "fla_vs_pytorch_max_diff": fla_vs_pt_diff,
        "streaming_vs_offline_max_diff": max_stream_diff,
        "v3_long_demo_streaming_max_diff": max(v3_stream_max_diff, m7_stream_diff),
    }
    print(f"[Self-Test] ALL PASSED: {json.dumps(report)}")
    return report


# =============================================================================
# 2. High-Throughput GPU-Resident Multi-Task Episode Store (`0` I/O Wait!)
# =============================================================================


class _MmapShardConcat:
    """Zero-RSS virtual concatenation over mmap'd task shards (safe inside 25GB MIG cgroups!)."""

    def __init__(self, tensors: List[torch.Tensor], target_dtype: torch.dtype) -> None:
        self.tensors = tensors
        self.target_dtype = target_dtype
        offs = [0]
        for t in tensors:
            offs.append(offs[-1] + int(t.shape[0]))
        self.boundaries = torch.tensor(offs[1:], dtype=torch.long)
        self.starts = offs[:-1]
        self.tail_shape = tuple(tensors[0].shape[1:])

    def __getitem__(self, rows: torch.Tensor) -> torch.Tensor:
        r_cpu = rows.to(device="cpu", dtype=torch.long)
        n = int(r_cpu.numel())
        if n == 0:
            return torch.empty((0, *self.tail_shape), dtype=self.target_dtype)
        shard_ids = torch.bucketize(r_cpu, self.boundaries, right=True)
        out = torch.empty(
            (n, *self.tail_shape),
            dtype=self.target_dtype,
            pin_memory=torch.cuda.is_available(),
        )
        for s_i, sh_t in enumerate(self.tensors):
            mask = shard_ids == s_i
            if bool(mask.any()):
                local_r = r_cpu[mask] - self.starts[s_i]
                out[mask] = sh_t[local_r].to(dtype=self.target_dtype)
        return out


class GPURoboMMEStore:
    """Holds multi-task SigLIP shards in GPU HBM3e (or zero-RSS mmap on CPU/MIG) for fast sampling."""

    def __init__(
        self,
        cache_dir: str | Path,
        tasks: Sequence[str],
        *,
        device: torch.device,
        feature_dtype: torch.dtype = torch.bfloat16,
        preload_gpu: bool = True,
        v3_mode: bool = False,
        action_horizon: int = 20,
        subgoal_dim: int = 64,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.tasks = tuple(tasks)
        self.device = device
        self.feature_dtype = feature_dtype
        self.v3_mode = bool(v3_mode)
        self.action_horizon = int(action_horizon)
        self.subgoal_dim = int(subgoal_dim)
        self.normalizer = load_cache_normalizer(self.cache_dir, action_mode="quantile_delta")

        t0 = time.perf_counter()
        raw_shards = load_task_shards(self.cache_dir, tasks=self.tasks, mmap=True)
        mmap_s = time.perf_counter() - t0

        # Estimate total tensor footprint to decide GPU HBM3e residency safely
        total_frames = sum(int(raw_shards[t]["front_emb"].shape[0]) for t in self.tasks)
        est_gb = total_frames * 80 * 768 * 2 / (1024**3)
        store_dev = torch.device("cpu")
        if preload_gpu and device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if est_gb * 1.15 < (free_bytes / (1024**3)):
                store_dev = device
        self.store_device = store_dev

        # Deterministic fixed projection for `U4` subgoal language alignment (`44-D -> subgoal_dim`)
        sg_in_dim = len(STAGE_VOCAB) + len(COLOR_VOCAB) + 11 + 9
        sg_gen = torch.Generator()
        sg_gen.manual_seed(20261112)
        sg_proj_mat = torch.randn((sg_in_dim, self.subgoal_dim), generator=sg_gen, dtype=torch.float32) / math.sqrt(
            sg_in_dim
        )

        t1 = time.perf_counter()
        front_list, wrist_list, raw_state_list, raw_act_list = [], [], [], []
        ep_starts, ep_ends, ep_exec_starts, ep_task_ids = [], [], [], []
        txt_tok_list, txt_mask_list = [], []
        gt_xy_list, gt_has_xy_list, gt_sg_list, gt_has_sg_list = [], [], [], []
        self.task_to_ep_ids: List[torch.Tensor] = []

        frame_offset = 0
        ep_counter = 0
        for t_idx, task in enumerate(self.tasks):
            sh = raw_shards[task]
            nf = int(sh["front_emb"].shape[0])
            n_eps = int(sh["epis_idx"].shape[0])

            if store_dev.type == "cuda":
                front_list.append(sh["front_emb"].to(dtype=feature_dtype, device=store_dev))
                wrist_list.append(sh["wrist_emb"].to(dtype=feature_dtype, device=store_dev))
            else:
                # Keep mmap views untouched so CPU RSS stays <1.5 GiB inside 25 GiB MIG cgroups!
                front_list.append(sh["front_emb"])
                wrist_list.append(sh["wrist_emb"])
            raw_state_list.append(sh["state"].to(dtype=torch.float32, device=store_dev))
            raw_act_list.append(sh["actions"].to(dtype=torch.float32, device=store_dev))

            offs = sh["ep_offsets"].to(dtype=torch.long)
            ex_s = sh["exec_start_idx"].to(dtype=torch.long)
            ep_starts.append(offs[:-1] + frame_offset)
            ep_ends.append(offs[1:] + frame_offset)
            ep_exec_starts.append(ex_s)
            ep_task_ids.append(torch.full((n_eps,), t_idx, dtype=torch.long))

            txt_tok_list.append(sh["text_token_emb"].to(dtype=feature_dtype, device=store_dev))
            txt_mask_list.append(sh["text_attn_mask"].to(dtype=torch.bool, device=store_dev))

            if self.v3_mode:
                g_parsed = parse_grounded_subgoals(
                    sh.get("grounded_subgoals", [""] * nf), ep_offsets=offs, resolve_complete=True
                )
                sg_meta = extract_subgoal_metadata(
                    sh.get("simple_subgoals", [""] * nf), ep_offsets=offs, exec_start_idx=ex_s
                )
                st_oh = F.one_hot(sg_meta["stage_id"].clamp(0, len(STAGE_VOCAB) - 1), len(STAGE_VOCAB)).float()
                co_oh = F.one_hot(sg_meta["color_id"].clamp(0, len(COLOR_VOCAB) - 1), len(COLOR_VOCAB)).float()
                or_oh = F.one_hot(sg_meta["ordinal_index"].clamp(0, 10), 11).float()
                pk_oh = F.one_hot(sg_meta["completed_picks"].clamp(0, 8), 9).float()
                sg_raw = torch.cat([st_oh, co_oh, or_oh, pk_oh], dim=-1) @ sg_proj_mat
                has_sg = sg_meta["stage_id"] != 0
                sg_norm = F.normalize(sg_raw, p=2.0, dim=-1) * has_sg.float().unsqueeze(-1)

                gt_xy_list.append(g_parsed["xy_norm"].to(dtype=torch.float32, device=store_dev))
                gt_has_xy_list.append(g_parsed["has_xy"].to(dtype=torch.bool, device=store_dev))
                gt_sg_list.append(sg_norm.to(dtype=torch.float32, device=store_dev))
                gt_has_sg_list.append(has_sg.to(dtype=torch.bool, device=store_dev))

            self.task_to_ep_ids.append(
                torch.arange(ep_counter, ep_counter + n_eps, dtype=torch.long, device=store_dev)
            )
            frame_offset += nf
            ep_counter += n_eps

        del raw_shards
        if store_dev.type == "cuda":
            self.front_emb: Any = torch.cat(front_list, dim=0)
            self.wrist_emb: Any = torch.cat(wrist_list, dim=0)
        else:
            self.front_emb = _MmapShardConcat(front_list, target_dtype=feature_dtype)
            self.wrist_emb = _MmapShardConcat(wrist_list, target_dtype=feature_dtype)
        self.raw_state = torch.cat(raw_state_list, dim=0)
        self.raw_actions = torch.cat(raw_act_list, dim=0)

        self.ep_starts = torch.cat(ep_starts, dim=0).to(store_dev)
        self.ep_ends = torch.cat(ep_ends, dim=0).to(store_dev)
        self.ep_exec_starts = torch.cat(ep_exec_starts, dim=0).to(store_dev)
        self.ep_task_ids = torch.cat(ep_task_ids, dim=0).to(store_dev)
        self.text_token_emb = torch.cat(txt_tok_list, dim=0)
        self.text_attn_mask = torch.cat(txt_mask_list, dim=0)

        if self.v3_mode:
            self.gt_xy_norm: Optional[torch.Tensor] = torch.cat(gt_xy_list, dim=0)
            self.gt_has_xy: Optional[torch.Tensor] = torch.cat(gt_has_xy_list, dim=0)
            self.gt_sg_emb: Optional[torch.Tensor] = torch.cat(gt_sg_list, dim=0)
            self.gt_has_sg: Optional[torch.Tensor] = torch.cat(gt_has_sg_list, dim=0)
        else:
            self.gt_xy_norm = None
            self.gt_has_xy = None
            self.gt_sg_emb = None
            self.gt_has_sg = None

        # Precompute normalized state on `store_dev`
        s_low = torch.tensor(self.normalizer.state_low, device=store_dev, dtype=torch.float32)
        s_high = torch.tensor(self.normalizer.state_high, device=store_dev, dtype=torch.float32)
        self.state_norm = (self.raw_state - s_low) / (s_high - s_low + 1e-6) * 2.0 - 1.0

        self.delta_mask = torch.tensor(DELTA_JOINT_MASK, device=device, dtype=torch.bool)
        if self.v3_mode:
            # Cycle 3 `U1`: Compute exact per-horizon `(20, 8)` `q01[h, d], q99[h, d]` over execution frames!
            exec_row_segs = []
            exec_end_segs = []
            s_cpu = self.ep_starts.cpu().tolist()
            e_cpu = self.ep_ends.cpu().tolist()
            x_cpu = self.ep_exec_starts.cpu().tolist()
            for st_e, en_e, ex_e in zip(s_cpu, e_cpu, x_cpu):
                r_start = st_e + ex_e
                if r_start < en_e:
                    r_seg = torch.arange(r_start, en_e, dtype=torch.long, device=store_dev)
                    exec_row_segs.append(r_seg)
                    exec_end_segs.append(torch.full_like(r_seg, en_e - 1))
            all_exec_rows = torch.cat(exec_row_segs, dim=0)
            all_exec_ends = torch.cat(exec_end_segs, dim=0)
            # Subsample at most 160k execution rows (`SE < 1e-4`, `< 50 MB` temp memory, `< 0.08s`!)
            stride_q = max(1, int(all_exec_rows.numel()) // 160000)
            q_rows = all_exec_rows[::stride_q]
            q_ends = all_exec_ends[::stride_q]
            h_grid = torch.arange(self.action_horizon, device=store_dev, dtype=torch.long).unsqueeze(0)
            g_rows = torch.minimum(q_rows.unsqueeze(1) + h_grid, q_ends.unsqueeze(1))
            raw_ch = self.raw_actions[g_rows].float()  # (N_q, 20, 8)
            st_ch = self.raw_state[q_rows].float().unsqueeze(1)  # (N_q, 1, 8)
            dm_store = torch.tensor(DELTA_JOINT_MASK, device=store_dev, dtype=torch.bool).view(1, 1, 8)
            delta_ch = raw_ch - torch.where(dm_store, st_ch, torch.zeros_like(st_ch))

            q01_hd = torch.quantile(delta_ch, 0.01, dim=0)  # (20, 8)
            q99_hd = torch.quantile(delta_ch, 0.99, dim=0)  # (20, 8)
            q99_hd = torch.maximum(q99_hd, q01_hd + 1e-3)
            q01_hd[:, 7] = -1.0
            q99_hd[:, 7] = 1.0
            self.act_low = q01_hd.to(device=device, dtype=torch.float32)
            self.act_high = q99_hd.to(device=device, dtype=torch.float32)
            self.action_q01_export: Any = q01_hd.detach().cpu().tolist()
            self.action_q99_export: Any = q99_hd.detach().cpu().tolist()
        else:
            self.act_low = torch.tensor(self.normalizer.action_low, device=device, dtype=torch.float32)
            self.act_high = torch.tensor(self.normalizer.action_high, device=device, dtype=torch.float32)
            self.action_q01_export = self.normalizer.action_low.tolist()
            self.action_q99_export = self.normalizer.action_high.tolist()

        self.num_episodes = ep_counter
        self.num_frames = frame_offset
        load_s = time.perf_counter() - t1
        print(
            f"[Dataset] Loaded {len(self.tasks)} tasks ({self.num_episodes} eps, {self.num_frames:,} frames, "
            f"~{est_gb:.2f} GiB, v3_mode={self.v3_mode}) onto `{store_dev}` in {load_s:.2f}s (mmap={mmap_s*1000:.1f}ms)."
        )

    def encode_action_chunks_gpu(
        self,
        ep_ids: torch.Tensor,       # (B,) on store_device
        step_in_ep: torch.Tensor,   # (B,) on store_device
        action_horizon: int,
    ) -> torch.Tensor:
        """Vectorized GPU target chunk builder matching `build_padded_action_chunk` 100% identically."""
        starts = self.ep_starts[ep_ids]
        ends = self.ep_ends[ep_ids]
        abs_step = starts + step_in_ep  # (B,)
        h_grid = torch.arange(action_horizon, device=self.store_device, dtype=torch.long).unsqueeze(0)
        gather_rows = (abs_step.unsqueeze(1) + h_grid).clamp(max=(ends - 1).unsqueeze(1))  # (B, H_act)

        raw_chunks = self.raw_actions[gather_rows].to(self.device)  # (B, H_act, 8)
        curr_raw_st = self.raw_state[abs_step].to(self.device)      # (B, 8)
        offset = torch.where(self.delta_mask, curr_raw_st, torch.zeros_like(curr_raw_st)).unsqueeze(1)
        delta_chunks = raw_chunks - offset
        return (delta_chunks - self.act_low) / (self.act_high - self.act_low + 1e-6) * 2.0 - 1.0

    def compute_aux_targets(
        self,
        ep_ids: torch.Tensor,        # (B,)
        step_in_ep: torch.Tensor,    # (B,)
        cutoff_excl: torch.Tensor,   # (B,)
        vis_anchor_proj: torch.Tensor,  # (768, 32) on model device
    ) -> torch.Tensor:
        """Build 49-D privilege-free self-supervised trajectory anchor targets `(B, 49)`."""
        starts = self.ep_starts[ep_ids]
        ends = self.ep_ends[ep_ids]
        exec_s = self.ep_exec_starts[ep_ids]
        ep_lens = ends - starts

        # 1. Execution progress in [0, 1] (1-D)
        denom = (ep_lens - 1 - exec_s).clamp(min=1).float()
        progress = ((step_in_ep - exec_s).clamp(min=0).float() / denom).unsqueeze(-1).to(self.device)

        # 2. Pre-cutoff anchor normalized state (8-D, zero if cutoff_excl == 0)
        has_hist = (cutoff_excl > 0).to(self.device).float().unsqueeze(-1)
        safe_cut_row = starts + (cutoff_excl - 1).clamp(min=0)
        anchor_st = self.state_norm[safe_cut_row].to(self.device).clamp(-3.0, 3.0) * has_hist

        # 3. Cumulative execution state displacement (8-D)
        curr_st = self.state_norm[starts + step_in_ep].to(self.device)
        start_st = self.state_norm[starts + exec_s.clamp(max=ep_lens - 1)].to(self.device)
        delta_st = (curr_st - start_st).clamp(-3.0, 3.0)

        # 4. Earliest memory frame (frame 0) pooled visual anchor (32-D)
        f0_mean = self.front_emb[starts].to(self.device).float().mean(dim=1)  # (B, 768)
        vis_anc = F.normalize(f0_mean @ vis_anchor_proj.float(), p=2.0, dim=-1) * has_hist

        return torch.cat([progress, anchor_st, delta_st, vis_anc], dim=-1)  # (B, 49)

    def sample_balanced_batch(
        self,
        eps_per_task: int,
        chunks_per_ep: int,
        cfg: DeltaLoopVLAConfig,
        vis_anchor_proj: torch.Tensor,
        gen: torch.Generator,
    ) -> Dict[str, Any]:
        """Sample `B_ep = len(tasks) * eps_per_task` episodes and `chunks_per_ep` random steps per episode."""
        chosen_eps = []
        for t_ep_ids in self.task_to_ep_ids:
            perm = torch.randint(
                0, t_ep_ids.numel(), (eps_per_task,), generator=gen, device="cpu"
            ).to(self.store_device)
            chosen_eps.append(t_ep_ids[perm])
        ep_ids_bep = torch.cat(chosen_eps, dim=0)  # (B_ep,)
        B_ep = int(ep_ids_bep.numel())

        starts_bep = self.ep_starts[ep_ids_bep]
        ends_bep = self.ep_ends[ep_ids_bep]
        exec_bep = self.ep_exec_starts[ep_ids_bep]
        lens_bep = ends_bep - starts_bep
        exec_lens_bep = (lens_bep - exec_bep).clamp(min=1)

        # Sample `chunks_per_ep` execution steps per episode and sort ascending within each episode
        u_rand = torch.rand((B_ep, chunks_per_ep), generator=gen, device="cpu").to(self.store_device)
        rel_steps = (u_rand * exec_lens_bep.unsqueeze(1).float()).long()
        rel_steps = torch.minimum(rel_steps, (exec_lens_bep - 1).unsqueeze(1))
        steps_mat, _ = torch.sort(exec_bep.unsqueeze(1) + rel_steps, dim=1)  # (B_ep, M)

        raw_cut = (steps_mat - cfg.exclusion_window + 1).clamp(min=0)
        cutoffs_mat = torch.maximum(exec_bep.unsqueeze(1), raw_cut)  # (B_ep, M)

        # Flatten `(B_ep, M) -> (B,)` where `B = B_ep * M`
        B = B_ep * chunks_per_ep
        sample_local_ep = torch.arange(B_ep, device=self.device, dtype=torch.long).repeat_interleave(chunks_per_ep)
        sample_global_ep = ep_ids_bep.repeat_interleave(chunks_per_ep)
        sample_steps = steps_mat.reshape(B)
        sample_cutoffs = cutoffs_mat.reshape(B)

        abs_rows = self.ep_starts[sample_global_ep] + sample_steps
        curr_front = self.front_emb[abs_rows].to(self.device, non_blocking=True)
        curr_wrist = self.wrist_emb[abs_rows].to(self.device, non_blocking=True)
        curr_state = self.state_norm[abs_rows].to(self.device, non_blocking=True)
        txt_tok = self.text_token_emb[sample_global_ep].to(self.device, non_blocking=True)
        txt_mask = self.text_attn_mask[sample_global_ep].to(self.device, non_blocking=True)

        target_actions = self.encode_action_chunks_gpu(sample_global_ep, sample_steps, cfg.action_horizon)
        aux_targets = self.compute_aux_targets(sample_global_ep, sample_steps, sample_cutoffs, vis_anchor_proj)

        prev_front = None
        prev_wrist = None
        prev_state = None
        if cfg.obs_horizon2:
            prev_steps = torch.maximum(self.ep_exec_starts[sample_global_ep], sample_steps - 1)
            prev_rows = self.ep_starts[sample_global_ep] + prev_steps
            prev_front = self.front_emb[prev_rows].to(self.device, non_blocking=True)
            prev_wrist = self.wrist_emb[prev_rows].to(self.device, non_blocking=True)
            prev_state = self.state_norm[prev_rows].to(self.device, non_blocking=True)

        gt_xy = self.gt_xy_norm[abs_rows].to(self.device, non_blocking=True) if self.gt_xy_norm is not None else None
        gt_has_xy = self.gt_has_xy[abs_rows].to(self.device, non_blocking=True) if self.gt_has_xy is not None else None
        gt_sg = self.gt_sg_emb[abs_rows].to(self.device, non_blocking=True) if self.gt_sg_emb is not None else None
        gt_has_sg = self.gt_has_sg[abs_rows].to(self.device, non_blocking=True) if self.gt_has_sg is not None else None

        ws = cfg.write_stride
        starts_cpu = starts_bep.tolist()
        exec_cpu = exec_bep.tolist()
        ep_frame_rows = []
        ep_step_indices = []
        ep_is_demo = []
        ep_w_counts = []

        if not cfg.v3_mode:
            sample_written_counts = ((sample_cutoffs + ws - 1) // ws).to(self.device)
            max_cut_per_ep = cutoffs_mat[:, -1]  # (B_ep,)
            ep_max_written = ((max_cut_per_ep + ws - 1) // ws).tolist()
            for e_i in range(B_ep):
                w_cnt = int(ep_max_written[e_i])
                ep_w_counts.append(w_cnt)
                if w_cnt > 0:
                    rel_idx = torch.arange(0, w_cnt * ws, ws, device=self.store_device, dtype=torch.long)
                    ep_frame_rows.append(starts_cpu[e_i] + rel_idx)
                    ep_step_indices.append(rel_idx)
                    ep_is_demo.append(rel_idx < exec_cpu[e_i])
            demo_strided_per_sample = ((self.ep_exec_starts[sample_global_ep] + ws - 1) // ws).to(self.device)
        else:
            # Cycle 3 `U5`: Exact adaptive demo striding (`stride=8` if `exec_start > 256` else `4`)
            ds_vec = torch.where(
                exec_bep > cfg.adaptive_demo_stride_threshold,
                torch.full_like(exec_bep, cfg.long_demo_stride),
                torch.full_like(exec_bep, ws),
            )  # (B_ep,)
            demo_lim_mat = torch.minimum(cutoffs_mat, exec_bep.unsqueeze(1))
            w_demo_mat = (demo_lim_mat + ds_vec.unsqueeze(1) - 1) // ds_vec.unsqueeze(1)
            exec_rem_mat = (cutoffs_mat - exec_bep.unsqueeze(1)).clamp(min=0)
            w_exec_mat = (exec_rem_mat + ws - 1) // ws
            sample_written_counts = (w_demo_mat + w_exec_mat).reshape(B).to(self.device)
            demo_strided_per_sample = w_demo_mat.reshape(B).to(self.device)

            max_cut_cpu = cutoffs_mat[:, -1].tolist()
            for e_i in range(B_ep):
                rel_idx, _ = DeltaLoopVLAPolicy.compute_written_frame_indices(
                    int(max_cut_cpu[e_i]), int(exec_cpu[e_i]), cfg, device=self.store_device
                )
                w_cnt = int(rel_idx.numel())
                ep_w_counts.append(w_cnt)
                if w_cnt > 0:
                    ep_frame_rows.append(starts_cpu[e_i] + rel_idx)
                    ep_step_indices.append(rel_idx)
                    ep_is_demo.append(rel_idx < exec_cpu[e_i])

        if ep_frame_rows:
            all_hist_rows = torch.cat(ep_frame_rows, dim=0)
            hist_front_cat = self.front_emb[all_hist_rows].to(self.device, non_blocking=True)
            hist_wrist_cat = self.wrist_emb[all_hist_rows].to(self.device, non_blocking=True)
            hist_state_cat = self.state_norm[all_hist_rows].to(self.device, non_blocking=True)
            hist_step_cat = torch.cat(ep_step_indices, dim=0).to(self.device, non_blocking=True)
            hist_demo_cat = torch.cat(ep_is_demo, dim=0).to(self.device, non_blocking=True)
        else:
            hist_front_cat = torch.empty((0, 64, 768), device=self.device, dtype=self.feature_dtype)
            hist_wrist_cat = torch.empty((0, 16, 768), device=self.device, dtype=self.feature_dtype)
            hist_state_cat = torch.empty((0, 8), device=self.device, dtype=torch.float32)
            hist_step_cat = torch.empty((0,), device=self.device, dtype=torch.long)
            hist_demo_cat = torch.empty((0,), device=self.device, dtype=torch.bool)

        ep_motion_seqs: Optional[List[torch.Tensor]] = None
        if cfg.use_motion_beta_gate:
            ep_motion_seqs = []
            cur_f = 0
            for wc in ep_w_counts:
                ep_motion_seqs.append(DeltaLoopVLAPolicy.compute_visual_motion_rms(hist_front_cat[cur_f : cur_f + wc]))
                cur_f += wc

        return {
            "B_ep": B_ep,
            "chunks_per_ep": chunks_per_ep,
            "curr_front": curr_front,
            "curr_wrist": curr_wrist,
            "curr_state": curr_state,
            "prev_front": prev_front,
            "prev_wrist": prev_wrist,
            "prev_state": prev_state,
            "txt_tok": txt_tok,
            "txt_mask": txt_mask,
            "target_actions": target_actions,
            "aux_targets": aux_targets,
            "gt_xy_norm": gt_xy,
            "gt_has_xy": gt_has_xy,
            "gt_sg_emb": gt_sg,
            "gt_has_sg": gt_has_sg,
            "sample_local_ep": sample_local_ep,
            "sample_written_counts": sample_written_counts,
            "sample_demo_strided": demo_strided_per_sample,
            "ep_w_counts": ep_w_counts,
            "ep_motion_seqs": ep_motion_seqs,
            "hist_front_cat": hist_front_cat,
            "hist_wrist_cat": hist_wrist_cat,
            "hist_state_cat": hist_state_cat,
            "hist_step_cat": hist_step_cat,
            "hist_demo_cat": hist_demo_cat,
        }

    def sample_writeback_chains(
        self,
        eps_per_task: int,
        chain_chunks: int,
        cfg: DeltaLoopVLAConfig,
        vis_anchor_proj: torch.Tensor,
        gen: torch.Generator,
    ) -> Dict[str, Any]:
        """Sample contiguous `exec_horizon`-spaced chunk chains for `m6b_deltaloop_wb` training."""
        chosen_eps = []
        for t_ep_ids in self.task_to_ep_ids:
            perm = torch.randint(
                0, t_ep_ids.numel(), (eps_per_task,), generator=gen, device="cpu"
            ).to(self.store_device)
            chosen_eps.append(t_ep_ids[perm])
        ep_ids = torch.cat(chosen_eps, dim=0)
        B_ep = int(ep_ids.numel())

        starts = self.ep_starts[ep_ids]
        ends = self.ep_ends[ep_ids]
        exec_s = self.ep_exec_starts[ep_ids]
        lens = ends - starts
        eh = cfg.exec_horizon
        ws = cfg.write_stride

        # Random phase in `[0, exec_horizon - 1]` + random chain start chunk
        phase = torch.randint(0, eh, (B_ep,), generator=gen, device="cpu").to(self.store_device)
        first_valid = torch.minimum(exec_s + phase, lens - 1)
        total_chunks_ep = ((lens - first_valid + eh - 1) // eh).clamp(min=1)
        max_c_start = (total_chunks_ep - chain_chunks).clamp(min=0)
        u_c = torch.rand((B_ep,), generator=gen, device="cpu").to(self.store_device)
        c_start = (u_c * (max_c_start + 1).float()).long().clamp(max=max_c_start)

        c_offsets = torch.arange(chain_chunks, device=self.store_device, dtype=torch.long).unsqueeze(0)
        chunk_idx_mat = c_start.unsqueeze(1) + c_offsets  # (B_ep, M_wb)
        raw_steps = first_valid.unsqueeze(1) + chunk_idx_mat * eh
        active_mask = (raw_steps < lens.unsqueeze(1)).to(self.device)  # (B_ep, M_wb)
        steps_mat = torch.minimum(raw_steps, (lens - 1).unsqueeze(1))

        raw_cut = (steps_mat - cfg.exclusion_window + 1).clamp(min=0)
        cutoffs_mat = torch.maximum(exec_s.unsqueeze(1), raw_cut)  # (B_ep, M_wb)

        # Gather all strided frames up to `cutoffs_mat[:, -1]` per episode
        starts_cpu = starts.tolist()
        exec_cpu = exec_s.tolist()

        ep_frame_rows, ep_step_idx, ep_demo, ep_max_w = [], [], [], []
        if not cfg.v3_mode:
            written_mat = ((cutoffs_mat + ws - 1) // ws).to(self.device)
            ep_max_w = ((cutoffs_mat[:, -1] + ws - 1) // ws).tolist()
            for e_i in range(B_ep):
                wc = int(ep_max_w[e_i])
                if wc > 0:
                    r_idx = torch.arange(0, wc * ws, ws, device=self.store_device, dtype=torch.long)
                    ep_frame_rows.append(starts_cpu[e_i] + r_idx)
                    ep_step_idx.append(r_idx)
                    ep_demo.append(r_idx < exec_cpu[e_i])
        else:
            ds_vec = torch.where(
                exec_s > cfg.adaptive_demo_stride_threshold,
                torch.full_like(exec_s, cfg.long_demo_stride),
                torch.full_like(exec_s, ws),
            )
            demo_lim_mat = torch.minimum(cutoffs_mat, exec_s.unsqueeze(1))
            w_demo_mat = (demo_lim_mat + ds_vec.unsqueeze(1) - 1) // ds_vec.unsqueeze(1)
            exec_rem_mat = (cutoffs_mat - exec_s.unsqueeze(1)).clamp(min=0)
            w_exec_mat = (exec_rem_mat + ws - 1) // ws
            written_mat = (w_demo_mat + w_exec_mat).to(self.device)
            max_cut_cpu = cutoffs_mat[:, -1].tolist()
            for e_i in range(B_ep):
                r_idx, _ = DeltaLoopVLAPolicy.compute_written_frame_indices(
                    int(max_cut_cpu[e_i]), int(exec_cpu[e_i]), cfg, device=self.store_device
                )
                wc = int(r_idx.numel())
                ep_max_w.append(wc)
                if wc > 0:
                    ep_frame_rows.append(starts_cpu[e_i] + r_idx)
                    ep_step_idx.append(r_idx)
                    ep_demo.append(r_idx < exec_cpu[e_i])

        if ep_frame_rows:
            rows_cat = torch.cat(ep_frame_rows, dim=0)
            hf = self.front_emb[rows_cat].to(self.device, non_blocking=True)
            hw = self.wrist_emb[rows_cat].to(self.device, non_blocking=True)
            hs = self.state_norm[rows_cat].to(self.device, non_blocking=True)
            ht = torch.cat(ep_step_idx, dim=0).to(self.device, non_blocking=True)
            hd = torch.cat(ep_demo, dim=0).to(self.device, non_blocking=True)
        else:
            hf = torch.empty((0, 64, 768), device=self.device, dtype=self.feature_dtype)
            hw = torch.empty((0, 16, 768), device=self.device, dtype=self.feature_dtype)
            hs = torch.empty((0, 8), device=self.device, dtype=torch.float32)
            ht = torch.empty((0,), device=self.device, dtype=torch.long)
            hd = torch.empty((0,), device=self.device, dtype=torch.bool)

        ep_motion_seqs: Optional[List[torch.Tensor]] = None
        if cfg.use_motion_beta_gate:
            ep_motion_seqs = []
            cur_f = 0
            for wc in ep_max_w:
                ep_motion_seqs.append(DeltaLoopVLAPolicy.compute_visual_motion_rms(hf[cur_f : cur_f + int(wc)]))
                cur_f += int(wc)

        # Gather per-step current observations & targets `(B_ep, M_wb, ...)`
        B_flat = B_ep * chain_chunks
        ep_rep = ep_ids.repeat_interleave(chain_chunks)
        st_flat = steps_mat.reshape(B_flat)
        cut_flat = cutoffs_mat.reshape(B_flat)
        abs_flat = self.ep_starts[ep_rep] + st_flat

        prev_front = None
        prev_wrist = None
        prev_state = None
        if cfg.obs_horizon2:
            prev_st_flat = torch.maximum(self.ep_exec_starts[ep_rep], st_flat - 1)
            prev_abs = self.ep_starts[ep_rep] + prev_st_flat
            prev_front = self.front_emb[prev_abs].to(self.device).view(B_ep, chain_chunks, 64, 768)
            prev_wrist = self.wrist_emb[prev_abs].to(self.device).view(B_ep, chain_chunks, 16, 768)
            prev_state = self.state_norm[prev_abs].to(self.device).view(B_ep, chain_chunks, 8)

        gt_xy = (
            self.gt_xy_norm[abs_flat].to(self.device).view(B_ep, chain_chunks, 2)
            if self.gt_xy_norm is not None
            else None
        )
        gt_has_xy = (
            self.gt_has_xy[abs_flat].to(self.device).view(B_ep, chain_chunks)
            if self.gt_has_xy is not None
            else None
        )
        gt_sg = (
            self.gt_sg_emb[abs_flat].to(self.device).view(B_ep, chain_chunks, -1)
            if self.gt_sg_emb is not None
            else None
        )
        gt_has_sg = (
            self.gt_has_sg[abs_flat].to(self.device).view(B_ep, chain_chunks)
            if self.gt_has_sg is not None
            else None
        )

        return {
            "B_ep": B_ep,
            "M_wb": chain_chunks,
            "steps_mat": steps_mat.to(self.device),
            "cutoffs_mat": cutoffs_mat.to(self.device),
            "written_mat": written_mat,
            "active_mask": active_mask,
            "ep_max_w": ep_max_w,
            "ep_motion_seqs": ep_motion_seqs,
            "hf": hf,
            "hw": hw,
            "hs": hs,
            "ht": ht,
            "hd": hd,
            "curr_front": self.front_emb[abs_flat].to(self.device).view(B_ep, chain_chunks, 64, 768),
            "curr_wrist": self.wrist_emb[abs_flat].to(self.device).view(B_ep, chain_chunks, 16, 768),
            "curr_state": self.state_norm[abs_flat].to(self.device).view(B_ep, chain_chunks, 8),
            "prev_front": prev_front,
            "prev_wrist": prev_wrist,
            "prev_state": prev_state,
            "txt_tok": self.text_token_emb[ep_ids].to(self.device),
            "txt_mask": self.text_attn_mask[ep_ids].to(self.device),
            "target_actions": self.encode_action_chunks_gpu(ep_rep, st_flat, cfg.action_horizon).view(
                B_ep, chain_chunks, cfg.action_horizon, cfg.action_dim
            ),
            "aux_targets": self.compute_aux_targets(ep_rep, st_flat, cut_flat, vis_anchor_proj).view(
                B_ep, chain_chunks, -1
            ),
            "gt_xy_norm": gt_xy,
            "gt_has_xy": gt_has_xy,
            "gt_sg_emb": gt_sg,
            "gt_has_sg": gt_has_sg,
        }


# =============================================================================
# 3. Forward Helpers & EMA Manager
# =============================================================================


class EMAModel:
    """Fast fp32 Exponential Moving Average shadow parameter tracker."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().float().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        one_m = 1.0 - d
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=one_m)
            else:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def update_from_master(self, master_entries: List[Tuple[str, nn.Parameter, torch.Tensor]], exact_copy: bool = False) -> None:
        if exact_copy:
            for k, _, mp in master_entries:
                self.shadow[k].copy_(mp.detach())
        else:
            d = self.decay
            one_m = 1.0 - d
            for k, _, mp in master_entries:
                self.shadow[k].mul_(d).add_(mp.detach(), alpha=one_m)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, v in sd.items():
            v.copy_(self.shadow[k].to(dtype=v.dtype, device=v.device))


def _batch_apply_episode_conv(
    slot_resampler: Any,
    raw_cat: torch.Tensor,
    ep_w_counts: List[int],
) -> List[torch.Tensor]:
    """Apply `apply_episode_slot_conv` across all episodes in ONE batched `Conv1d` launch (no in-place CopySlices!)."""
    B_ep = len(ep_w_counts)
    max_w = max(ep_w_counts) if ep_w_counts else 0
    if max_w == 0 or raw_cat.shape[0] == 0:
        return [raw_cat[:0] for _ in range(B_ep)]

    total_w, nw, d = raw_cat.shape
    dev = raw_cat.device
    # Append 1 dummy zero frame at index `total_w` so padded positions gather cleanly without in-place mutation
    raw_pad_row = torch.cat(
        [raw_cat, torch.zeros((1, nw, d), device=dev, dtype=raw_cat.dtype)], dim=0
    )
    pad_idx = torch.full((B_ep, max_w), total_w, dtype=torch.long)
    cursor = 0
    for e_i, wc in enumerate(ep_w_counts):
        if wc > 0:
            pad_idx[e_i, :wc] = torch.arange(cursor, cursor + wc, dtype=torch.long)
            cursor += wc
    padded_raw = raw_pad_row[pad_idx.to(device=dev, non_blocking=True)]  # (B_ep, max_w, nw, d)
    padded_conv = slot_resampler.apply_episode_slot_conv(padded_raw)     # (B_ep, max_w, nw, d)
    return [padded_conv[e_i, :wc] if wc > 0 else raw_cat[:0] for e_i, wc in enumerate(ep_w_counts)]


def build_batch_memory_state(
    model: DeltaLoopVLAPolicy,
    batch: Dict[str, Any],
) -> Optional[Dict[str, torch.Tensor]]:
    """Encode episode history frames ONCE per episode and build per-sample memory state (zero-loop autograd!)."""
    if not model.uses_history:
        return None

    assert model.slot_resampler is not None
    ep_w_counts: List[int] = batch["ep_w_counts"]
    dev = batch["curr_front"].device

    # Fast zero-loop vectorized path for `m2_framesamp_modul` (`temporal_conv is None`!):
    # Encodes ONLY unique sampled buffer frames and gathers `(B, B_sz, d)` in ONE single tensor op!
    if model.arch == "m2_framesamp_modul":
        B = int(batch["sample_local_ep"].shape[0])
        B_sz = model.cfg.framesamp_buffer_size
        D = model.cfg.d_model
        dt = model.slot_resampler.slot_queries.dtype
        total_hist = int(batch["hist_front_cat"].shape[0])
        if total_hist == 0:
            return {
                "buf_tokens": torch.zeros((B, B_sz, D), device=dev, dtype=dt),
                "buf_mask": torch.zeros((B, B_sz), device=dev, dtype=torch.bool),
            }

        ep_offsets = [0]
        for wc in ep_w_counts:
            ep_offsets.append(ep_offsets[-1] + wc)

        ep_ids_l = batch["sample_local_ep"].tolist()
        wc_l = batch["sample_written_counts"].tolist()
        dc_l = batch["sample_demo_strided"].tolist()

        idx_cpu = torch.zeros((B, B_sz), dtype=torch.long)
        mask_cpu = torch.zeros((B, B_sz), dtype=torch.bool)
        cpu_dev = torch.device("cpu")
        for b_i in range(B):
            n_elig = wc_l[b_i]
            if n_elig > 0:
                sel_idx, n_sel = model.select_framesamp_slot_indices(
                    n_elig, min(dc_l[b_i], n_elig), budget=B_sz, device=cpu_dev
                )
                idx_cpu[b_i, :n_sel] = ep_offsets[ep_ids_l[b_i]] + sel_idx
                mask_cpu[b_i, :n_sel] = True

        if not bool(mask_cpu.any()):
            return {
                "buf_tokens": torch.zeros((B, B_sz, D), device=dev, dtype=dt),
                "buf_mask": torch.zeros((B, B_sz), device=dev, dtype=torch.bool),
            }

        uniq_cpu, inv_valid_cpu = torch.unique(idx_cpu[mask_cpu], sorted=True, return_inverse=True)
        uniq_dev = uniq_cpu.to(device=dev, non_blocking=True)
        raw_uniq = model.slot_resampler.encode_frames_raw(
            batch["hist_front_cat"][uniq_dev],
            batch["hist_wrist_cat"][uniq_dev],
            batch["hist_state_cat"][uniq_dev],
            batch["hist_step_cat"][uniq_dev],
            batch["hist_demo_cat"][uniq_dev],
        )  # (U, 1, D)
        slots_uniq = model.slot_resampler.out_norm(raw_uniq[:, 0, :].to(dt))  # (U, D)
        n_u = int(slots_uniq.shape[0])
        slots_pad = torch.cat([slots_uniq, torch.zeros((1, D), device=dev, dtype=dt)], dim=0)

        gather_cpu = torch.full((B, B_sz), n_u, dtype=torch.long)
        gather_cpu[mask_cpu] = inv_valid_cpu
        buf_tok = slots_pad[gather_cpu.to(device=dev, non_blocking=True)]  # (B, B_sz, D)
        buf_mask = mask_cpu.to(device=dev, non_blocking=True)
        return {"buf_tokens": buf_tok, "buf_mask": buf_mask}

    raw_cat = model.slot_resampler.encode_frames_raw(
        batch["hist_front_cat"],
        batch["hist_wrist_cat"],
        batch["hist_state_cat"],
        batch["hist_step_cat"],
        batch["hist_demo_cat"],
    )  # (Total_W, N_w, d)
    ep_slot_seqs = _batch_apply_episode_conv(model.slot_resampler, raw_cat, ep_w_counts)

    return model.compute_packed_gdn_states(
        ep_slot_seqs,
        batch["sample_local_ep"],
        batch["sample_written_counts"],
        episode_motion_seqs=batch.get("ep_motion_seqs"),
    )


def forward_writeback_chain_step(
    model: DeltaLoopVLAPolicy,
    chain_batch: Dict[str, Any],
    eta_aux: float,
    gen: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Batched segment-sequential forward/loss for `m6b_deltaloop_wb` (`RESEARCH_PLAN.md` §3.3 lines 122-145).

    Fuses `obs_encoder`, `project_frame_slots`, `project_writeback_slot`, and the 6-layer `FlowActionExpert`
    (`compute_crn_flow_and_aux_loss`) across `M_wb * B_ep = 128` chunks while keeping the sequential
    chunk-to-chunk GDN write-back state recurrence mathematically identical.
    """
    assert model.arch == "m6b_deltaloop_wb"
    assert model.slot_resampler is not None and model.gdn is not None and model.wb_proj is not None

    B_ep: int = chain_batch["B_ep"]
    M_wb: int = chain_batch["M_wb"]
    B_tot = M_wb * B_ep
    dev = chain_batch["curr_front"].device
    ep_motion_seqs: Optional[List[torch.Tensor]] = chain_batch.get("ep_motion_seqs")

    raw_cat = model.slot_resampler.encode_frames_raw(
        chain_batch["hf"],
        chain_batch["hw"],
        chain_batch["hs"],
        chain_batch["ht"],
        chain_batch["hd"],
    )
    ep_slots = _batch_apply_episode_conv(model.slot_resampler, raw_cat, chain_batch["ep_max_w"])

    # Initial prefix state at `m = 0` (`written_mat[:, 0]`)
    w0 = chain_batch["written_mat"][:, 0]
    init_st = model.compute_packed_gdn_states(
        ep_slots,
        torch.arange(B_ep, device=dev, dtype=torch.long),
        w0,
        episode_motion_seqs=ep_motion_seqs,
    )
    S_cur = init_st["S"]     # (B_ep, H, V, K)
    occ_cur = init_st["occ"]  # (B_ep, H, K)

    # Pre-project ALL episode observation slots in ONE single batched call
    Nw = model.cfg.num_write_slots
    ep_proj: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    nonempty_e = [e_i for e_i, z_ep in enumerate(ep_slots) if z_ep.shape[0] > 0]
    if nonempty_e:
        z_cat = torch.cat([ep_slots[e_i] for e_i in nonempty_e], dim=0)
        m_cat = (
            torch.cat([ep_motion_seqs[e_i] for e_i in nonempty_e], dim=0).unsqueeze(0)
            if ep_motion_seqs is not None
            else None
        )
        k_cat, v_cat, g_cat, b_cat, _ = model.gdn.project_frame_slots(z_cat.unsqueeze(0), motion_feat=m_cat)
        tok_off = 0
        for e_i in nonempty_e:
            ntok = int(ep_slots[e_i].shape[0]) * Nw
            ep_proj[e_i] = (
                k_cat[0, tok_off : tok_off + ntok],
                v_cat[0, tok_off : tok_off + ntok],
                g_cat[0, tok_off : tok_off + ntok],
                b_cat[0, tok_off : tok_off + ntok],
            )
            tok_off += ntok

    # Encode all `M_wb * B_ep` current observations in ONE single `obs_encoder` pass (chunk-major `(M_wb, B_ep)`)
    cf_flat = chain_batch["curr_front"].transpose(0, 1).reshape(B_tot, 64, 768)
    cw_flat = chain_batch["curr_wrist"].transpose(0, 1).reshape(B_tot, 16, 768)
    cs_flat = chain_batch["curr_state"].transpose(0, 1).reshape(B_tot, 8)
    pf_flat = (
        chain_batch["prev_front"].transpose(0, 1).reshape(B_tot, 64, 768)
        if chain_batch.get("prev_front") is not None
        else None
    )
    pw_flat = (
        chain_batch["prev_wrist"].transpose(0, 1).reshape(B_tot, 16, 768)
        if chain_batch.get("prev_wrist") is not None
        else None
    )
    ps_flat = (
        chain_batch["prev_state"].transpose(0, 1).reshape(B_tot, 8)
        if chain_batch.get("prev_state") is not None
        else None
    )
    tt_flat = chain_batch["txt_tok"].unsqueeze(0).expand(M_wb, B_ep, -1, -1).reshape(B_tot, -1, 768)
    tm_flat = chain_batch["txt_mask"].unsqueeze(0).expand(M_wb, B_ep, -1).reshape(B_tot, -1)
    aug_theta = None
    if model.training and model.cfg.subgrid_aug:
        aug_theta = sample_subgrid_crop_affine(
            B_tot,
            scale_range=(model.cfg.subgrid_scale_min, 1.0),
            max_extra_shift=model.cfg.subgrid_max_shift,
            device=dev,
        )
    ctx_flat, cmask_flat = model.obs_encoder(
        cf_flat,
        cw_flat,
        cs_flat,
        tt_flat,
        tm_flat,
        prev_front_emb=pf_flat,
        prev_wrist_emb=pw_flat,
        prev_state_norm=ps_flat,
        aug_theta=aug_theta,
    )
    ctx_by_m = ctx_flat.view(M_wb, B_ep, ctx_flat.shape[1], ctx_flat.shape[2])
    cmask_by_m = cmask_flat.view(M_wb, B_ep, cmask_flat.shape[1])

    # Pending write-back queue per episode `(step_idx, wk_1, wv_1, wg_1, wb_beta_1)`
    pending_wb: List[List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]] = [
        [] for _ in range(B_ep)
    ]
    L_max = model.cfg.max_loops
    coda_by_m: List[List[Optional[torch.Tensor]]] = []

    steps_cpu = chain_batch["steps_mat"].tolist()
    written_cpu = chain_batch["written_mat"].tolist()
    cutoffs_cpu = chain_batch["cutoffs_mat"].tolist()

    for m_i in range(M_wb):
        ro_m = model.run_coda_readout(
            ctx_by_m[m_i], cmask_by_m[m_i], mem_state={"S": S_cur, "occ": occ_cur}
        )
        coda_by_m.append(ro_m["coda_per_depth"])

        if m_i < M_wb - 1:
            # A2: Uniform depth selection for write-back during Stage-I training
            l_wb = int(torch.randint(0, L_max, (1,), generator=gen).item())
            c_sel = ro_m["coda_per_depth"][l_wb]
            assert c_sel is not None
            wb_all = model.wb_proj(c_sel.mean(dim=1))  # (B_ep, d)
            if (m_i + 1) % model.cfg.wb_trunc_chunks == 0:
                wb_all = wb_all.detach()
                S_cur = S_cur.detach()
                occ_cur = occ_cur.detach()

            # Project all `B_ep` write-back slots in ONE batched call
            wk_all, wv_all, wg_all, wbeta_all = model.gdn.project_writeback_slot(wb_all.unsqueeze(1))
            for e_i in range(B_ep):
                pending_wb[e_i].append(
                    (
                        int(steps_cpu[e_i][m_i]),
                        wk_all[e_i],
                        wv_all[e_i],
                        wg_all[e_i],
                        wbeta_all[e_i],
                    )
                )

            # Advance `[w_curr, w_next)` + eligible write-back slots `< next_cutoff` via packed GDN!
            seg_k, seg_v, seg_g, seg_b = [], [], [], []
            seg_lens = []
            seg_ep_ids = []
            for e_i in range(B_ep):
                wp = int(written_cpu[e_i][m_i])
                wn = int(written_cpu[e_i][m_i + 1])
                cn = int(cutoffs_cpu[e_i][m_i + 1])
                wb_elig = [item for item in pending_wb[e_i] if item[0] < cn]
                pending_wb[e_i] = [item for item in pending_wb[e_i] if item[0] >= cn]
                if wn <= wp and not wb_elig:
                    continue

                ev_k, ev_v, ev_g, ev_b = [], [], [], []
                if wn > wp and e_i in ep_proj:
                    pk, pv, pg, pb = ep_proj[e_i]
                    ev_k.append(pk[wp * Nw : wn * Nw])
                    ev_v.append(pv[wp * Nw : wn * Nw])
                    ev_g.append(pg[wp * Nw : wn * Nw])
                    ev_b.append(pb[wp * Nw : wn * Nw])
                for _, wk_i, wv_i, wg_i, wbeta_i in wb_elig:
                    ev_k.append(wk_i)
                    ev_v.append(wv_i)
                    ev_g.append(wg_i)
                    ev_b.append(wbeta_i)

                if ev_k:
                    kc = torch.cat(ev_k, dim=0)
                    seg_k.append(kc)
                    seg_v.append(torch.cat(ev_v, dim=0))
                    seg_g.append(torch.cat(ev_g, dim=0))
                    seg_b.append(torch.cat(ev_b, dim=0))
                    seg_lens.append(kc.shape[0])
                    seg_ep_ids.append(e_i)

            if seg_ep_ids:
                idx_seg = torch.tensor(seg_ep_ids, device=dev, dtype=torch.long)
                cu_seg = torch.tensor(
                    [0] + list(torch.cumsum(torch.tensor(seg_lens), dim=0).tolist()),
                    device=dev,
                    dtype=torch.int32,
                )
                _, s_upd, occ_upd = run_dual_backend_gdn(
                    k=torch.cat(seg_k, dim=0).unsqueeze(0),
                    v=torch.cat(seg_v, dim=0).unsqueeze(0),
                    g_fp32=torch.cat(seg_g, dim=0).unsqueeze(0),
                    beta_fp32=torch.cat(seg_b, dim=0).unsqueeze(0),
                    initial_state_fp32=S_cur[idx_seg],
                    initial_occ_fp32=occ_cur[idx_seg],
                    cu_seqlens=cu_seg,
                    track_occ=True,
                )
                # Differentiable scatter update without in-place autograd violation
                S_next = S_cur.clone()
                occ_next = occ_cur.clone()
                S_next[idx_seg] = s_upd
                if occ_upd is not None:
                    occ_next[idx_seg] = occ_upd
                S_cur = S_next
                occ_cur = occ_next

    # Single fused CRN FlowActionExpert + Aux pass across all `M_wb * B_ep = 128` chunks!
    coda_cat_per_depth: List[Optional[torch.Tensor]] = [
        torch.cat([coda_by_m[m_i][l_idx] for m_i in range(M_wb)], dim=0)  # type: ignore[arg-type]
        for l_idx in range(L_max)
    ]
    targ_flat = chain_batch["target_actions"].transpose(0, 1).reshape(
        B_tot, model.cfg.action_horizon, model.cfg.action_dim
    )
    aux_flat = chain_batch["aux_targets"].transpose(0, 1).reshape(B_tot, -1)
    xy_flat = None
    xy_mask_flat = None
    if chain_batch.get("gt_xy_norm") is not None and chain_batch.get("gt_has_xy") is not None:
        xy_raw = chain_batch["gt_xy_norm"].transpose(0, 1).reshape(B_tot, 2)
        xy_mask_raw = chain_batch["gt_has_xy"].transpose(0, 1).reshape(B_tot)
        if aug_theta is not None:
            xy_flat, xy_mask_flat = transform_coords_under_affine(xy_raw, xy_mask_raw, aug_theta)
        else:
            xy_flat, xy_mask_flat = xy_raw, xy_mask_raw
    sg_flat = (
        chain_batch["gt_sg_emb"].transpose(0, 1).reshape(B_tot, -1)
        if chain_batch.get("gt_sg_emb") is not None
        else None
    )
    sg_mask_flat = (
        chain_batch["gt_has_sg"].transpose(0, 1).reshape(B_tot)
        if chain_batch.get("gt_has_sg") is not None
        else None
    )
    loss_all = model.compute_crn_flow_and_aux_loss(
        ctx_flat,
        cmask_flat,
        coda_cat_per_depth,
        None,
        targ_flat,
        aux_flat,
        eta_aux=eta_aux,
        grounding_xy_target=xy_flat,
        grounding_xy_mask=xy_mask_flat,
        subgoal_emb_target=sg_flat,
        subgoal_emb_mask=sg_mask_flat,
    )
    pd_mat = loss_all["per_depth_fm_loss"].view(M_wb, B_ep, L_max)  # (M_wb, B_ep, L_max)
    act_mb = chain_batch["active_mask"].transpose(0, 1).float()      # (M_wb, B_ep)
    w_sums = act_mb.sum(dim=1).clamp(min=1.0)                        # (M_wb,)
    fm_per_m = (pd_mat.mean(dim=2) * act_mb).sum(dim=1) / w_sums     # (M_wb,)
    fm_mean = fm_per_m.mean()
    aux_mean = loss_all["aux_loss"]
    gnd_mean = loss_all["ground_loss"]
    return {
        "loss": fm_mean + eta_aux * aux_mean + model.cfg.grounding_loss_weight * gnd_mean,
        "fm_loss": fm_mean,
        "aux_loss": aux_mean,
        "ground_loss": gnd_mean,
        "grip_loss": loss_all["grip_loss"],
        "per_depth_mean": pd_mat.detach().mean(dim=(0, 1)),
    }


# =============================================================================
# 4. Stage-I + Stage-II Training Runner per Controlled Model
# =============================================================================


def train_single_model(
    arch: ArchVLA,
    store: GPURoboMMEStore,
    args: argparse.Namespace,
    device: torch.device,
    train_dtype: torch.dtype,
    out_dir: Path,
) -> Dict[str, Any]:
    ckpt_path = out_dir / f"{arch}.pt"
    if args.resume and ckpt_path.exists():
        try:
            existing = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if existing.get("completed", False):
                print(f"[{arch}] Checkpoint already completed at {ckpt_path} — skipping (--resume).")
                return existing.get("metrics", {})
        except Exception as exc:
            print(f"[{arch}] Warning: corrupt existing checkpoint ({exc}), retraining...")

    # Reset ALL RNGs identically per model so every model sees identical batch order & CRN draws!
    set_all_seeds(args.seed)
    batch_gen = torch.Generator(device="cpu")
    batch_gen.manual_seed(args.seed + 10007)

    if getattr(args, "v3", False):
        cfg = DeltaLoopVLAConfig.v3_preset(
            arch=arch,
            d_model=args.d_model,
            attn_heads=args.attn_heads,
            d_ff=args.d_ff,
            ctx_layers=args.ctx_layers,
            expert_layers=args.expert_layers,
            num_write_slots=args.num_write_slots if args.num_write_slots is not None else 8,
            num_coda_tokens=args.num_coda_tokens,
            write_stride=args.write_stride,
            gdn_heads=args.gdn_heads,
            gdn_d_k=args.gdn_d_k if args.gdn_d_k is not None else 128,
            gdn_d_v=args.gdn_d_v if args.gdn_d_v is not None else 128,
            max_loops=args.max_loops,
        )
    else:
        cfg = DeltaLoopVLAConfig(
            arch=arch,
            d_model=args.d_model,
            attn_heads=args.attn_heads,
            d_ff=args.d_ff,
            ctx_layers=args.ctx_layers,
            expert_layers=args.expert_layers,
            num_write_slots=args.num_write_slots if args.num_write_slots is not None else 4,
            num_coda_tokens=args.num_coda_tokens,
            write_stride=args.write_stride,
            gdn_heads=args.gdn_heads,
            gdn_d_k=args.gdn_d_k if args.gdn_d_k is not None else 64,
            gdn_d_v=args.gdn_d_v if args.gdn_d_v is not None else 64,
            max_loops=args.max_loops,
        )
    model = DeltaLoopVLAPolicy(cfg).to(device=device)
    param_info = model.parameter_breakdown()
    state_bytes = model.state_size_bytes()

    print(
        f"\n================================================================================\n"
        f"[{arch}] Starting Stage-I Training (steps={args.steps}, seed={args.seed}, v3_mode={cfg.v3_mode})\n"
        f"  params={param_info['total']:,} (shared_backbone={param_info['shared_backbone']:,}, "
        f"mem_loop={param_info['memory_and_loop']:,}, halt={param_info['halting_gate']:,})\n"
        f"  state_bytes={state_bytes:,} B ({state_bytes/1024:.1f} KB) | train_dtype={train_dtype} | HAS_FLA={HAS_FLA}\n"
        f"================================================================================"
    )

    # Freeze halting_gate during Stage I; train it in Stage II on frozen EMA policy!
    if model.halting_gate is not None:
        model.halting_gate.requires_grad_(False)

    ema = EMAModel(model, decay=args.ema_decay)

    use_bf16_weights = (
        device.type == "cuda"
        and train_dtype == torch.bfloat16
        and getattr(args, "bf16_weights", True)
    )
    fp32_keep_suffixes = (".A_log", ".dt_bias", ".gamma_ctx")
    write_prefixes = (
        "slot_resampler.",
        "gdn.",
        "gdn1.",
        "gdn2.",
        "mid_norm.",
        "mid_ffn.",
        "framesamp_refine.",
        "wb_proj.",
    )
    master_entries: List[Tuple[str, nn.Parameter, torch.Tensor]] = []
    if use_bf16_weights:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            mp = p.detach().float().clone().requires_grad_(True)
            if not name.endswith(fp32_keep_suffixes):
                p.data = p.data.to(torch.bfloat16)
            master_entries.append((name, p, mp))
        opt_params = [mp for _, _, mp in master_entries]
    else:
        opt_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        opt_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )

    use_autocast = device.type == "cuda" and train_dtype in (torch.bfloat16, torch.float16)
    log_interval = max(1, args.log_every)
    save_interval = max(0, int(getattr(args, "save_every", 2500)))
    resume_path = out_dir / f"_{arch}_resume.pt"

    start_step = 1
    prior_elapsed_s = 0.0
    history_logs: List[Dict[str, Any]] = []
    if args.resume and resume_path.exists():
        try:
            r_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            start_step = int(r_ckpt["step"]) + 1
            prior_elapsed_s = float(r_ckpt.get("elapsed_s", 0.0))
            history_logs = list(r_ckpt.get("history_logs", []))
            if use_bf16_weights and "master_state" in r_ckpt:
                ms_dict = r_ckpt["master_state"]
                for name, p, mp in master_entries:
                    if name in ms_dict:
                        mp.data.copy_(ms_dict[name].to(device=device, dtype=torch.float32))
                        p.data.copy_(mp.data.to(dtype=p.dtype))
            elif "model_state_dict" in r_ckpt:
                model.load_state_dict(r_ckpt["model_state_dict"], strict=True)
            if "ema_shadow" in r_ckpt:
                for k_s, v_s in r_ckpt["ema_shadow"].items():
                    if k_s in ema.shadow:
                        ema.shadow[k_s].copy_(v_s.to(device=ema.shadow[k_s].device, dtype=torch.float32))
            if "optimizer_state" in r_ckpt:
                optimizer.load_state_dict(r_ckpt["optimizer_state"])
            if "batch_gen_state" in r_ckpt:
                batch_gen.set_state(r_ckpt["batch_gen_state"])
            print(f"[{arch}] Resumed Stage-I from step {start_step - 1}/{args.steps} ({resume_path})")
        except Exception as exc:
            print(f"[{arch}] Warning: failed to load resume file {resume_path} ({exc}), starting from step 1.")
            start_step = 1
            prior_elapsed_s = 0.0
            history_logs = []

    t_start = time.perf_counter() - prior_elapsed_s

    running_fm = 0.0
    running_aux = 0.0
    running_gnd = 0.0
    running_grip = 0.0
    running_gn_w = 0.0
    running_gn_o = 0.0
    running_pd: Optional[torch.Tensor] = None
    window_steps = 0

    model.train()
    for step in range(start_step, args.steps + 1):
        # Cosine LR schedule with linear warmup
        if step <= args.warmup_steps:
            lr_now = args.lr * (step / max(1, args.warmup_steps))
        else:
            prog = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            lr_now = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        if arch == "m6b_deltaloop_wb":
            wb_batch = store.sample_writeback_chains(
                eps_per_task=max(1, args.eps_per_task // 2),
                chain_chunks=args.wb_chain_chunks,
                cfg=cfg,
                vis_anchor_proj=model.vis_anchor_proj,
                gen=batch_gen,
            )
            with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                out = forward_writeback_chain_step(model, wb_batch, eta_aux=args.eta_aux, gen=batch_gen)
                loss = out["loss"]
            pd_step = out["per_depth_mean"].detach().cpu()
        else:
            batch = store.sample_balanced_batch(
                eps_per_task=args.eps_per_task,
                chunks_per_ep=args.chunks_per_ep,
                cfg=cfg,
                vis_anchor_proj=model.vis_anchor_proj,
                gen=batch_gen,
            )
            with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                aug_theta = None
                if cfg.subgrid_aug:
                    aug_theta = sample_subgrid_crop_affine(
                        batch["curr_front"].shape[0],
                        scale_range=(cfg.subgrid_scale_min, 1.0),
                        max_extra_shift=cfg.subgrid_max_shift,
                        device=device,
                    )
                ctx_tok, ctx_mask = model.obs_encoder(
                    batch["curr_front"],
                    batch["curr_wrist"],
                    batch["curr_state"],
                    batch["txt_tok"],
                    batch["txt_mask"],
                    prev_front_emb=batch["prev_front"],
                    prev_wrist_emb=batch["prev_wrist"],
                    prev_state_norm=batch["prev_state"],
                    aug_theta=aug_theta,
                )
                xy_tgt = batch["gt_xy_norm"]
                xy_mask = batch["gt_has_xy"]
                if aug_theta is not None and xy_tgt is not None and xy_mask is not None:
                    xy_tgt, xy_mask = transform_coords_under_affine(xy_tgt, xy_mask, aug_theta)
                mem_state = build_batch_memory_state(model, batch)
                ro_out = model.run_coda_readout(ctx_tok, ctx_mask, mem_state=mem_state)
                out = model.compute_crn_flow_and_aux_loss(
                    ctx_tok,
                    ctx_mask,
                    ro_out["coda_per_depth"],
                    ro_out["mem_mask"],
                    batch["target_actions"],
                    batch["aux_targets"],
                    eta_aux=args.eta_aux,
                    grounding_xy_target=xy_tgt,
                    grounding_xy_mask=xy_mask,
                    subgoal_emb_target=batch["gt_sg_emb"],
                    subgoal_emb_mask=batch["gt_has_sg"],
                )
                loss = out["loss"]
            pd_step = out["per_depth_fm_loss"].detach().mean(dim=0).cpu()

        loss.backward()

        # Separate Gradient Norm Clipping: Write-side vs Loop/Action-Expert!
        if use_bf16_weights:
            write_mp: List[torch.Tensor] = []
            other_mp: List[torch.Tensor] = []
            for name, p, mp in master_entries:
                if p.grad is not None:
                    mp.grad = p.grad.detach().float()
                    if name.startswith(write_prefixes):
                        write_mp.append(mp)
                    else:
                        other_mp.append(mp)
                else:
                    mp.grad = None
            gn_w = float(torch.nn.utils.clip_grad_norm_(write_mp, args.max_grad_norm).item()) if write_mp else 0.0
            gn_o = float(torch.nn.utils.clip_grad_norm_(other_mp, args.max_grad_norm).item()) if other_mp else 0.0
            optimizer.step()
            with torch.no_grad():
                for _, p, mp in master_entries:
                    p.data.copy_(mp.data.to(dtype=p.dtype))
            ema.update_from_master(master_entries, exact_copy=(step < args.ema_start_step))
        else:
            write_p, other_p = model.split_write_vs_other_params()
            gn_w = float(torch.nn.utils.clip_grad_norm_(write_p, args.max_grad_norm).item()) if write_p else 0.0
            gn_o = float(torch.nn.utils.clip_grad_norm_(other_p, args.max_grad_norm).item()) if other_p else 0.0
            optimizer.step()
            if step >= args.ema_start_step:
                ema.update(model)
            else:
                for k_s, v_s in model.state_dict().items():
                    ema.shadow[k_s].copy_(v_s.detach().float())

        running_fm += float(out["fm_loss"].detach().item())
        running_aux += float(out["aux_loss"].detach().item())
        running_gnd += float(out.get("ground_loss", torch.tensor(0.0)).detach().item())
        running_grip += float(out.get("grip_loss", torch.tensor(0.0)).detach().item())
        running_gn_w += gn_w
        running_gn_o += gn_o
        running_pd = pd_step if running_pd is None else (running_pd + pd_step)
        window_steps += 1

        if step % log_interval == 0 or step in (1, 100, args.steps):
            denom = max(1, window_steps)
            avg_fm = running_fm / denom
            avg_aux = running_aux / denom
            avg_gnd = running_gnd / denom
            avg_grip = running_grip / denom
            avg_gw = running_gn_w / denom
            avg_go = running_gn_o / denom
            pd_vals = [round(float(x), 5) for x in (running_pd / denom).tolist()]  # type: ignore[operator]
            elapsed = time.perf_counter() - t_start
            ms_per_step = (elapsed / step) * 1000.0
            eta_s = max(0.0, (args.steps - step) * (elapsed / step))

            # Inspect Flamingo gamma gates & memory cross-attn o_proj norms across the 6 Action Expert layers
            gammas = [round(float(torch.tanh(b.gamma_ctx).item()), 4) for b in model.action_expert.blocks]
            ca_norms = [
                round(float(b.mem_cross_attn.o_proj.weight.detach().float().norm().item()), 3)
                for b in model.action_expert.blocks
            ]
            v3_str = f"gnd={avg_gnd:.4f} grip={avg_grip:.4f} ca_w={ca_norms[0]:.2f}/{ca_norms[-1]:.2f} " if cfg.v3_mode else ""
            print(
                f"[{arch} | step {step:5d}/{args.steps}] "
                f"fm={avg_fm:.5f} aux={avg_aux:.5f} {v3_str}pd={pd_vals} "
                f"lr={lr_now:.2e} gn_w={avg_gw:.3f} gn_o={avg_go:.3f} "
                f"tanh_g={gammas[0]:+.3f}/{gammas[-1]:+.3f} "
                f"({ms_per_step:.1f} ms/step, ETA {eta_s/60:.1f}m)",
                flush=True,
            )
            history_logs.append(
                {
                    "step": step,
                    "fm_loss": round(avg_fm, 6),
                    "aux_loss": round(avg_aux, 6),
                    "ground_loss": round(avg_gnd, 6),
                    "grip_loss": round(avg_grip, 6),
                    "per_depth_fm": pd_vals,
                    "grad_norm_write": round(avg_gw, 4),
                    "grad_norm_other": round(avg_go, 4),
                    "tanh_gamma": gammas,
                    "mem_ca_o_norm": ca_norms,
                    "elapsed_s": round(elapsed, 2),
                }
            )
            running_fm = 0.0
            running_aux = 0.0
            running_gnd = 0.0
            running_grip = 0.0
            running_gn_w = 0.0
            running_gn_o = 0.0
            running_pd = None
            window_steps = 0

        if save_interval > 0 and step < args.steps and (step % save_interval == 0):
            out_dir.mkdir(parents=True, exist_ok=True)
            r_tmp = out_dir / f"._{arch}_resume.pt.tmp"
            r_payload: Dict[str, Any] = {
                "step": step,
                "elapsed_s": time.perf_counter() - t_start,
                "history_logs": history_logs,
                "ema_shadow": {k_s: v_s.detach().cpu() for k_s, v_s in ema.shadow.items()},
                "optimizer_state": optimizer.state_dict(),
                "batch_gen_state": batch_gen.get_state(),
            }
            if use_bf16_weights:
                r_payload["master_state"] = {name: mp.detach().cpu() for name, _, mp in master_entries}
            else:
                r_payload["model_state_dict"] = {k_s: v_s.detach().cpu() for k_s, v_s in model.state_dict().items()}
            torch.save(r_payload, r_tmp)
            r_tmp.replace(resume_path)

    stage1_wall_s = time.perf_counter() - t_start

    # Copy smoothed EMA weights into `model` before Stage II & final checkpoint export!
    ema.copy_to(model)
    if model.halting_gate is not None:
        model.halting_gate.float()

    # -------------------------------------------------------------------------
    # Stage II: Frozen-Backbone CRN (`M_CRN = 8`) Adaptive Halting Gate Training
    # -------------------------------------------------------------------------
    stage2_metrics: Dict[str, Any] = {}
    if model.halting_gate is not None and args.stage2_steps > 0:
        print(
            f"[{arch}] Starting Stage-II Frozen-Backbone CRN Halting Training "
            f"(steps={args.stage2_steps}, M_CRN={args.m_crn})..."
        )
        t_s2 = time.perf_counter()
        model.eval()
        model.requires_grad_(False)
        model.halting_gate.train()
        model.halting_gate.requires_grad_(True)

        # Calibrate marginal-gain delta on 6 batches with M_CRN=8 draws
        pos_gaps = []
        with torch.no_grad():
            for _ in range(6):
                cb = store.sample_balanced_batch(
                    eps_per_task=args.eps_per_task,
                    chunks_per_ep=max(2, args.chunks_per_ep // 2),
                    cfg=cfg,
                    vis_anchor_proj=model.vis_anchor_proj,
                    gen=batch_gen,
                )
                with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                    ctx_c, cm_c = model.obs_encoder(
                        cb["curr_front"],
                        cb["curr_wrist"],
                        cb["curr_state"],
                        cb["txt_tok"],
                        cb["txt_mask"],
                        prev_front_emb=cb["prev_front"],
                        prev_wrist_emb=cb["prev_wrist"],
                        prev_state_norm=cb["prev_state"],
                    )
                    ms_c = build_batch_memory_state(model, cb)
                    ro_c = model.run_coda_readout(ctx_c, cm_c, mem_state=ms_c)
                    crn_acc = torch.zeros(
                        (cb["curr_front"].shape[0], cfg.max_loops), device=device, dtype=torch.float32
                    )
                    for _ in range(args.m_crn):
                        lo_m = model.compute_crn_flow_and_aux_loss(
                            ctx_c, cm_c, ro_c["coda_per_depth"], None, cb["target_actions"], eta_aux=0.0
                        )
                        crn_acc += lo_m["per_depth_fm_loss"]
                    crn_mean = crn_acc / float(args.m_crn)
                    gap0 = (crn_mean[:, 0] - crn_mean[:, -1]).clamp(min=0.0)
                    pos_gaps.append(gap0)

        all_gaps = torch.cat(pos_gaps, dim=0)
        calib_delta = float(torch.quantile(all_gaps, 0.55).clamp(min=3e-4, max=0.02).item())
        model.halting_gate.calibrated_delta.fill_(calib_delta)
        print(f"[{arch} | Stage-II] Calibrated marginal-gain delta = {calib_delta:.6f}")

        s2_opt = torch.optim.AdamW(model.halting_gate.parameters(), lr=1e-3, weight_decay=1e-4)
        exp_loops_running = 0.0
        s2_loss_running = 0.0

        for s2_step in range(1, args.stage2_steps + 1):
            cb = store.sample_balanced_batch(
                eps_per_task=args.eps_per_task,
                chunks_per_ep=max(2, args.chunks_per_ep // 2),
                cfg=cfg,
                vis_anchor_proj=model.vis_anchor_proj,
                gen=batch_gen,
            )
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                    ctx_c, cm_c = model.obs_encoder(
                        cb["curr_front"],
                        cb["curr_wrist"],
                        cb["curr_state"],
                        cb["txt_tok"],
                        cb["txt_mask"],
                        prev_front_emb=cb["prev_front"],
                        prev_wrist_emb=cb["prev_wrist"],
                        prev_state_norm=cb["prev_state"],
                    )
                    ms_c = build_batch_memory_state(model, cb)
                    crn_acc = torch.zeros(
                        (cb["curr_front"].shape[0], cfg.max_loops), device=device, dtype=torch.float32
                    )

            # Forward Coda readout so `halting_gate` receives live inputs (`Policy` params remain frozen!)
            with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                ro_c = model.run_coda_readout(ctx_c, cm_c, mem_state=ms_c)

            with torch.no_grad():
                coda_det = [c.detach() if c is not None else None for c in ro_c["coda_per_depth"]]
                with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                    for _ in range(args.m_crn):
                        lo_m = model.compute_crn_flow_and_aux_loss(
                            ctx_c, cm_c, coda_det, None, cb["target_actions"], eta_aux=0.0
                        )
                        crn_acc += lo_m["per_depth_fm_loss"]
                crn_mean = crn_acc / float(args.m_crn)

            s2_opt.zero_grad(set_to_none=True)
            assert ro_c["halt"] is not None
            s2_loss = model.halting_gate.compute_stage2_loss(
                ro_c["halt"],
                crn_mean,
                delta_margin=calib_delta,
                ponder_cost=args.ponder_cost,
            )
            s2_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.halting_gate.parameters(), 1.0)
            s2_opt.step()

            s2_loss_running += float(s2_loss.detach().item())
            exp_loops_running += float(ro_c["halt"]["expected_loops"].detach().mean().item())

            if s2_step % max(1, args.stage2_steps // 4) == 0 or s2_step == args.stage2_steps:
                denom_s2 = max(1, args.stage2_steps // 4) if s2_step < args.stage2_steps else (
                    (s2_step % max(1, args.stage2_steps // 4)) or max(1, args.stage2_steps // 4)
                )
                print(
                    f"[{arch} | Stage-II {s2_step:4d}/{args.stage2_steps}] "
                    f"loss={s2_loss_running/denom_s2:.5f} E[L_t]={exp_loops_running/denom_s2:.2f}"
                )
                stage2_metrics = {
                    "calibrated_delta": round(calib_delta, 6),
                    "final_stage2_loss": round(s2_loss_running / denom_s2, 6),
                    "mean_expected_loops": round(exp_loops_running / denom_s2, 3),
                    "stage2_wall_s": round(time.perf_counter() - t_s2, 2),
                }
                s2_loss_running = 0.0
                exp_loops_running = 0.0

    model.eval()
    total_wall_s = time.perf_counter() - t_start
    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated(device) / (1024**3), 3)
        if device.type == "cuda"
        else 0.0
    )

    metrics = {
        "arch": arch,
        "v3_mode": bool(cfg.v3_mode),
        "seed": args.seed,
        "steps": args.steps,
        "tasks": list(store.tasks),
        "parameter_breakdown": param_info,
        "state_size_bytes": state_bytes,
        "stage1_wall_s": round(stage1_wall_s, 2),
        "total_wall_s": round(total_wall_s, 2),
        "peak_vram_gb": peak_vram_gb,
        "final_log": history_logs[-1] if history_logs else {},
        "stage2": stage2_metrics,
        "history": history_logs,
    }

    norm_stats_dict = {
        "state": {
            "q01": store.normalizer.state_low.tolist(),
            "q99": store.normalizer.state_high.tolist(),
        },
        "actions": {
            "q01": store.action_q01_export,
            "q99": store.action_q99_export,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f".{arch}.pt.tmp"
    cpu_sd: Dict[str, torch.Tensor] = {}
    for k_s, v_s in model.state_dict().items():
        if (not k_s.startswith("halting_gate.")) and (k_s in ema.shadow):
            cpu_sd[k_s] = ema.shadow[k_s].detach().float().cpu()
        else:
            cpu_sd[k_s] = v_s.detach().float().cpu()
    payload = {
        "completed": True,
        "arch": arch,
        "model_config": cfg.to_dict(),
        "model_state_dict": cpu_sd,
        "action_mode": "quantile_delta",
        "norm_stats": norm_stats_dict,
        "metrics": metrics,
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(ckpt_path)
    if resume_path.exists():
        resume_path.unlink(missing_ok=True)

    metrics_json_path = out_dir / f"{arch}_metrics.json"
    metrics_json_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"[MODEL_DONE] {arch} -> {ckpt_path} "
        f"({ckpt_path.stat().st_size / (1024**2):.1f} MiB, wall={total_wall_s/60:.2f}m)"
    )
    return metrics


# =============================================================================
# 5. CLI Entrypoint
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Cycle 2A / Cycle 3 DeltaLoop-VLA & Controlled Baselines")
    p.add_argument(
        "--cache-dir",
        type=str,
        default="/work/gj26/b20098/datasets/robomme_siglip_v1",
        help="Path to pre-extracted multi-camera SigLIP shard directory",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "robomme" / "cycle2a"),
        help="Directory to write `.pt` checkpoints and metrics JSONs",
    )
    p.add_argument(
        "--v3",
        action="store_true",
        default=False,
        help="Enable Cycle 3 shared visuomotor upgrades (`U1-U5`, `v3_mode=True`). Default False preserves exact Cycle 2A behavior.",
    )
    p.add_argument(
        "--pilot-4tasks",
        action="store_true",
        default=False,
        help="Use the 4-task Cycle 3A pilot subset (`ButtonUnmaskSwap`, `VideoUnmaskSwap`, `StopCube`, `MoveCube`).",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=[
            "m0_memoryless",
            "m2_framesamp_modul",
            "m4_gdn_b2_l1",
            "m6_deltaloop_b2",
            "m7_deltaloop_2l_b2",
            "m1_rdvla_loop4",
            "m3_gdn_b1_l1",
            "m5_gdn2layer_b2_l1",
            "m6b_deltaloop_wb",
        ],
        choices=list(ALL_VLA_ARCHS),
        help="Ordered list of models to train (Priority Batch 1 first by default!)",
    )
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--stage2-steps", type=int, default=1000)
    p.add_argument("--m-crn", type=int, default=8)
    p.add_argument("--ponder-cost", type=float, default=0.003)

    p.add_argument("--eps-per-task", type=int, default=None, help="Default: 2 for 8 tasks, 4 for 4 tasks (= 16 eps/step)")
    p.add_argument("--chunks-per-ep", type=int, default=8, help="16 eps x 8 chunks = 128 chunks/step")
    p.add_argument("--wb-chain-chunks", type=int, default=6)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eta-aux", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-start-step", type=int, default=400)

    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--attn-heads", type=int, default=6)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument("--ctx-layers", type=int, default=2)
    p.add_argument("--expert-layers", type=int, default=6)
    p.add_argument("--num-write-slots", type=int, default=None, help="Default: 4 (v2) or 8 (--v3)")
    p.add_argument("--num-coda-tokens", type=int, default=4)
    p.add_argument("--write-stride", type=int, default=4)
    p.add_argument("--gdn-heads", type=int, default=8)
    p.add_argument("--gdn-d-k", type=int, default=None, help="Default: 64 (v2) or 128 (--v3)")
    p.add_argument("--gdn-d-v", type=int, default=None, help="Default: 64 (v2) or 128 (--v3)")
    p.add_argument("--max-loops", type=int, default=4)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-preload-gpu", action="store_true")
    p.add_argument("--bf16-weights", action="store_true", default=True)
    p.add_argument("--no-bf16-weights", dest="bf16_weights", action="store_false")
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--summary-name", type=str, default="cycle2a_train_summary.json")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument(
        "--smoke-test-only",
        action="store_true",
        help="Run full mathematical/streaming self-tests + checkpoint save/load smoke test without cache",
    )
    args = p.parse_args(argv)
    if args.tasks is None:
        args.tasks = list(PILOT_4_TASKS) if args.pilot_4tasks else list(CYCLE2A_TASKS)
    if args.eps_per_task is None:
        args.eps_per_task = max(1, 16 // max(1, len(args.tasks)))
    return args


def run_local_cpu_checkpoint_smoke_test(out_dir: Path, device: torch.device, v3_mode: bool = False) -> None:
    """Verify backward-compatible Cycle 2A load AND full checkpoint save -> `RoboMMEPolicyWrapper` + real SigLIP!"""
    c2a_dir = REPO_ROOT / "checkpoints" / "robomme" / "cycle2a"
    if c2a_dir.exists():
        mock_enc = _MockSiglipEncoder(device=device, dtype=torch.float32)
        verified_c2a = []
        for c2a_name in ("m0_memoryless.pt", "m2_framesamp_modul.pt", "m4_gdn_b2_l1.pt", "m6_deltaloop_b2.pt"):
            c2a_p = c2a_dir / c2a_name
            if c2a_p.exists():
                w_c2a = RoboMMEPolicyWrapper(
                    c2a_p, device=device, dtype=torch.float32, vision_encoder=mock_enc, ode_steps=2  # type: ignore[arg-type]
                )
                assert not w_c2a.cfg.v3_mode
                w_c2a.reset()
                f0 = np.zeros((256, 256, 3), dtype=np.uint8)
                s0 = np.zeros(8, dtype=np.float32)
                w_c2a.ingest_demo_frames([f0] * 4, [f0] * 4, [s0] * 4, "test prompt")
                res_c2a = w_c2a.predict_chunk(f0, f0, s0, "test prompt", step_idx=4, exec_start_idx=4)
                assert res_c2a.actions.shape == (20, 8) and np.isfinite(res_c2a.actions).all()
                verified_c2a.append(c2a_name)
        if verified_c2a:
            print(f"[Smoke-Test] Cycle 2A backward compatibility (`strict=True`) PASS on {verified_c2a}!")

    print(
        f"[Smoke-Test] Verifying full checkpoint save + `RoboMMEPolicyWrapper` + real `SiglipFeatureEncoder` (v3_mode={v3_mode})..."
    )
    cfg = DeltaLoopVLAConfig.v3_preset("m6_deltaloop_b2") if v3_mode else DeltaLoopVLAConfig(arch="m6_deltaloop_b2")
    model = DeltaLoopVLAPolicy(cfg).to(device=device)
    smoke_ckpt = out_dir / f"_smoke_test_m6_v{'3' if v3_mode else '2'}.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    if v3_mode:
        q01_b = np.array(MME_VLA_NORM_STATS["actions"]["q01"], dtype=np.float32)
        q99_b = np.array(MME_VLA_NORM_STATS["actions"]["q99"], dtype=np.float32)
        ns_smoke: Dict[str, Any] = {
            "state": MME_VLA_NORM_STATS["state"],
            "actions": {
                "q01": np.broadcast_to(q01_b, (20, 8)).tolist(),
                "q99": np.broadcast_to(q99_b, (20, 8)).tolist(),
            },
        }
    else:
        ns_smoke = MME_VLA_NORM_STATS
    torch.save(
        {
            "completed": True,
            "arch": cfg.arch,
            "model_config": cfg.to_dict(),
            "model_state_dict": model.state_dict(),
            "action_mode": "quantile_delta",
            "norm_stats": ns_smoke,
            "metrics": {},
        },
        smoke_ckpt,
    )
    try:
        dt = torch.bfloat16 if device.type == "cuda" else torch.float32
        wrapper = RoboMMEPolicyWrapper(smoke_ckpt, device=device, dtype=dt, ode_steps=2)
        wrapper.reset()
        rng = np.random.default_rng(0)
        demo_f = [rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8) for _ in range(8)]
        demo_w = [rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8) for _ in range(8)]
        demo_s = [np.zeros(8, dtype=np.float32) for _ in range(8)]
        wrapper.ingest_demo_frames(demo_f, demo_w, demo_s, "pick up the red cube")
        out = wrapper.predict_chunk(
            demo_f[0],
            demo_w[0],
            demo_s[0],
            "pick up the red cube",
            step_idx=8,
            exec_start_idx=8,
        )
        out2 = wrapper.predict_chunk(
            demo_f[1],
            demo_w[1],
            demo_s[1],
            "pick up the red cube",
            step_idx=9,
            exec_start_idx=8,
        )
        assert out.actions.shape == (20, 8) and np.isfinite(out.actions).all()
        assert out2.actions.shape == (20, 8) and np.isfinite(out2.actions).all()
        print(
            f"[Smoke-Test] End-to-End Wrapper + SigLIP OK (v3_mode={v3_mode})! "
            f"actions.shape={out.actions.shape}, loop_depth={out.loop_depth}, aux={out.aux_info}"
        )
    finally:
        if smoke_ckpt.exists():
            smoke_ckpt.unlink()


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    train_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    out_dir = Path(args.output_dir)
    self_test_report = run_mathematical_self_tests(device=device, dtype=train_dtype)

    if args.smoke_test_only:
        run_local_cpu_checkpoint_smoke_test(out_dir=out_dir, device=device, v3_mode=bool(args.v3))
        return

    store = GPURoboMMEStore(
        args.cache_dir,
        tasks=args.tasks,
        device=device,
        feature_dtype=torch.bfloat16,
        preload_gpu=(not args.no_preload_gpu),
        v3_mode=bool(args.v3),
        action_horizon=20,
        subgoal_dim=64,
    )

    summary_all: Dict[str, Any] = {
        "self_tests": self_test_report,
        "models": {},
    }
    for arch in args.models:
        m_res = train_single_model(
            arch=arch,  # type: ignore[arg-type]
            store=store,
            args=args,
            device=device,
            train_dtype=train_dtype,
            out_dir=out_dir,
        )
        summary_all["models"][arch] = {
            k: v for k, v in m_res.items() if k != "history"
        }
        (out_dir / args.summary_name).write_text(json.dumps(summary_all, indent=2))

    print("\n[ALL_MODELS_COMPLETE] Summary written to", out_dir / args.summary_name)


if __name__ == "__main__":
    main()
