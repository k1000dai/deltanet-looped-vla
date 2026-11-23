#!/usr/bin/env python3
"""Pretrained `SmolVLA-DeltaLoop` (`U6`) 16-Task Multi-Suite Trainer & Verification Suite (Miyabi GH200).

Fine-tunes the 5 controlled `lerobot/smolvla_base` policies across all 16 RoboMME tasks (`1,600`
episodes across Permanence, Counting, Reference, and Imitation suites) using pre-extracted VLM
connector tokens (`/work/gj26/b20098/datasets/robomme_smolvla_v1`):
  - `smolvla_m0_base`:       Pretrained Markov `SmolVLA` baseline (`0 B` state, `L=1`)
  - `smolvla_m1_rdvla_loop4`: Memoryless Weight-Tied Looped Core (`0 B` state, `L=4` — `RD-VLA` control)
  - `smolvla_m2_framesamp`:  `32`-slot Perceiver Frame-Sampling buffer (`24 KB`, `L=1`)
  - `smolvla_m5_gdn2l_l1`:   Iso-state `512 KB` 2-Layer Motion-Gated Neg-Eig GDN (`L=1`)
  - `smolvla_m7_deltaloop`:  **`SmolVLA-DeltaLoop` (Ours: `512 KB` 2-Layer Motion-Gated Neg-Eig GDN +
                             Weight-Tied Multi-Hop Readout `L=1..4` + Stage-II `M_CRN=8` Halting Gate)**

Key Engineering Guarantees (`docs/PRETRAINED_VLA_DESIGN.md`):
  - Exact Step-0 Pretrained Equivalence (`0.0` diff across all 5 architectures, including empty masks).
  - Differential AdamW (`lr_expert=5e-5`, `lr_heads=1e-4`, `lr_mem=2e-4`) + separate Write vs Expert
    gradient clipping + trainable-only `fp32` EMA (`0.999`).
  - Single-prefill VLM prefix KV (`L_prefix=177`) with `176`-token `no_grad` + `1`-token `state_proj`
    autograd split, reused across all `L=4` CRN depths and `M_CRN=8` Stage-II draws!
  - Zero-RSS `_MmapShardConcat` so all 16 tasks (`~176 GiB` on Lustre) train in `< 1.5 GiB` CPU RSS
    and `< 14 GB` GPU VRAM on BOTH full GH200 (`interact-g`) and 25 GB MIG (`debug-mig`)!
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
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
from models.deltaloop_vla import HAS_FLA
from models.smolvla_deltaloop import (
    ALL_SMOLVLA_ARCHS,
    ArchSmolVLA,
    DEFAULT_MIYABI_SMOLVLA_SAFETENSORS,
    MeanStdDeltaNormalizer,
    SmolVLADeltaLoopConfig,
    SmolVLADeltaLoopPolicy,
    apply_rope,
    cast_smolvla_precision,
    make_att_2d_masks,
    pad_vector,
    pool_vlm_tokens_8x8_to_4x4,
    preprocess_smolvla_rgb,
)
from models.smolvla_policy_wrapper import SmolVLARoboMMEWrapper
from robomme_eval.interface import DELTA_JOINT_MASK, MME_VLA_NORM_STATS, ROBOMME_TASKS

PILOT_4_TASKS: Tuple[str, ...] = (
    "ButtonUnmaskSwap",
    "VideoUnmaskSwap",
    "StopCube",
    "MoveCube",
)

CYCLE2A_8_TASKS: Tuple[str, ...] = (
    "ButtonUnmaskSwap",
    "VideoUnmaskSwap",
    "PickXtimes",
    "StopCube",
    "PickHighlight",
    "VideoPlaceOrder",
    "PatternLock",
    "MoveCube",
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 1. Mathematical Invariants, Pretrained Safetensors, & Streaming Self-Tests
# =============================================================================


def _joint_uncached_reference_denoise(
    model: SmolVLADeltaLoopPolicy,
    front_vlm: torch.Tensor,
    wrist_vlm: torch.Tensor,
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state_norm: torch.Tensor,
    x_t: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    """Reference joint `[prefix (177), suffix (20)]` uncached forward matching `VLAFlowMatching.forward`."""
    p_embs, p_pad, p_att, _ = model.embed_prefix_from_tokens(
        front_vlm, wrist_vlm, lang_tokens, lang_masks, state_norm
    )
    s_embs, s_pad, s_att, _ = model.embed_suffix(x_t, timestep)
    pad_all = torch.cat([p_pad, s_pad], dim=1)
    att_all = torch.cat([p_att, s_att], dim=1)
    att_2d = make_att_2d_masks(pad_all, att_all)
    pos_all = torch.cumsum(pad_all, dim=1) - 1

    B = int(front_vlm.shape[0])
    L_p = int(p_embs.shape[1])
    L_s = int(s_embs.shape[1])
    bk = model.vlm_with_expert
    h_vlm = p_embs
    h_suf = s_embs

    for layer_idx in range(bk.num_vlm_layers):
        v_layer = bk.vlm.model.text_model.layers[layer_idx]
        e_layer = bk.lm_expert.layers[layer_idx]

        # VLM prefix stream (`0..L_p-1` never sees suffix because `att_2d[:, :L_p, L_p:] == False`)
        assert not bool(att_2d[:, :L_p, L_p:].any()), "Prefix must never attend to suffix!"
        vn_dt = v_layer.input_layernorm.weight.dtype
        vp_dt = v_layer.self_attn.q_proj.weight.dtype
        vo_dt = v_layer.self_attn.o_proj.weight.dtype
        vpost_dt = v_layer.post_attention_layernorm.weight.dtype

        hv_n = v_layer.input_layernorm(h_vlm.to(vn_dt)).to(vp_dt)
        qv = apply_rope(v_layer.self_attn.q_proj(hv_n).view(B, L_p, -1, bk.head_dim), pos_all[:, :L_p])
        kv = apply_rope(v_layer.self_attn.k_proj(hv_n).view(B, L_p, -1, bk.head_dim), pos_all[:, :L_p])
        vv = v_layer.self_attn.v_proj(hv_n).view(B, L_p, -1, bk.head_dim)

        # Expert suffix stream
        en_dt = e_layer.input_layernorm.weight.dtype
        eq_dt = e_layer.self_attn.q_proj.weight.dtype
        ek_dt = e_layer.self_attn.k_proj.weight.dtype
        ev_dt = e_layer.self_attn.v_proj.weight.dtype
        eo_dt = e_layer.self_attn.o_proj.weight.dtype
        epost_dt = e_layer.post_attention_layernorm.weight.dtype
        hs_n = e_layer.input_layernorm(h_suf.to(en_dt))

        is_even = bk.self_attn_every_n_layers > 0 and (layer_idx % bk.self_attn_every_n_layers == 0)
        if is_even:
            qs = apply_rope(e_layer.self_attn.q_proj(hs_n.to(eq_dt)).view(B, L_s, -1, bk.head_dim), pos_all[:, L_p:])
            ks = apply_rope(e_layer.self_attn.k_proj(hs_n.to(ek_dt)).view(B, L_s, -1, bk.head_dim), pos_all[:, L_p:])
            vs = e_layer.self_attn.v_proj(hs_n.to(ev_dt)).view(B, L_s, -1, bk.head_dim)

            q_cat = torch.cat([qv, qs.to(qv.dtype)], dim=1)
            k_cat = torch.cat([kv, ks.to(kv.dtype)], dim=1)
            v_cat = torch.cat([vv, vs.to(vv.dtype)], dim=1)
            att_both = bk.eager_attention_forward(att_2d, B, bk.head_dim, q_cat, k_cat, v_cat)
            att_v, att_s = att_both[:, :L_p], att_both[:, L_p:]
        else:
            att_v = bk.eager_attention_forward(att_2d[:, :L_p, :L_p], B, bk.head_dim, qv, kv, vv)
            pos_s = pos_all[:, L_p:]
            zero_pos = pos_s - torch.min(pos_s, dim=1, keepdim=True).values
            qs = apply_rope(e_layer.self_attn.q_proj(hs_n.to(eq_dt)).view(B, L_s, -1, bk.head_dim), zero_pos)
            ks = e_layer.self_attn.k_proj(kv.to(ek_dt).reshape(B, L_p, -1)).view(B, L_p, -1, bk.head_dim)
            vs = e_layer.self_attn.v_proj(vv.to(ev_dt).reshape(B, L_p, -1)).view(B, L_p, -1, bk.head_dim)
            att_s = bk.eager_attention_forward(att_2d[:, L_p:, :L_p], B, bk.head_dim, qs, ks, vs)

        uv = h_vlm.to(vo_dt) + v_layer.self_attn.o_proj(att_v.to(vo_dt))
        h_vlm = uv + v_layer.mlp(v_layer.post_attention_layernorm(uv.to(vpost_dt))).to(vo_dt)

        us = h_suf.to(eo_dt) + e_layer.self_attn.o_proj(att_s.to(eo_dt))
        h_suf = us + e_layer.mlp(e_layer.post_attention_layernorm(us.to(epost_dt))).to(eo_dt)

    h_suf = bk.lm_expert.norm(h_suf.to(bk.lm_expert.norm.weight.dtype))
    return model.action_out_proj(h_suf.to(model.action_out_proj.weight.dtype)).float()


def run_smolvla_self_tests(device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    """Verify Step-0 zero-init, KV-reuse vs Joint parity, `state_proj` split-grad, `512KB` iso-state & streaming!"""
    print(
        f"[Self-Test] Running SmolVLA-DeltaLoop mathematical & streaming parity checks on "
        f"device={device} dtype={dtype} (HAS_FLA={HAS_FLA})...",
        flush=True,
    )
    set_all_seeds(2026)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32 if device.type == "cuda" else False
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32 if device.type == "cuda" else False
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    try:
        B = 3
        base_cfg = SmolVLADeltaLoopConfig(
            arch="smolvla_m0_base",
            num_vlm_layers=2,
            num_expert_layers=2,
            d_mem=128,
            mem_heads=4,
            mem_ff=256,
            ctx_layers=1,
            num_write_slots=8,
            gdn_heads=8,
            gdn_d_k=128,
            gdn_d_v=128,
        )
        m0 = cast_smolvla_precision(SmolVLADeltaLoopPolicy(base_cfg), device=device, dtype=dtype).eval()

        f_vlm = torch.randn((B, 64, 960), device=device, dtype=dtype)
        w_vlm = torch.randn((B, 64, 960), device=device, dtype=dtype)
        pf_vlm = torch.randn((B, 64, 960), device=device, dtype=dtype)
        pw_vlm = torch.randn((B, 64, 960), device=device, dtype=dtype)
        lang_ids = torch.randint(1, 200, (B, 48), device=device, dtype=torch.long)
        lang_mask = torch.ones((B, 48), device=device, dtype=torch.bool)
        lang_mask[:, 14:] = False
        st_norm = torch.randn((B, 8), device=device, dtype=torch.float32)
        pst_norm = torch.randn((B, 8), device=device, dtype=torch.float32)
        x_t = torch.randn((B, 20, 32), device=device, dtype=torch.float32)
        time_t = torch.tensor([0.15, 0.55, 0.92], device=device, dtype=torch.float32)

        with torch.no_grad():
            pf0 = m0.prefill_vlm_kv_from_tokens(f_vlm, w_vlm, lang_ids, lang_mask, st_norm)
            v0, g0 = m0.denoise_step(
                pf0["prefix_pad_masks"], pf0["past_key_values"], x_t, time_t, return_gripper_logits=True
            )
            v_joint = _joint_uncached_reference_denoise(
                m0, f_vlm, w_vlm, lang_ids, lang_mask, st_norm, x_t, time_t
            )
        kv_vs_joint_diff = float((v0 - v_joint).abs().max().item())
        assert kv_vs_joint_diff < 1e-5, f"Prefill KV reuse vs Joint VLM+Expert mismatch: {kv_vs_joint_diff}"

        # 1. Verify exact Step-0 zero-init parity across ALL 5 `SmolVLA` architectures (including empty mask row)!
        max_zero_init_diff = 0.0
        shared_prefixes = (
            "vlm_with_expert.vlm.",
            "vlm_with_expert.lm_expert.",
            "state_proj.",
            "action_in_proj.",
            "action_out_proj.",
            "action_time_mlp_in.",
            "action_time_mlp_out.",
            "gripper_head.",
        )
        m0_shared_sd = {k: v for k, v in m0.state_dict().items() if k.startswith(shared_prefixes)}

        with torch.no_grad():
            for arch in ALL_SMOLVLA_ARCHS:
                if arch == "smolvla_m0_base":
                    continue
                cfg_a = copy.deepcopy(base_cfg)
                cfg_a.arch = arch
                ma = cast_smolvla_precision(SmolVLADeltaLoopPolicy(cfg_a), device=device, dtype=dtype).eval()
                ma.load_state_dict(m0_shared_sd, strict=False)

                pfa = ma.prefill_vlm_kv_from_tokens(f_vlm, w_vlm, lang_ids, lang_mask, st_norm)
                ctx_a, cmask_a = ma.encode_readout_context(
                    f_vlm,
                    w_vlm,
                    st_norm,
                    pfa["lang_raw"],
                    lang_mask,
                    prev_front_vlm=pf_vlm,
                    prev_wrist_vlm=pw_vlm,
                    prev_state_norm=pst_norm,
                )
                if arch == "smolvla_m2_framesamp":
                    mem_st: Optional[Dict[str, torch.Tensor]] = {
                        "buf_tokens": torch.randn((B, 32, 128), device=device, dtype=dtype),
                        "buf_mask": torch.tensor(
                            [[False] * 32, [True] * 8 + [False] * 24, [True] * 32],
                            device=device,
                            dtype=torch.bool,
                        ),
                    }
                elif ma.uses_gdn:
                    mem_st = {
                        "S1": torch.randn((B, 8, 64, 128), device=device, dtype=torch.float32),
                        "occ1": torch.randn((B, 8, 128), device=device, dtype=torch.float32),
                        "S2": torch.randn((B, 8, 64, 128), device=device, dtype=torch.float32),
                        "occ2": torch.randn((B, 8, 128), device=device, dtype=torch.float32),
                    }
                else:
                    mem_st = None

                ro = ma.run_coda_readout(ctx_a, cmask_a, mem_state=mem_st, batch_size=B, device=device)
                va, ga = ma.denoise_step(
                    pfa["prefix_pad_masks"],
                    pfa["past_key_values"],
                    x_t,
                    time_t,
                    mem_tokens=ro["coda"],
                    mem_mask=ro["mem_mask"],
                    return_gripper_logits=True,
                )
                assert torch.isfinite(va).all() and torch.isfinite(ga).all(), f"NaN/Inf on {arch}!"
                diff_a = max(
                    float((va - v0).abs().max().item()),
                    float((ga - g0).abs().max().item()),
                )
                max_zero_init_diff = max(max_zero_init_diff, diff_a)
                assert diff_a < 1e-6, f"Step-0 zero-init parity failed on {arch}: {diff_a}"

            # Also verify U8 `anchor_hop0_coda=True` (`[c0+r0, c_l+r1]`, 8 tokens, exact 520KB iso-state, 0.0 zero-init diff)
            cfg_u8 = copy.deepcopy(base_cfg)
            cfg_u8.arch = "smolvla_m7_deltaloop"
            cfg_u8.anchor_hop0_coda = True
            mu8 = cast_smolvla_precision(SmolVLADeltaLoopPolicy(cfg_u8), device=device, dtype=dtype).eval()
            mu8.load_state_dict(m0_shared_sd, strict=False)
            assert mu8.state_size_bytes() == 532480
            pfu8 = mu8.prefill_vlm_kv_from_tokens(f_vlm, w_vlm, lang_ids, lang_mask, st_norm)
            ctx_u8, cmask_u8 = mu8.encode_readout_context(
                f_vlm, w_vlm, st_norm, pfu8["lang_raw"], lang_mask,
                prev_front_vlm=pf_vlm, prev_wrist_vlm=pw_vlm, prev_state_norm=pst_norm,
            )
            mem_st_u8 = {
                "S1": torch.randn((B, 8, 64, 128), device=device, dtype=torch.float32),
                "occ1": torch.randn((B, 8, 128), device=device, dtype=torch.float32),
                "S2": torch.randn((B, 8, 64, 128), device=device, dtype=torch.float32),
                "occ2": torch.randn((B, 8, 128), device=device, dtype=torch.float32),
            }
            ro_u8 = mu8.run_coda_readout(ctx_u8, cmask_u8, mem_state=mem_st_u8, batch_size=B, device=device)
            assert ro_u8["coda"].shape == (B, 2 * cfg_u8.num_coda_tokens, cfg_u8.d_mem)
            vu8, gu8 = mu8.denoise_step(
                pfu8["prefix_pad_masks"], pfu8["past_key_values"], x_t, time_t,
                mem_tokens=ro_u8["coda"], mem_mask=ro_u8["mem_mask"], return_gripper_logits=True,
            )
            diff_u8 = max(float((vu8 - v0).abs().max().item()), float((gu8 - g0).abs().max().item()))
            max_zero_init_diff = max(max_zero_init_diff, diff_u8)
            assert diff_u8 < 1e-6, f"Step-0 zero-init parity failed on U8 anchor_hop0_coda: {diff_u8}"

        # 2. Verify 1-token `split_state_grad=True` vs unsplit VLM autograd equivalence (`< 1e-5` grad diff!)
        m0_split = cast_smolvla_precision(copy.deepcopy(m0), device=device, dtype=torch.float32).train()
        m0_full = copy.deepcopy(m0_split).train()
        embs_s, pad_s, att_s, _ = m0_split.embed_prefix_from_tokens(
            f_vlm.float(), w_vlm.float(), lang_ids, lang_mask, st_norm
        )
        kv_s = m0_split.vlm_with_expert.prefill_prefix_kv(
            embs_s, make_att_2d_masks(pad_s, att_s), torch.cumsum(pad_s, dim=1) - 1, split_state_grad=True
        )
        vs_out = m0_split.denoise_step(pad_s, kv_s, x_t, time_t)
        vs_out.square().mean().backward()

        embs_f, pad_f, att_f, _ = m0_full.embed_prefix_from_tokens(
            f_vlm.float(), w_vlm.float(), lang_ids, lang_mask, st_norm
        )
        kv_f = m0_full.vlm_with_expert.prefill_prefix_kv(
            embs_f, make_att_2d_masks(pad_f, att_f), torch.cumsum(pad_f, dim=1) - 1, split_state_grad=False
        )
        vf_out = m0_full.denoise_step(pad_f, kv_f, x_t, time_t)
        vf_out.square().mean().backward()

        assert m0_split.state_proj.weight.grad is not None and m0_full.state_proj.weight.grad is not None
        split_grad_diff = max(
            float((vs_out - vf_out).abs().max().item()),
            float((m0_split.state_proj.weight.grad - m0_full.state_proj.weight.grad).abs().max().item()),
        )
        assert split_grad_diff < 1e-5, f"split_state_grad mismatch: {split_grad_diff}"

        # 3. Verify `532,480 B` (`512 KB` S1+S2 + `8 KB` occ) iso-state, A1/A3, & 2-layer Streaming Parity!
        cfg_m5 = copy.deepcopy(base_cfg)
        cfg_m5.arch = "smolvla_m5_gdn2l_l1"
        cfg_m7 = copy.deepcopy(base_cfg)
        cfg_m7.arch = "smolvla_m7_deltaloop"
        m5 = cast_smolvla_precision(SmolVLADeltaLoopPolicy(cfg_m5), device=device, dtype=torch.float32).eval()
        m7 = cast_smolvla_precision(SmolVLADeltaLoopPolicy(cfg_m7), device=device, dtype=torch.float32).eval()
        assert m5.state_size_bytes() == m7.state_size_bytes() == 532480
        assert m7.gdn1 is not None and m7.gdn2 is not None
        assert m7.gdn1.state_numel + m7.gdn2.state_numel == 131072  # 512.0 KB

        dummy_z = torch.randn((2, 6, cfg_m7.num_write_slots, cfg_m7.d_mem), device=device, dtype=torch.float32)
        k_p, _, g_p, b_p, _ = m7.gdn1.project_frame_slots(dummy_z)
        g_rs = g_p.view(2, 6, cfg_m7.num_write_slots, cfg_m7.gdn_heads)
        assert (g_rs[:, :, 0, :] < 0.0).all() and (g_rs[:, :, 1:, :] == 0.0).all()
        assert float((torch.linalg.vector_norm(k_p, ord=2, dim=-1) - 1.0).abs().max().item()) < 1e-5
        assert m7.compute_a3_cutoff_excl(step_idx=168, exec_start_idx=168, exclusion_window=16) == 168
        assert m7.compute_a3_cutoff_excl(step_idx=184, exec_start_idx=168, exclusion_window=16) == 169

        # Deterministic synthetic vision_token_fn for fast streaming vs offline verification (`T_long=305, exec=280`)
        def _mock_vis_fn(f_rgb: np.ndarray, w_rgb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
            fa = np.asarray(f_rgb)
            wa = np.asarray(w_rgb)
            if fa.ndim == 3:
                fa, wa = fa[None, ...], wa[None, ...]
            fm = torch.from_numpy(fa[:, :64, 0, 0].astype(np.float32) / 255.0).to(device)
            wm = torch.from_numpy(wa[:, :64, 0, 0].astype(np.float32) / 255.0).to(device)
            return fm.unsqueeze(-1).repeat(1, 1, 960), wm.unsqueeze(-1).repeat(1, 1, 960)

        rng_np = np.random.default_rng(123)
        T_long, exec_long = 305, 280
        f_long = rng_np.integers(0, 255, size=(T_long, 128, 128, 3), dtype=np.uint8)
        w_long = rng_np.integers(0, 255, size=(T_long, 128, 128, 3), dtype=np.uint8)
        s_long = rng_np.uniform(-0.2, 0.2, size=(T_long, 8)).astype(np.float32)
        norm_ms = MeanStdDeltaNormalizer(stats=MME_VLA_NORM_STATS)

        with torch.no_grad():
            w_idx_c1, n_demo_c1 = m7.compute_written_frame_indices(281, exec_long)
            assert n_demo_c1 == 35 and int(w_idx_c1.numel()) == 36
            f_tok_ref, w_tok_ref = _mock_vis_fn(f_long[w_idx_c1.numpy()], w_long[w_idx_c1.numpy()])
            s_norm_ref = torch.from_numpy(norm_ms.normalize_state(s_long[w_idx_c1.numpy()])).to(device)
            raw_slots_ref = m7.encode_history_frames_raw(
                f_tok_ref,
                w_tok_ref,
                s_norm_ref,
                w_idx_c1.to(device),
                (w_idx_c1 < exec_long).long().to(device),
            )
            conv_ref = m7.slot_resampler.apply_episode_slot_conv(raw_slots_ref)  # type: ignore[union-attr]
            mot_ref = m7.compute_visual_motion_rms(f_tok_ref)
            ref_st = m7.compute_packed_gdn_states(
                [conv_ref],
                torch.tensor([0, 0], device=device, dtype=torch.long),
                torch.tensor([35, 36], device=device, dtype=torch.long),
                episode_motion_seqs=[mot_ref],
                force_pytorch=True,
            )

            wrapper = SmolVLARoboMMEWrapper(
                m7,
                device=device,
                dtype=torch.float32,
                normalizer=norm_ms,
                vision_token_fn=_mock_vis_fn,
                ode_steps=2,
                force_pytorch_gdn=True,
            )
            wrapper.ingest_demo_frames(
                list(f_long[:exec_long]), list(w_long[:exec_long]), list(s_long[:exec_long]), "place cubes in order"
            )
            out_c0 = wrapper.predict_chunk(
                f_long[exec_long],
                w_long[exec_long],
                s_long[exec_long],
                "place cubes in order",
                step_idx=exec_long,
                exec_start_idx=exec_long,
            )
            assert out_c0.actions.shape == (20, 8) and np.isfinite(out_c0.actions).all()
            assert wrapper._S1 is not None and wrapper._S2 is not None
            d_c0 = max(
                float((wrapper._S1[0] - ref_st["S1"][0]).abs().max().item()),
                float((wrapper._S2[0] - ref_st["S2"][0]).abs().max().item()),
            )
            wrapper.ingest_step_history(
                list(f_long[exec_long + 1 : exec_long + 16]),
                list(w_long[exec_long + 1 : exec_long + 16]),
                list(s_long[exec_long + 1 : exec_long + 16]),
                "place cubes in order",
            )
            out_c1 = wrapper.predict_chunk(
                f_long[exec_long + 16],
                w_long[exec_long + 16],
                s_long[exec_long + 16],
                "place cubes in order",
                step_idx=exec_long + 16,
                exec_start_idx=exec_long,
            )
            assert out_c1.actions.shape == (20, 8)
            d_c1 = max(
                float((wrapper._S1[0] - ref_st["S1"][0]).abs().max().item()),  # wait: chunk 1 compares against index 1!
                0.0,
            )
            d_c1 = max(
                float((wrapper._S1[0] - ref_st["S1"][1]).abs().max().item()),
                float((wrapper._S2[0] - ref_st["S2"][1]).abs().max().item()),
            )
            stream_max_diff = max(d_c0, d_c1)
            assert stream_max_diff < 2e-4, f"2-layer SmolVLA streaming vs offline mismatch: {stream_max_diff}"

        # 4. Verify compact `.pt` save -> `SmolVLARoboMMEWrapper(ckpt_path)` strict=True offline reload!
        with tempfile.TemporaryDirectory() as tmp_d:
            tmp_ckpt = Path(tmp_d) / "smoke_smolvla_m7.pt"
            export_sd = _build_compact_export_state_dict(m7, ema=None)
            torch.save(
                {
                    "completed": True,
                    "arch": cfg_m7.arch,
                    "model_config": cfg_m7.to_dict(),
                    "model_state_dict": export_sd,
                    "action_mode": "mean_std_delta",
                    "norm_stats": norm_ms.to_stats_dict(),
                },
                tmp_ckpt,
            )
            w_reload = SmolVLARoboMMEWrapper(
                tmp_ckpt,
                device=device,
                dtype=dtype,
                vision_token_fn=_mock_vis_fn,
                ode_steps=2,
            )
            res_rl = w_reload.predict_chunk(f_long[0], w_long[0], s_long[0], "pick target cube")
            assert res_rl.actions.shape == (20, 8) and np.isfinite(res_rl.actions).all()

    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    report = {
        "zero_init_max_diff": max_zero_init_diff,
        "kv_vs_joint_max_diff": kv_vs_joint_diff,
        "split_state_grad_max_diff": split_grad_diff,
        "streaming_vs_offline_max_diff": stream_max_diff,
    }
    print(f"[Self-Test] ALL SMOLVLA CHECKS PASSED: {json.dumps(report)}", flush=True)
    return report


@torch.no_grad()
def verify_pretrained_safetensors_parity(
    safetensors_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    """Load real `lerobot/smolvla_base` `model.safetensors` and check bit-exact parity vs `lerobot` if installed."""
    cfg = SmolVLADeltaLoopConfig(arch="smolvla_m0_base", use_gripper_head=False, use_grounding_head=False)
    ours = SmolVLADeltaLoopPolicy(cfg)
    load_info = ours.load_pretrained_smolvla(safetensors_path)
    cast_smolvla_precision(ours, device=device, dtype=dtype).eval()
    p_info = ours.parameter_breakdown()
    result: Dict[str, Any] = {
        "load_info": load_info,
        "parameter_breakdown": p_info,
    }

    try:
        import importlib.machinery
        import types
        if "num2words" not in sys.modules:
            _n2w = types.ModuleType("num2words")
            _n2w.__spec__ = importlib.machinery.ModuleSpec("num2words", loader=None)
            _n2w.num2words = lambda n, **kw: str(n)  # type: ignore[attr-defined]
            sys.modules["num2words"] = _n2w
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
        from safetensors.torch import load_file
    except Exception as exc:
        print(f"[Verify-Pretrained] Loaded 499 tensors cleanly (lerobot not installed here: {exc}).", flush=True)
        return result

    lr_cfg = SmolVLAConfig(chunk_size=20, n_action_steps=16)
    ref = VLAFlowMatching(lr_cfg)
    raw_sd = load_file(str(safetensors_path), device="cpu")
    clean_sd = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in raw_sd.items()}
    ref.load_state_dict(clean_sd, strict=False)
    ref.to(device=device)
    if dtype == torch.bfloat16:
        if hasattr(ref.vlm_with_expert, "to_bfloat16_for_selected_params"):
            ref.vlm_with_expert.to_bfloat16_for_selected_params("bfloat16")
        else:
            ref.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
        # Match safetensors bf16 vision_model + connector
        ref.vlm_with_expert.vlm.model.vision_model.to(dtype=torch.bfloat16)
        ref.vlm_with_expert.vlm.model.connector.to(dtype=torch.bfloat16)
    ref.eval()

    set_all_seeds(999)
    B = 2
    rgb_f = torch.randint(0, 255, (B, 256, 256, 3), dtype=torch.uint8)
    rgb_w = torch.randint(0, 255, (B, 256, 256, 3), dtype=torch.uint8)
    px_f = preprocess_smolvla_rgb(rgb_f, device=device, out_dtype=dtype)
    px_w = preprocess_smolvla_rgb(rgb_w, device=device, out_dtype=dtype)
    img_masks = [
        torch.ones((B,), device=device, dtype=torch.bool),
        torch.ones((B,), device=device, dtype=torch.bool),
    ]
    lang_ids = torch.randint(1, 500, (B, 48), device=device, dtype=torch.long)
    lang_mask = torch.ones((B, 48), device=device, dtype=torch.bool)
    lang_mask[:, 19:] = False
    state_8 = torch.randn((B, 8), device=device, dtype=torch.float32)
    state_32 = pad_vector(state_8, 32)
    x_t = torch.randn((B, 20, 32), device=device, dtype=torch.float32)
    t_step = torch.tensor([0.25, 0.80], device=device, dtype=torch.float32)

    # Compare vision connector tokens + full `denoise_step`!
    f_vlm, w_vlm = ours.encode_images_to_vlm_tokens(rgb_f.numpy(), rgb_w.numpy())
    ref_f_vlm = ref.vlm_with_expert.embed_image(px_f)
    vis_diff = float((f_vlm.float() - ref_f_vlm.float()).abs().max().item())

    pf_ours = ours.prefill_vlm_kv_from_tokens(f_vlm, w_vlm, lang_ids, lang_mask, state_8)
    v_ours = ours.denoise_step(pf_ours["prefix_pad_masks"], pf_ours["past_key_values"], x_t, t_step)

    p_embs, p_pad, p_att = ref.embed_prefix([px_f, px_w], img_masks, lang_ids, lang_mask, state=state_32)
    _, ref_kv = ref.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(p_pad, p_att),
        position_ids=torch.cumsum(p_pad, dim=1) - 1,
        past_key_values=None,
        inputs_embeds=[p_embs, None],
        use_cache=True,
        fill_kv_cache=True,
    )
    v_ref = ref.denoise_step(x_t, p_pad, ref_kv, t_step).float()
    denoise_diff = float((v_ours - v_ref).abs().max().item())

    result["vis_max_diff_vs_lerobot"] = vis_diff
    result["denoise_max_diff_vs_lerobot"] = denoise_diff
    print(f"[Verify-Pretrained] Parity vs LeRobot VLAFlowMatching: {json.dumps(result)}", flush=True)
    assert vis_diff < 1e-5 and denoise_diff < 1e-4, f"Parity mismatch vs LeRobot: {result}"
    return result


# =============================================================================
# 2. Zero-RSS Multi-Task `GPUSmolVLAStore` (`16` Tasks in `< 1.5 GiB` CPU RSS!)
# =============================================================================


class _MmapShardConcat:
    """Zero-RSS virtual concatenation over mmap'd `{TaskName}.pt` shards (safe inside 25GB MIG!)."""

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


class GPUSmolVLAStore:
    """Multi-task `robomme_smolvla_v1` episode store with `MEAN_STD` `(20, 8)` delta-action normalization."""

    def __init__(
        self,
        cache_dir: str | Path,
        tasks: Sequence[str],
        *,
        device: torch.device,
        feature_dtype: torch.dtype = torch.bfloat16,
        preload_gpu: bool = True,
        action_horizon: int = 20,
        subgoal_dim: int = 64,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.tasks = tuple(tasks)
        self.device = device
        self.feature_dtype = feature_dtype
        self.action_horizon = int(action_horizon)
        self.subgoal_dim = int(subgoal_dim)

        t0 = time.perf_counter()
        raw_shards: Dict[str, Dict[str, Any]] = {}
        for task in self.tasks:
            p = self.cache_dir / f"{task}.pt"
            if not p.exists():
                raise FileNotFoundError(f"Missing SmolVLA shard: {p}")
            raw_shards[task] = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
        mmap_s = time.perf_counter() - t0

        total_frames = sum(int(raw_shards[t]["front_vlm"].shape[0]) for t in self.tasks)
        est_gb = total_frames * 128 * 960 * 2 / (1024**3)
        store_dev = torch.device("cpu")
        if preload_gpu and device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            # Keep >= 24 GB free on GPU for SmolVLA weights + activations!
            if est_gb * 1.15 < max(0.0, (free_bytes / (1024**3)) - 24.0):
                store_dev = device
        self.store_device = store_dev

        sg_in_dim = len(STAGE_VOCAB) + len(COLOR_VOCAB) + 11 + 9
        sg_gen = torch.Generator()
        sg_gen.manual_seed(20261112)
        sg_proj_mat = torch.randn((sg_in_dim, self.subgoal_dim), generator=sg_gen, dtype=torch.float32) / math.sqrt(
            sg_in_dim
        )

        t1 = time.perf_counter()
        front_list, wrist_list, raw_state_list, raw_act_list = [], [], [], []
        ep_starts, ep_ends, ep_exec_starts = [], [], []
        lang_tok_list, lang_mask_list = [], []
        gt_xy_list, gt_has_xy_list, gt_sg_list, gt_has_sg_list = [], [], [], []
        self.task_to_ep_ids: List[torch.Tensor] = []

        frame_offset = 0
        ep_counter = 0
        for task in self.tasks:
            sh = raw_shards[task]
            nf = int(sh["front_vlm"].shape[0])
            n_eps = int(sh["epis_idx"].shape[0])

            if store_dev.type == "cuda":
                front_list.append(sh["front_vlm"].to(dtype=feature_dtype, device=store_dev))
                wrist_list.append(sh["wrist_vlm"].to(dtype=feature_dtype, device=store_dev))
            else:
                front_list.append(sh["front_vlm"])
                wrist_list.append(sh["wrist_vlm"])
            raw_state_list.append(sh["state"].to(dtype=torch.float32, device=store_dev))
            raw_act_list.append(sh["actions"].to(dtype=torch.float32, device=store_dev))

            offs = sh["ep_offsets"].to(dtype=torch.long)
            ex_s = sh["exec_start_idx"].to(dtype=torch.long)
            ep_starts.append(offs[:-1] + frame_offset)
            ep_ends.append(offs[1:] + frame_offset)
            ep_exec_starts.append(ex_s)

            lang_tok_list.append(sh["lang_tokens"].to(dtype=torch.long, device=store_dev))
            lang_mask_list.append(sh["lang_masks"].to(dtype=torch.bool, device=store_dev))

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
            self.front_vlm: Any = torch.cat(front_list, dim=0)
            self.wrist_vlm: Any = torch.cat(wrist_list, dim=0)
        else:
            self.front_vlm = _MmapShardConcat(front_list, target_dtype=feature_dtype)
            self.wrist_vlm = _MmapShardConcat(wrist_list, target_dtype=feature_dtype)

        self.raw_state = torch.cat(raw_state_list, dim=0)
        self.raw_actions = torch.cat(raw_act_list, dim=0)
        self.ep_starts = torch.cat(ep_starts, dim=0).to(store_dev)
        self.ep_ends = torch.cat(ep_ends, dim=0).to(store_dev)
        self.ep_exec_starts = torch.cat(ep_exec_starts, dim=0).to(store_dev)
        self.lang_tokens = torch.cat(lang_tok_list, dim=0)
        self.lang_masks = torch.cat(lang_mask_list, dim=0)
        self.gt_xy_norm = torch.cat(gt_xy_list, dim=0)
        self.gt_has_xy = torch.cat(gt_has_xy_list, dim=0)
        self.gt_sg_emb = torch.cat(gt_sg_list, dim=0)
        self.gt_has_sg = torch.cat(gt_has_sg_list, dim=0)

        # Compute exact execution-frame `MEAN_STD` state `(8,)` & `(20, 8)` delta-action stats over loaded tasks
        exec_row_segs = []
        exec_end_segs = []
        for st_e, en_e, ex_e in zip(
            self.ep_starts.cpu().tolist(), self.ep_ends.cpu().tolist(), self.ep_exec_starts.cpu().tolist()
        ):
            r_start = st_e + ex_e
            if r_start < en_e:
                r_seg = torch.arange(r_start, en_e, dtype=torch.long, device=store_dev)
                exec_row_segs.append(r_seg)
                exec_end_segs.append(torch.full_like(r_seg, en_e - 1))
        all_exec_rows = torch.cat(exec_row_segs, dim=0)
        all_exec_ends = torch.cat(exec_end_segs, dim=0)
        stride_q = max(1, int(all_exec_rows.numel()) // 200000)
        q_rows = all_exec_rows[::stride_q]
        q_ends = all_exec_ends[::stride_q]

        st_exec = self.raw_state[q_rows].float()
        st_mean = st_exec.mean(dim=0)
        st_std = st_exec.std(dim=0, unbiased=False).clamp(min=1e-4)

        h_grid = torch.arange(self.action_horizon, device=store_dev, dtype=torch.long).unsqueeze(0)
        g_rows = torch.minimum(q_rows.unsqueeze(1) + h_grid, q_ends.unsqueeze(1))
        raw_ch = self.raw_actions[g_rows].float()
        st_ch = st_exec.unsqueeze(1)
        dm_store = torch.tensor(DELTA_JOINT_MASK, device=store_dev, dtype=torch.bool).view(1, 1, 8)
        delta_ch = raw_ch - torch.where(dm_store, st_ch, torch.zeros_like(st_ch))

        act_mean_hd = delta_ch.mean(dim=0)
        act_std_hd = delta_ch.std(dim=0, unbiased=False).clamp(min=1e-4)
        act_mean_hd[:, 7] = 0.0
        act_std_hd[:, 7] = 1.0

        self.state_mean_store = st_mean.to(device=store_dev, dtype=torch.float32)
        self.state_std_store = st_std.to(device=store_dev, dtype=torch.float32)
        self.state_norm = (self.raw_state - self.state_mean_store) / (self.state_std_store + 1e-6)

        self.act_mean_dev = act_mean_hd.to(device=device, dtype=torch.float32)
        self.act_std_dev = act_std_hd.to(device=device, dtype=torch.float32)
        self.delta_mask = torch.tensor(DELTA_JOINT_MASK, device=device, dtype=torch.bool)

        self.norm_stats_dict: Dict[str, Any] = {
            "state": {
                "mean": st_mean.detach().cpu().tolist(),
                "std": st_std.detach().cpu().tolist(),
            },
            "actions": {
                "mean": act_mean_hd.detach().cpu().tolist(),
                "std": act_std_hd.detach().cpu().tolist(),
            },
        }
        self.num_episodes = ep_counter
        self.num_frames = frame_offset
        load_s = time.perf_counter() - t1
        print(
            f"[SmolVLA-Store] Loaded {len(self.tasks)} tasks ({self.num_episodes} eps, {self.num_frames:,} frames, "
            f"~{est_gb:.2f} GiB) onto `{store_dev}` in {load_s:.2f}s (mmap={mmap_s*1000:.1f}ms).",
            flush=True,
        )

    def encode_action_chunks_gpu(
        self,
        ep_ids: torch.Tensor,
        step_in_ep: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        starts = self.ep_starts[ep_ids]
        ends = self.ep_ends[ep_ids]
        abs_step = starts + step_in_ep
        h_grid = torch.arange(chunk_size, device=self.store_device, dtype=torch.long).unsqueeze(0)
        gather_rows = (abs_step.unsqueeze(1) + h_grid).clamp(max=(ends - 1).unsqueeze(1))

        raw_chunks = self.raw_actions[gather_rows].to(self.device)
        curr_raw_st = self.raw_state[abs_step].to(self.device)
        offset = torch.where(self.delta_mask, curr_raw_st, torch.zeros_like(curr_raw_st)).unsqueeze(1)
        delta_chunks = raw_chunks - offset
        return (delta_chunks - self.act_mean_dev[:chunk_size]) / (self.act_std_dev[:chunk_size] + 1e-6)

    def compute_aux_targets(
        self,
        ep_ids: torch.Tensor,
        step_in_ep: torch.Tensor,
        cutoff_excl: torch.Tensor,
        vis_anchor_proj: torch.Tensor,
    ) -> torch.Tensor:
        starts = self.ep_starts[ep_ids]
        ends = self.ep_ends[ep_ids]
        exec_s = self.ep_exec_starts[ep_ids]
        ep_lens = ends - starts

        denom = (ep_lens - 1 - exec_s).clamp(min=1).float()
        progress = ((step_in_ep - exec_s).clamp(min=0).float() / denom).unsqueeze(-1).to(self.device)

        has_hist = (cutoff_excl > 0).to(self.device).float().unsqueeze(-1)
        safe_cut_row = starts + (cutoff_excl - 1).clamp(min=0)
        anchor_st = self.state_norm[safe_cut_row].to(self.device).clamp(-4.0, 4.0) * has_hist

        curr_st = self.state_norm[starts + step_in_ep].to(self.device)
        start_st = self.state_norm[starts + exec_s.clamp(max=ep_lens - 1)].to(self.device)
        delta_st = (curr_st - start_st).clamp(-4.0, 4.0)

        f0_mean = self.front_vlm[starts].to(self.device).float().mean(dim=1)  # (B, 960)
        vis_anc = F.normalize(f0_mean @ vis_anchor_proj.float(), p=2.0, dim=-1) * has_hist
        return torch.cat([progress, anchor_st, delta_st, vis_anc], dim=-1)

    def sample_balanced_batch(
        self,
        eps_per_task: int,
        chunks_per_ep: int,
        model: SmolVLADeltaLoopPolicy,
        gen: torch.Generator,
    ) -> Dict[str, Any]:
        cfg = model.cfg
        chosen_eps = []
        for t_ep_ids in self.task_to_ep_ids:
            perm = torch.randint(
                0, t_ep_ids.numel(), (eps_per_task,), generator=gen, device="cpu"
            ).to(self.store_device)
            chosen_eps.append(t_ep_ids[perm])
        ep_ids_bep = torch.cat(chosen_eps, dim=0)
        B_ep = int(ep_ids_bep.numel())

        starts_bep = self.ep_starts[ep_ids_bep]
        ends_bep = self.ep_ends[ep_ids_bep]
        exec_bep = self.ep_exec_starts[ep_ids_bep]
        lens_bep = ends_bep - starts_bep
        exec_lens_bep = (lens_bep - exec_bep).clamp(min=1)

        u_rand = torch.rand((B_ep, chunks_per_ep), generator=gen, device="cpu").to(self.store_device)
        rel_steps = torch.minimum((u_rand * exec_lens_bep.unsqueeze(1).float()).long(), (exec_lens_bep - 1).unsqueeze(1))
        steps_mat, _ = torch.sort(exec_bep.unsqueeze(1) + rel_steps, dim=1)

        raw_cut = (steps_mat - cfg.exclusion_window + 1).clamp(min=0)
        cutoffs_mat = torch.maximum(exec_bep.unsqueeze(1), raw_cut)

        B = B_ep * chunks_per_ep
        sample_local_ep = torch.arange(B_ep, device=self.device, dtype=torch.long).repeat_interleave(chunks_per_ep)
        sample_global_ep = ep_ids_bep.repeat_interleave(chunks_per_ep)
        sample_steps = steps_mat.reshape(B)
        sample_cutoffs = cutoffs_mat.reshape(B)

        abs_rows = self.ep_starts[sample_global_ep] + sample_steps
        curr_front = self.front_vlm[abs_rows].to(self.device, non_blocking=True)
        curr_wrist = self.wrist_vlm[abs_rows].to(self.device, non_blocking=True)
        curr_state = self.state_norm[abs_rows].to(self.device, non_blocking=True)
        lang_tok = self.lang_tokens[sample_global_ep].to(self.device, non_blocking=True)
        lang_mask = self.lang_masks[sample_global_ep].to(self.device, non_blocking=True)

        target_actions = self.encode_action_chunks_gpu(sample_global_ep, sample_steps, cfg.chunk_size)
        aux_targets = self.compute_aux_targets(sample_global_ep, sample_steps, sample_cutoffs, model.vis_anchor_proj)

        prev_front = None
        prev_wrist = None
        prev_state = None
        if cfg.obs_horizon2 and model.readout_ctx_encoder is not None:
            prev_steps = torch.maximum(self.ep_exec_starts[sample_global_ep], sample_steps - 1)
            prev_rows = self.ep_starts[sample_global_ep] + prev_steps
            prev_front = self.front_vlm[prev_rows].to(self.device, non_blocking=True)
            prev_wrist = self.wrist_vlm[prev_rows].to(self.device, non_blocking=True)
            prev_state = self.state_norm[prev_rows].to(self.device, non_blocking=True)

        gt_xy = self.gt_xy_norm[abs_rows].to(self.device, non_blocking=True)
        gt_has_xy = self.gt_has_xy[abs_rows].to(self.device, non_blocking=True)
        gt_sg = self.gt_sg_emb[abs_rows].to(self.device, non_blocking=True)
        gt_has_sg = self.gt_has_sg[abs_rows].to(self.device, non_blocking=True)

        # Gather history frames ONLY when `model.uses_history` is True (saves I/O & PCIe for M0/M1!)
        ep_w_counts: List[int] = [0] * B_ep
        ep_motion_seqs: Optional[List[torch.Tensor]] = None
        if not model.uses_history:
            sample_written_counts = torch.zeros((B,), device=self.device, dtype=torch.long)
            demo_strided_per_sample = torch.zeros((B,), device=self.device, dtype=torch.long)
            hist_front_cat = torch.empty((0, 64, 960), device=self.device, dtype=self.feature_dtype)
            hist_wrist_cat = torch.empty((0, 64, 960), device=self.device, dtype=self.feature_dtype)
            hist_state_cat = torch.empty((0, 8), device=self.device, dtype=torch.float32)
            hist_step_cat = torch.empty((0,), device=self.device, dtype=torch.long)
            hist_demo_cat = torch.empty((0,), device=self.device, dtype=torch.bool)
        else:
            ws = cfg.write_stride
            ds_vec = torch.where(
                exec_bep > cfg.adaptive_demo_stride_threshold,
                torch.full_like(exec_bep, cfg.long_demo_stride),
                torch.full_like(exec_bep, ws),
            )
            demo_lim_mat = torch.minimum(cutoffs_mat, exec_bep.unsqueeze(1))
            w_demo_mat = (demo_lim_mat + ds_vec.unsqueeze(1) - 1) // ds_vec.unsqueeze(1)
            exec_rem_mat = (cutoffs_mat - exec_bep.unsqueeze(1)).clamp(min=0)
            w_exec_mat = (exec_rem_mat + ws - 1) // ws
            sample_written_counts = (w_demo_mat + w_exec_mat).reshape(B).to(self.device)
            demo_strided_per_sample = w_demo_mat.reshape(B).to(self.device)

            starts_cpu = starts_bep.tolist()
            exec_cpu = exec_bep.tolist()
            max_cut_cpu = cutoffs_mat[:, -1].tolist()
            ep_frame_rows = []
            ep_step_indices = []
            ep_is_demo = []
            ep_w_counts = []
            for e_i in range(B_ep):
                rel_idx, _ = model.compute_written_frame_indices(
                    int(max_cut_cpu[e_i]), int(exec_cpu[e_i]), device=self.store_device
                )
                w_cnt = int(rel_idx.numel())
                ep_w_counts.append(w_cnt)
                if w_cnt > 0:
                    ep_frame_rows.append(starts_cpu[e_i] + rel_idx)
                    ep_step_indices.append(rel_idx)
                    ep_is_demo.append(rel_idx < exec_cpu[e_i])

            if ep_frame_rows:
                all_hist_rows = torch.cat(ep_frame_rows, dim=0)
                hist_front_cat = self.front_vlm[all_hist_rows].to(self.device, non_blocking=True)
                hist_wrist_cat = self.wrist_vlm[all_hist_rows].to(self.device, non_blocking=True)
                hist_state_cat = self.state_norm[all_hist_rows].to(self.device, non_blocking=True)
                hist_step_cat = torch.cat(ep_step_indices, dim=0).to(self.device, non_blocking=True)
                hist_demo_cat = torch.cat(ep_is_demo, dim=0).to(self.device, non_blocking=True)
            else:
                hist_front_cat = torch.empty((0, 64, 960), device=self.device, dtype=self.feature_dtype)
                hist_wrist_cat = torch.empty((0, 64, 960), device=self.device, dtype=self.feature_dtype)
                hist_state_cat = torch.empty((0, 8), device=self.device, dtype=torch.float32)
                hist_step_cat = torch.empty((0,), device=self.device, dtype=torch.long)
                hist_demo_cat = torch.empty((0,), device=self.device, dtype=torch.bool)

            if cfg.use_motion_beta_gate and model.uses_gdn:
                ep_motion_seqs = []
                cur_f = 0
                for wc in ep_w_counts:
                    ep_motion_seqs.append(model.compute_visual_motion_rms(hist_front_cat[cur_f : cur_f + wc]))
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
            "lang_tok": lang_tok,
            "lang_mask": lang_mask,
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


# =============================================================================
# 3. Trainable-Only EMA & Vectorized Batch Memory State Builder
# =============================================================================


class TrainableEMAModel:
    """Fast `fp32` EMA tracker over trainable parameters ONLY (saves 1.22 GB over frozen VLM!)."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            name: p.detach().float().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module, *, exact_copy: bool = False) -> None:
        if exact_copy:
            for name, p in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].copy_(p.detach().float())
            return
        d = self.decay
        one_m = 1.0 - d
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach().float(), alpha=one_m)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].to(dtype=p.dtype, device=p.device))


def _build_compact_export_state_dict(
    model: SmolVLADeltaLoopPolicy,
    ema: Optional[TrainableEMAModel],
) -> Dict[str, torch.Tensor]:
    """Export self-contained `~850 MiB` `.pt` state_dict (`bf16` frozen VLM + `fp32` projections/GDN)."""
    fp32_prefixes = (
        "state_proj.",
        "action_in_proj.",
        "action_out_proj.",
        "action_time_mlp_in.",
        "action_time_mlp_out.",
        "gripper_head.",
        "halting_gate.",
    )
    fp32_suffixes = ("A_log", "dt_bias", "gamma_ctx", "calibrated_delta", "vis_anchor_proj", "_u1_loss_weights")
    odd_kv_keys = set()
    sa_every = model.vlm_with_expert.self_attn_every_n_layers
    for layer_idx in range(model.vlm_with_expert.num_expert_layers):
        if sa_every > 0 and (layer_idx % sa_every != 0):
            odd_kv_keys.add(f"vlm_with_expert.lm_expert.layers.{layer_idx}.self_attn.k_proj.weight")
            odd_kv_keys.add(f"vlm_with_expert.lm_expert.layers.{layer_idx}.self_attn.v_proj.weight")

    out_sd: Dict[str, torch.Tensor] = {}
    for k_s, v_s in model.state_dict().items():
        if ema is not None and (not k_s.startswith("halting_gate.")) and k_s in ema.shadow:
            src = ema.shadow[k_s].detach()
        else:
            src = v_s.detach()

        if not src.is_floating_point():
            out_sd[k_s] = src.cpu()
        elif k_s.startswith(fp32_prefixes) or k_s.endswith(fp32_suffixes) or (k_s in odd_kv_keys):
            out_sd[k_s] = src.to(dtype=torch.float32, device="cpu")
        else:
            out_sd[k_s] = src.to(dtype=torch.bfloat16, device="cpu")
    return out_sd


def _batch_apply_episode_conv(
    slot_resampler: Any,
    raw_cat: torch.Tensor,
    ep_w_counts: List[int],
) -> List[torch.Tensor]:
    B_ep = len(ep_w_counts)
    max_w = max(ep_w_counts) if ep_w_counts else 0
    if max_w == 0 or raw_cat.shape[0] == 0:
        return [raw_cat[:0] for _ in range(B_ep)]

    total_w, nw, d = raw_cat.shape
    dev = raw_cat.device
    raw_pad_row = torch.cat([raw_cat, torch.zeros((1, nw, d), device=dev, dtype=raw_cat.dtype)], dim=0)
    pad_idx = torch.full((B_ep, max_w), total_w, dtype=torch.long)
    cursor = 0
    for e_i, wc in enumerate(ep_w_counts):
        if wc > 0:
            pad_idx[e_i, :wc] = torch.arange(cursor, cursor + wc, dtype=torch.long)
            cursor += wc
    padded_raw = raw_pad_row[pad_idx.to(device=dev, non_blocking=True)]
    padded_conv = slot_resampler.apply_episode_slot_conv(padded_raw)
    return [padded_conv[e_i, :wc] if wc > 0 else raw_cat[:0] for e_i, wc in enumerate(ep_w_counts)]


def build_smolvla_batch_memory_state(
    model: SmolVLADeltaLoopPolicy,
    batch: Dict[str, Any],
) -> Optional[Dict[str, torch.Tensor]]:
    if not model.uses_history:
        return None
    assert model.slot_resampler is not None
    ep_w_counts: List[int] = batch["ep_w_counts"]
    dev = batch["curr_front"].device

    if model.arch == "smolvla_m2_framesamp":
        B = int(batch["sample_local_ep"].shape[0])
        B_sz = model.cfg.framesamp_buffer_size
        D = model.cfg.d_mem
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
        raw_uniq = model.encode_history_frames_raw(
            batch["hist_front_cat"][uniq_dev],
            batch["hist_wrist_cat"][uniq_dev],
            batch["hist_state_cat"][uniq_dev],
            batch["hist_step_cat"][uniq_dev],
            batch["hist_demo_cat"][uniq_dev],
        )
        slots_uniq = model.slot_resampler.out_norm(raw_uniq[:, 0, :].to(dt))
        n_u = int(slots_uniq.shape[0])
        slots_pad = torch.cat([slots_uniq, torch.zeros((1, D), device=dev, dtype=dt)], dim=0)

        gather_cpu = torch.full((B, B_sz), n_u, dtype=torch.long)
        gather_cpu[mask_cpu] = inv_valid_cpu
        buf_tok = slots_pad[gather_cpu.to(device=dev, non_blocking=True)]
        buf_mask = mask_cpu.to(device=dev, non_blocking=True)
        return {"buf_tokens": buf_tok, "buf_mask": buf_mask}

    raw_cat = model.encode_history_frames_raw(
        batch["hist_front_cat"],
        batch["hist_wrist_cat"],
        batch["hist_state_cat"],
        batch["hist_step_cat"],
        batch["hist_demo_cat"],
    )
    ep_slot_seqs = _batch_apply_episode_conv(model.slot_resampler, raw_cat, ep_w_counts)
    return model.compute_packed_gdn_states(
        ep_slot_seqs,
        batch["sample_local_ep"],
        batch["sample_written_counts"],
        episode_motion_seqs=batch.get("ep_motion_seqs"),
    )


# =============================================================================
# 4. Stage-I + Stage-II Training Runner per Controlled `SmolVLA` Model
# =============================================================================


def train_single_smolvla_model(
    arch: ArchSmolVLA,
    store: GPUSmolVLAStore,
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
                print(f"[{arch}] Checkpoint already completed at {ckpt_path} — skipping (--resume).", flush=True)
                return existing.get("metrics", {})
        except Exception as exc:
            print(f"[{arch}] Warning: corrupt existing checkpoint ({exc}), retraining...", flush=True)

    set_all_seeds(args.seed)
    batch_gen = torch.Generator(device="cpu")
    batch_gen.manual_seed(args.seed + 20011)

    cfg = SmolVLADeltaLoopConfig(
        arch=arch,
        d_mem=args.d_mem,
        mem_heads=args.mem_heads,
        mem_ff=args.mem_ff,
        num_write_slots=args.num_write_slots,
        num_coda_tokens=args.num_coda_tokens,
        write_stride=args.write_stride,
        gdn_heads=args.gdn_heads,
        gdn_d_k=args.gdn_d_k,
        gdn_d_v=args.gdn_d_v,
        max_loops=args.max_loops,
        anchor_hop0_coda=args.anchor_hop0_coda,
        hook_dropout=args.hook_dropout,
    )
    model = SmolVLADeltaLoopPolicy(cfg)
    load_info = model.load_pretrained_smolvla(args.pretrained_safetensors)
    # Keep trainable parameters in `fp32` (exact AdamW master precision!) and frozen VLM in `bf16`!
    model.to(device=device, dtype=torch.float32)
    if train_dtype == torch.bfloat16:
        model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)

    if model.halting_gate is not None:
        model.halting_gate.requires_grad_(False)

    param_info = model.parameter_breakdown()
    state_bytes = model.state_size_bytes()
    print(
        f"\n================================================================================\n"
        f"[{arch}] Starting Pretrained SmolVLA Stage-I Training (steps={args.steps}, tasks={len(store.tasks)}, seed={args.seed})\n"
        f"  loaded_pretrained={load_info['loaded_tensors']} | total_active={param_info['total_active']:,} "
        f"(trainable={param_info['trainable']:,}, vlm_frozen={param_info['vlm_frozen']:,}, "
        f"hooks={param_info['expert_memory_hooks']:,}, mem_loop={param_info['memory_loop_and_aux']:,})\n"
        f"  state_bytes={state_bytes:,} B ({state_bytes/1024:.1f} KB) | num_coda={cfg.num_coda_tokens} | anchor_hop0={cfg.anchor_hop0_coda} | train_dtype={train_dtype} | HAS_FLA={HAS_FLA}\n"
        f"================================================================================",
        flush=True,
    )

    ema = TrainableEMAModel(model, decay=args.ema_decay)
    param_groups = model.build_differential_param_groups(
        lr_expert=args.lr_expert,
        lr_heads=args.lr_heads,
        lr_mem=args.lr_mem,
        wd_pretrained=args.wd_pretrained,
        wd_mem=args.wd_mem,
    )
    base_lrs = [float(pg["lr"]) for pg in param_groups]
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    use_autocast = device.type == "cuda" and train_dtype in (torch.bfloat16, torch.float16)
    log_interval = max(1, args.log_every)
    save_interval = max(0, int(args.save_every))
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
            if "trainable_state" in r_ckpt:
                tr_dict = r_ckpt["trainable_state"]
                for name, p in model.named_parameters():
                    if name in tr_dict:
                        p.data.copy_(tr_dict[name].to(device=device, dtype=p.dtype))
            if "ema_shadow" in r_ckpt:
                for k_s, v_s in r_ckpt["ema_shadow"].items():
                    if k_s in ema.shadow:
                        ema.shadow[k_s].copy_(v_s.to(device=ema.shadow[k_s].device, dtype=torch.float32))
            if "optimizer_state" in r_ckpt:
                optimizer.load_state_dict(r_ckpt["optimizer_state"])
            if "batch_gen_state" in r_ckpt:
                batch_gen.set_state(r_ckpt["batch_gen_state"])
            print(f"[{arch}] Resumed Stage-I from step {start_step - 1}/{args.steps} ({resume_path})", flush=True)
        except Exception as exc:
            print(f"[{arch}] Warning: failed to load resume {resume_path} ({exc}), starting from step 1.", flush=True)
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
        if step <= args.warmup_steps:
            lr_mult = step / max(1, args.warmup_steps)
        else:
            prog = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
            min_ratio = 0.05
            lr_mult = min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog))
        for pg, b_lr in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = b_lr * lr_mult

        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)

        batch = store.sample_balanced_batch(
            eps_per_task=args.eps_per_task,
            chunks_per_ep=args.chunks_per_ep,
            model=model,
            gen=batch_gen,
        )

        with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
            f_aug, w_aug, pf_aug, eff_theta = model.apply_synchronized_front_aug(
                batch["curr_front"],
                batch["curr_wrist"],
                prev_front_vlm=batch["prev_front"],
            )
            prefill = model.prefill_vlm_kv_from_tokens(
                f_aug,
                w_aug,
                batch["lang_tok"],
                batch["lang_mask"],
                batch["curr_state"],
            )
            ctx_tok, ctx_mask = model.encode_readout_context(
                f_aug,
                w_aug,
                batch["curr_state"],
                prefill["lang_raw"],
                batch["lang_mask"],
                prev_front_vlm=pf_aug,
                prev_wrist_vlm=batch["prev_wrist"],
                prev_state_norm=batch["prev_state"],
            )
            xy_tgt = batch["gt_xy_norm"]
            xy_mask = batch["gt_has_xy"]
            if eff_theta is not None and xy_tgt is not None and xy_mask is not None:
                xy_tgt, xy_mask = transform_coords_under_affine(xy_tgt, xy_mask, eff_theta)

            mem_state = build_smolvla_batch_memory_state(model, batch)
            ro_out = model.run_coda_readout(
                ctx_tok,
                ctx_mask,
                mem_state=mem_state,
                batch_size=int(f_aug.shape[0]),
                device=device,
            )
            out = model.compute_crn_flow_and_aux_loss(
                prefill["prefix_pad_masks"],
                prefill["past_key_values"],
                ctx_tok,
                ro_out["coda_per_depth"],
                ro_out["mem_mask"],
                batch["target_actions"],
                aux_targets=batch["aux_targets"],
                eta_aux=args.eta_aux,
                grounding_xy_target=xy_tgt,
                grounding_xy_mask=xy_mask,
                subgoal_emb_target=batch["gt_sg_emb"],
                subgoal_emb_mask=batch["gt_has_sg"],
            )
            loss = out["loss"]

        pd_step = out["per_depth_fm_loss"].detach().mean(dim=0).cpu()
        loss.backward()

        write_p, other_p = model.split_write_vs_other_params()
        gn_w = float(torch.nn.utils.clip_grad_norm_(write_p, args.max_grad_norm).item()) if write_p else 0.0
        gn_o = float(torch.nn.utils.clip_grad_norm_(other_p, args.max_grad_norm).item()) if other_p else 0.0
        optimizer.step()
        ema.update(model, exact_copy=(step < args.ema_start_step))

        running_fm += float(out["fm_loss"].detach().item())
        running_aux += float(out["aux_loss"].detach().item())
        running_gnd += float(out["ground_loss"].detach().item())
        running_grip += float(out["grip_loss"].detach().item())
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

            hooks = model.vlm_with_expert.memory_hooks
            if hooks is not None and len(hooks) > 0:
                h_list_m = list(hooks.values())
                gammas = [round(float(torch.tanh(hm.gamma_ctx).item()), 4) for hm in h_list_m]
                ca_norms = [round(float(hm.mem_o_proj.weight.detach().float().norm().item()), 3) for hm in h_list_m]
                hook_str = f"ca_w={ca_norms[0]:.2f}/{ca_norms[-1]:.2f} tanh_g={gammas[0]:+.3f}/{gammas[-1]:+.3f} "
            else:
                gammas = []
                ca_norms = []
                hook_str = ""

            print(
                f"[{arch} | step {step:5d}/{args.steps}] "
                f"fm={avg_fm:.5f} aux={avg_aux:.5f} gnd={avg_gnd:.4f} grip={avg_grip:.4f} pd={pd_vals} "
                f"lr_m={args.lr_mem * lr_mult:.2e} gn_w={avg_gw:.3f} gn_o={avg_go:.3f} "
                f"{hook_str}({ms_per_step:.1f} ms/step, ETA {eta_s/60:.1f}m)",
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
                "trainable_state": {
                    name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad
                },
                "ema_shadow": {k_s: v_s.detach().cpu() for k_s, v_s in ema.shadow.items()},
                "optimizer_state": optimizer.state_dict(),
                "batch_gen_state": batch_gen.get_state(),
            }
            torch.save(r_payload, r_tmp)
            r_tmp.replace(resume_path)

    stage1_wall_s = time.perf_counter() - t_start
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
            f"(steps={args.stage2_steps}, M_CRN={args.m_crn})...",
            flush=True,
        )
        t_s2 = time.perf_counter()
        model.eval()
        model.requires_grad_(False)
        model.halting_gate.train()
        model.halting_gate.requires_grad_(True)

        s2_chunks = max(2, args.chunks_per_ep // 2)
        raw_gaps = []
        with torch.no_grad():
            for _ in range(8):
                cb = store.sample_balanced_batch(
                    eps_per_task=args.eps_per_task,
                    chunks_per_ep=s2_chunks,
                    model=model,
                    gen=batch_gen,
                )
                with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                    pf_c = model.prefill_vlm_kv_from_tokens(
                        cb["curr_front"], cb["curr_wrist"], cb["lang_tok"], cb["lang_mask"], cb["curr_state"]
                    )
                    ctx_c, cm_c = model.encode_readout_context(
                        cb["curr_front"],
                        cb["curr_wrist"],
                        cb["curr_state"],
                        pf_c["lang_raw"],
                        cb["lang_mask"],
                        prev_front_vlm=cb["prev_front"],
                        prev_wrist_vlm=cb["prev_wrist"],
                        prev_state_norm=cb["prev_state"],
                    )
                    ms_c = build_smolvla_batch_memory_state(model, cb)
                    ro_c = model.run_coda_readout(
                        ctx_c, cm_c, mem_state=ms_c, batch_size=int(cb["curr_front"].shape[0]), device=device
                    )
                    crn_acc = torch.zeros(
                        (cb["curr_front"].shape[0], cfg.max_loops), device=device, dtype=torch.float32
                    )
                    for _ in range(args.m_crn):
                        lo_m = model.compute_crn_flow_and_aux_loss(
                            pf_c["prefix_pad_masks"],
                            pf_c["past_key_values"],
                            ctx_c,
                            ro_c["coda_per_depth"],
                            None,
                            cb["target_actions"],
                            eta_aux=0.0,
                        )
                        crn_acc += lo_m["per_depth_fm_loss"]
                    crn_mean = crn_acc / float(args.m_crn)
                    # Raw marginal gain `(L_FM(0) - L_FM(L_max-1))` WITHOUT `.clamp(min=0)` (#80 item 1):
                    # 58th percentile guarantees ~42% of chunks have `w_t^(0) = 1` (`target E[L_t] ≈ 1.45-1.75`)!
                    raw_gaps.append(crn_mean[:, 0] - crn_mean[:, -1])

        all_gaps = torch.cat(raw_gaps, dim=0)
        calib_delta = float(torch.quantile(all_gaps, 0.58).clamp(min=1e-5, max=0.05).item())
        frac_w0 = float((all_gaps > calib_delta).float().mean().item())
        model.halting_gate.calibrated_delta.fill_(calib_delta)
        print(
            f"[{arch} | Stage-II] Calibrated marginal-gain delta = {calib_delta:.6f} "
            f"(min_delta=1e-5, p58, frac_w0={frac_w0:.3f})",
            flush=True,
        )

        s2_opt = torch.optim.AdamW(model.halting_gate.parameters(), lr=1e-3, weight_decay=1e-4)
        exp_loops_running = 0.0
        s2_loss_running = 0.0

        for s2_step in range(1, args.stage2_steps + 1):
            cb = store.sample_balanced_batch(
                eps_per_task=args.eps_per_task,
                chunks_per_ep=s2_chunks,
                model=model,
                gen=batch_gen,
            )
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                    pf_c = model.prefill_vlm_kv_from_tokens(
                        cb["curr_front"], cb["curr_wrist"], cb["lang_tok"], cb["lang_mask"], cb["curr_state"]
                    )
                    ctx_c, cm_c = model.encode_readout_context(
                        cb["curr_front"],
                        cb["curr_wrist"],
                        cb["curr_state"],
                        pf_c["lang_raw"],
                        cb["lang_mask"],
                        prev_front_vlm=cb["prev_front"],
                        prev_wrist_vlm=cb["prev_wrist"],
                        prev_state_norm=cb["prev_state"],
                    )
                    ms_c = build_smolvla_batch_memory_state(model, cb)
                    crn_acc = torch.zeros(
                        (cb["curr_front"].shape[0], cfg.max_loops), device=device, dtype=torch.float32
                    )

            with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                ro_c = model.run_coda_readout(
                    ctx_c, cm_c, mem_state=ms_c, batch_size=int(cb["curr_front"].shape[0]), device=device
                )

            with torch.no_grad():
                coda_det = [c.detach() if c is not None else None for c in ro_c["coda_per_depth"]]
                with torch.autocast(device_type=device.type, dtype=train_dtype, enabled=use_autocast):
                    for _ in range(args.m_crn):
                        lo_m = model.compute_crn_flow_and_aux_loss(
                            pf_c["prefix_pad_masks"],
                            pf_c["past_key_values"],
                            ctx_c,
                            coda_det,
                            None,
                            cb["target_actions"],
                            eta_aux=0.0,
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
                    f"loss={s2_loss_running/denom_s2:.5f} E[L_t]={exp_loops_running/denom_s2:.2f}",
                    flush=True,
                )
                stage2_metrics = {
                    "calibrated_delta": round(calib_delta, 6),
                    "frac_w0_positive": round(frac_w0, 4),
                    "final_stage2_loss": round(s2_loss_running / denom_s2, 6),
                    "mean_expected_loops": round(exp_loops_running / denom_s2, 3),
                    "stage2_wall_s": round(time.perf_counter() - t_s2, 2),
                }
                s2_loss_running = 0.0
                exp_loops_running = 0.0

    model.eval()
    total_wall_s = time.perf_counter() - t_start
    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated(device) / (1024**3), 3) if device.type == "cuda" else 0.0
    )

    metrics = {
        "arch": arch,
        "pretrained_backbone": "lerobot/smolvla_base",
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

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f".{arch}.pt.tmp"
    export_sd = _build_compact_export_state_dict(model, ema=ema)
    payload = {
        "completed": True,
        "arch": arch,
        "model_config": cfg.to_dict(),
        "model_state_dict": export_sd,
        "action_mode": "mean_std_delta",
        "norm_stats": store.norm_stats_dict,
        "metrics": metrics,
    }
    torch.save(payload, tmp_path)
    tmp_path.replace(ckpt_path)
    if resume_path.exists():
        resume_path.unlink(missing_ok=True)

    (out_dir / f"{arch}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"[MODEL_DONE] {arch} -> {ckpt_path} "
        f"({ckpt_path.stat().st_size / (1024**2):.1f} MiB, wall={total_wall_s/60:.2f}m)",
        flush=True,
    )
    return metrics


# =============================================================================
# 5. CLI Entrypoint
# =============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Pretrained SmolVLA-DeltaLoop (`U6`) Suite across 16 RoboMME Tasks")
    p.add_argument(
        "--cache-dir",
        type=str,
        default="/work/gj26/b20098/datasets/robomme_smolvla_v1",
        help="Path to pre-extracted SmolVLA VLM-connector shard directory",
    )
    p.add_argument(
        "--pretrained-safetensors",
        type=str,
        default=DEFAULT_MIYABI_SMOLVLA_SAFETENSORS,
        help="Path to `lerobot/smolvla_base` `model.safetensors`",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "robomme" / "cycle3b_smolvla"),
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(ALL_SMOLVLA_ARCHS),
        choices=list(ALL_SMOLVLA_ARCHS),
    )
    p.add_argument("--tasks", nargs="+", default=None, help="Defaults to all 16 ROBOMME_TASKS")
    p.add_argument("--pilot-4tasks", action="store_true", default=False)
    p.add_argument("--cycle2a-8tasks", action="store_true", default=False)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--stage2-steps", type=int, default=800)
    p.add_argument("--m-crn", type=int, default=8)
    p.add_argument("--ponder-cost", type=float, default=0.003)

    p.add_argument("--eps-per-task", type=int, default=None, help="Default: 16 // len(tasks) (= 16 eps/step)")
    p.add_argument("--chunks-per-ep", type=int, default=4, help="16 eps x 4 chunks = 64 chunks/step")

    p.add_argument("--lr-expert", type=float, default=5e-5)
    p.add_argument("--lr-heads", type=float, default=1e-4)
    p.add_argument("--lr-mem", type=float, default=2e-4)
    p.add_argument("--wd-pretrained", type=float, default=1e-4)
    p.add_argument("--wd-mem", type=float, default=1e-3)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--eta-aux", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--ema-start-step", type=int, default=300)

    p.add_argument("--d-mem", type=int, default=384)
    p.add_argument("--mem-heads", type=int, default=6)
    p.add_argument("--mem-ff", type=int, default=1024)
    p.add_argument("--num-write-slots", type=int, default=8)
    p.add_argument("--num-coda-tokens", type=int, default=4)
    p.add_argument("--write-stride", type=int, default=4)
    p.add_argument("--gdn-heads", type=int, default=8)
    p.add_argument("--gdn-d-k", type=int, default=128)
    p.add_argument("--gdn-d-v", type=int, default=128)
    p.add_argument("--max-loops", type=int, default=4)
    p.add_argument(
        "--anchor-hop0-coda",
        action="store_true",
        default=False,
        help="Cycle 4B U8: partition memory tokens into [c0 + role0, c_l + role1] (4 immutable hop-0 anchors + 4 hop-l pointers)",
    )
    p.add_argument("--hook-dropout", type=float, default=0.05)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-preload-gpu", action="store_true")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--summary-name", type=str, default="smolvla_u6_train_summary.json")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--self-test-only", action="store_true")
    p.add_argument("--verify-pretrained-only", action="store_true")

    args = p.parse_args(argv)
    if args.tasks is None or args.tasks == ["all"]:
        if args.pilot_4tasks:
            args.tasks = list(PILOT_4_TASKS)
        elif args.cycle2a_8tasks:
            args.tasks = list(CYCLE2A_8_TASKS)
        else:
            args.tasks = list(ROBOMME_TASKS)
    if args.eps_per_task is None:
        args.eps_per_task = max(1, 16 // max(1, len(args.tasks)))
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    train_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    self_test_report = run_smolvla_self_tests(device=device, dtype=train_dtype)
    if args.self_test_only:
        return

    if args.verify_pretrained_only:
        verify_pretrained_safetensors_parity(args.pretrained_safetensors, device=device, dtype=train_dtype)
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    store = GPUSmolVLAStore(
        args.cache_dir,
        args.tasks,
        device=device,
        feature_dtype=train_dtype,
        preload_gpu=(not args.no_preload_gpu),
        action_horizon=20,
    )

    all_metrics: Dict[str, Any] = {}
    for arch in args.models:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        m = train_single_smolvla_model(arch, store, args, device, train_dtype, out_dir)
        all_metrics[arch] = m

    summary = {
        "self_tests": self_test_report,
        "tasks": list(store.tasks),
        "models": all_metrics,
    }
    summary_path = out_dir / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[ALL_DONE] Saved SmolVLA-DeltaLoop summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
