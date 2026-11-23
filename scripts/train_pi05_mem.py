#!/usr/bin/env python3
"""Phase-2 `DeltaMem-π0.5` & `T²Mem` Alternating Co-Adaptation Trainer (`scripts/train_pi05_mem.py`).

Supports all Phase-2 architectures (`p2_b0_pi05_base`, `p2_b1_t2mem`, `p2_b2_framesamp`,
`p2_m1_gdn2l_ms`, `p2_m2_deltamem`, `p2_m3_deltamem_loop`) and multi-round alternating stages:
  - `stage_a` (`A0`, `A1`): Freeze `Yinpei/pi05_baseline` (`PaliGemma-2B` + `ActionExpert-300M`);
    train `DeltaMem` + 18-layer `MemoryAttention`/`MemoryRMSNorm` (`lr_mem=2e-4`, `lr_gate=5e-4`
    with `wd=0`, `p_distractor=0.25` realistic `k=1..3` boundary-reset training).
  - `stage_b` (`B1`, `B2`): Freeze (or `0.1x`-LR) `DeltaMem` slow weights; co-adapt `ActionExpert-300M`
    + `PaliGemma-2B` LoRA (`r=32`, `lr_policy=3e-5`, `EMA=0.999`, `p_mem_drop=0.10` for `Mem-CFG`,
    sub-grid `RandomCrop` + `±8px` pointer jitter).
  - `stage_c` (`C`): Freeze Stage-B `L=1` weights; train zero-init `ResidualMultiHopRefiner`
    (`gamma_0 = 0`) + `M_CRN = 8` relative-quantile halting gate (`provably >= L=1` at init).

Miyabi Interactive-Queue First Design (`interact-g` / `interact-mig` `<= 02:00:00` ONLY):
  - Atomic crash-safe checkpoints every `--save-every 2500` steps (`latest.pt` + `step_XXXXX.pt`),
  - Automatic graceful walltime exit (`--max-wall-minutes 112`) before 2h PBS limit so chained
    `interact-g` runs (`--resume auto`) lose **zero** training steps!
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.deltaloop_vla import HAS_FLA  # noqa: E402
from memory.gated_deltanet import recurrent_gated_delta_rule  # noqa: E402
from pi05 import Pi05Model  # noqa: E402
from pi05.perceptual_mem import PosEmb3D  # noqa: E402
from pi05_mem import (  # noqa: E402
    ALL_PI05_MEM_ARCHS,
    DeltaGroundConfig,
    DeltaGroundEpisodeDataset,
    DeltaGroundModel,
    DeltaMemPi05Policy,
    GraftedMemoryRMSNorm,
    MultiTimescaleSlotGDNLayer,
    Pi05MemConfig,
    RealisticCrossSessionEpisodeDataset,
    Stratified16TaskBatchSampler,
    SyntheticPi05EpisodeDataset,
    TorchPosEmb3D,
    UnifiedGroundedHybridEvalAdapter,
    build_deltaground_adapter,
    build_eval_adapter,
    build_grounded_hybrid_adapter,
    build_occlusion_twin_template_pairs,
    build_pi05_config_from_mem_cfg,
    collate_episode_list,
    decompose_grounded_subgoal,
    fold_occlusion_twin_logits,
    pool_spatial_tokens_to_square,
    precompute_deltaground_subgoals,
    render_grounded_subgoal_from_norm_xy,
    select_episode_write_indices,
)
from analysis.subgoal_grounding import pixel_xy_to_norm_xy  # noqa: E402
from robomme_data.constants import ROBOMME_SUITES, ROBOMME_TASKS  # noqa: E402

DEFAULT_MIYABI_PI05_CKPT = "/work/gj26/b20098/checkpoints/pi05_init/pi05_baseline_pt"
DEFAULT_MIYABI_PI05_CACHE = "/work/gj26/b20098/datasets/robomme_pi05_cache"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_task_filter(spec: str) -> Optional[Sequence[str]]:
    s = spec.strip()
    if s.lower() in ("all", "16", "*"):
        return None
    if s in ROBOMME_SUITES:
        return ROBOMME_SUITES[s]
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    for p in parts:
        if p not in ROBOMME_TASKS:
            raise ValueError(f"Unknown task {p!r}; valid tasks: {ROBOMME_TASKS}")
    return tuple(parts)


# =============================================================================
# 1. Trainable-Only `fp32` EMA & Atomic Resumable Checkpointing
# =============================================================================


class TrainableFP32EMA:
    """Maintains `fp32` exponential moving average strictly over trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_ema: bool = False) -> None:
        self.decay = float(decay)
        self.warmup_ema = bool(warmup_ema)
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self.refresh_tracked_params(model)

    def refresh_tracked_params(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.detach().float().clone()

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
        master_override: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self.num_updates += 1
        d = (
            min(self.decay, (1.0 + float(self.num_updates)) / (10.0 + float(self.num_updates)))
            if self.warmup_ema
            else self.decay
        )
        one_minus_d = 1.0 - d
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            src = (
                master_override[name].detach().float()
                if (master_override is not None and name in master_override)
                else param.detach().float()
            )
            if name not in self.shadow:
                self.shadow[name] = src.clone()
            else:
                self.shadow[name].mul_(d).add_(src, alpha=one_minus_d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor], device: torch.device) -> None:
        self.shadow = {k: v.to(device=device, dtype=torch.float32).clone() for k, v in state.items()}

    @torch.no_grad()
    def export_ema_delta(self, policy: DeltaMemPi05Policy) -> Dict[str, Any]:
        """Export a delta checkpoint with EMA shadow weights substituted in-place temporarily."""
        backups: Dict[str, torch.Tensor] = {}
        param_map = dict(policy.named_parameters())
        try:
            for name, ema_val in self.shadow.items():
                if name in param_map:
                    p = param_map[name]
                    backups[name] = p.data.clone()
                    p.data.copy_(ema_val.to(device=p.device, dtype=p.dtype))
            return policy.export_delta_state_dict()
        finally:
            for name, orig in backups.items():
                param_map[name].data.copy_(orig)


class FP32MasterAdamW:
    """`fp32` Master-Weight `AdamW` wrapper preventing `bf16` sub-ULP rounding (`#241`).

    Matches `openpi` / `mme_vla_suite` mixed-precision semantics: forward/backward run in `bf16`
    while ALL trainable parameters (`ActionExpert-300M`, `18L` hooks, `LoRA`, `DeltaMem`) maintain
    exact `float32` master weights + `float32` AdamW moments across every step and `--resume`.
    """

    def __init__(
        self,
        model: nn.Module,
        param_groups: List[Dict[str, Any]],
        *,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        id_to_name = {id(p): n for n, p in model.named_parameters()}
        self._triples: List[tuple[str, nn.Parameter, nn.Parameter]] = []
        self._name_to_master: Dict[str, nn.Parameter] = {}
        master_groups: List[Dict[str, Any]] = []

        for g in param_groups:
            m_list: List[nn.Parameter] = []
            for p in g["params"]:
                name = id_to_name[id(p)]
                mp = nn.Parameter(p.detach().float().clone())
                self._triples.append((name, p, mp))
                self._name_to_master[name] = mp
                m_list.append(mp)
            mg = {k: v for k, v in g.items() if k != "params"}
            mg["params"] = m_list
            master_groups.append(mg)

        self.opt = torch.optim.AdamW(master_groups, betas=betas, eps=eps)

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        return self.opt.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for _, p, mp in self._triples:
            if set_to_none:
                p.grad = None
                mp.grad = None
            else:
                if p.grad is not None:
                    p.grad.zero_()
                if mp.grad is not None:
                    mp.grad.zero_()

    @torch.no_grad()
    def clip_and_step(self, grad_clip: float = 1.0) -> tuple[float, float]:
        for _, p, mp in self._triples:
            if p.grad is not None:
                mp.grad = p.grad.detach().float()
            else:
                mp.grad = None

        mem_gate_mps = [
            mp
            for g in self.opt.param_groups
            if g.get("name") in ("gates", "memory")
            for mp in g["params"]
            if mp.grad is not None
        ]
        pol_mps = [
            mp
            for g in self.opt.param_groups
            if g.get("name") not in ("gates", "memory")
            for mp in g["params"]
            if mp.grad is not None
        ]
        gn_mem = (
            float(torch.nn.utils.clip_grad_norm_(mem_gate_mps, grad_clip).item())
            if mem_gate_mps
            else 0.0
        )
        gn_pol = (
            float(torch.nn.utils.clip_grad_norm_(pol_mps, grad_clip).item())
            if pol_mps
            else 0.0
        )

        self.opt.step()
        for _, p, mp in self._triples:
            p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
        return gn_mem, gn_pol

    def master_named_tensors(self) -> Dict[str, torch.Tensor]:
        return {name: mp.data for name, _, mp in self._triples}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "opt": self.opt.state_dict(),
            "master_weights": {name: mp.detach().cpu() for name, _, mp in self._triples},
        }

    @torch.no_grad()
    def load_state_dict(self, sd: Dict[str, Any], device: torch.device) -> None:
        if "opt" in sd and "master_weights" in sd:
            self.opt.load_state_dict(sd["opt"])
            mw = sd["master_weights"]
            for name, p, mp in self._triples:
                if name in mw:
                    mp.data.copy_(mw[name].to(device=device, dtype=torch.float32))
                    p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
        else:
            # Backward-compatible load from legacy raw `torch.optim.AdamW` state_dict
            self.opt.load_state_dict(sd)


def atomic_torch_save(payload: Dict[str, Any], target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target_path.parent), prefix=".tmp_ckpt_", suffix=".pt")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def cosine_lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    if total_steps <= warmup_steps:
        return 1.0
    prog = min(1.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * prog))


# =============================================================================
# 2. Comprehensive CPU / GPU Mathematical & Systems Self-Test Suite (`10` Checks)
# =============================================================================


def run_pi05_mem_self_tests(device_str: str = "cpu") -> Dict[str, Any]:
    """Run all 10 mathematical & integration parity tests (`CUDA_VISIBLE_DEVICES=""` safe!)."""
    t0 = time.perf_counter()
    dev = torch.device(device_str)
    set_all_seeds(2026)
    print(f"[Self-Test] Running Phase-2 DeltaMem-pi0.5 verification suite on device={dev} (HAS_FLA={HAS_FLA})...")

    base_cfg = Pi05MemConfig.tiny_test_preset("p2_b0_pi05_base")
    ref_pi05 = Pi05Model(build_pi05_config_from_mem_cfg(base_cfg)).to(dev).eval()
    ref_sd = copy.deepcopy(ref_pi05.state_dict())

    synth_ds = SyntheticPi05EpisodeDataset(Pi05MemConfig.tiny_test_preset("p2_m2_deltamem"), num_episodes=4, seed=77)
    ep0 = synth_ds[0]  # Video-conditioned + distractor prefix episode
    f_cur = ep0["exec_front_256"][:2].to(dev)
    w_cur = ep0["exec_wrist_256"][:2].to(dev)
    lt = ep0["lang_tokens"].unsqueeze(0).expand(2, -1).to(dev)
    lm = ep0["lang_masks"].unsqueeze(0).expand(2, -1).to(dev)
    x_t = torch.randn((2, base_cfg.action_horizon, base_cfg.padded_action_dim), device=dev)
    t_vec = torch.tensor([0.25, 0.80], device=dev, dtype=torch.float32)

    with torch.no_grad():
        kv_ref = ref_pi05.prefill_prefix_kv(lt, lm, image_embs=[f_cur, w_cur])
        v_ref = ref_pi05.denoise_step_from_kv(kv_ref, x_t, t_vec)  # type: ignore[arg-type]

    # ---- Check 1: Step-0 Bit-Exact `0.0` Diff Across ALL 6 Track-1 (`modul_zero_init`) Architectures ----
    max_step0_diff = 0.0
    for arch in ALL_PI05_MEM_ARCHS:
        cfg_a = Pi05MemConfig.tiny_test_preset(arch)
        if cfg_a.modulator_mode == "mme_vla_exact":
            continue
        p_copy = Pi05Model(build_pi05_config_from_mem_cfg(base_cfg)).to(dev)
        p_copy.load_state_dict(ref_sd, strict=True)
        pol = DeltaMemPi05Policy(cfg_a, pi05_model=p_copy, device=dev, precision="float32").eval()
        with torch.no_grad():
            kv_a, _ = pol.pi05.prefill_prefix_kv(lt, lm, image_embs=[f_cur, w_cur], return_prefix_hidden=True)  # type: ignore[misc]
            t_emb = pol.pi05.paligemma_with_expert.embed_language(lt)
            p_pool = (t_emb * lm.float().unsqueeze(-1)).sum(dim=1) / lm.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            mem_out = pol.mem_core.forward_episode_chunks(
                ep_front_emb=ep0["front_emb"].to(dev),
                ep_wrist_emb=ep0["wrist_emb"].to(dev),
                ep_state_norm=ep0["state_norm"].to(dev),
                ep_is_demo=ep0["is_demo"].to(dev),
                exec_start_idx=int(ep0["exec_start_idx"]),
                chunk_step_indices=ep0["chunk_step_indices"][:2].to(dev),
                readout_cutoff_excl=ep0["readout_cutoff_excl"][:2].to(dev),
                curr_front_emb=f_cur,
                curr_wrist_emb=w_cur,
                curr_state_norm=ep0["state_norm"][ep0["chunk_step_indices"][:2]].to(dev),
                prompt_pool=p_pool,
                ep_prompt_cos_jump=ep0["prompt_cos_jump"].to(dev),
                perceptual_mem_encoder=pol.pi05.mem_encoder,
            )
            mt = mem_out["mem_tokens_l2"] if mem_out["mem_tokens_l2"] is not None else mem_out["mem_tokens_l1"]
            v_a = pol.pi05.denoise_step_from_kv(kv_a, x_t, t_vec, mem_tokens=mt, mem_mask=mem_out["mem_mask"])
            diff = float((v_a - v_ref).abs().max().item())
            max_step0_diff = max(max_step0_diff, diff)
            assert diff == 0.0, f"Step-0 zero-init violated on {arch}: diff={diff}"

    # ---- Check 2: Exact Shifted-Query Parallel Write-Surprise vs Step-by-Step Recurrence ----
    bsz_s, w_s, nw_s, h_s, dk_s, dv_s = 2, 11, 4, 6, 32, 32
    t_sl = w_s * nw_s
    k_rand = torch.nn.functional.normalize(torch.randn((bsz_s, t_sl, h_s, dk_s), device=dev), dim=-1)
    v_rand = torch.randn((bsz_s, t_sl, h_s, dv_s), device=dev)
    g_rand = -torch.rand((bsz_s, t_sl, h_s), device=dev) * 0.2
    b_rand = torch.rand((bsz_s, t_sl, h_s), device=dev) * 1.4
    s0_rand = torch.randn((bsz_s, h_s, dv_s, dk_s), device=dev) * 0.3

    surp_fast = MultiTimescaleSlotGDNLayer.compute_exact_write_surprise(
        k_rand, v_rand, g_rand, b_rand, num_write_slots=nw_s, initial_state_fp32=s0_rand
    )
    # Explicit sequential online reference
    s_seq = s0_rand.clone()
    seq_errs = []
    for u in range(t_sl):
        ku, vu, gu, bu = k_rand[:, u], v_rand[:, u], g_rand[:, u], b_rand[:, u]
        sk = torch.einsum("bhvk,bhk->bhv", s_seq, ku)
        de = vu - torch.exp(gu).unsqueeze(-1) * sk
        rel = torch.linalg.vector_norm(de, ord=2, dim=-1) / (torch.linalg.vector_norm(vu, ord=2, dim=-1) + 0.1)
        seq_errs.append(rel.mean(dim=-1))
        s_seq = torch.exp(gu)[:, :, None, None] * s_seq + torch.einsum(
            "bhv,bhk->bhvk", bu.unsqueeze(-1) * de, ku
        )
    surp_ref = torch.stack(seq_errs, dim=1).view(bsz_s, w_s, nw_s).mean(dim=-1)
    surp_diff = float((surp_fast - surp_ref).abs().max().item())
    assert surp_diff < 1e-5, f"Shifted-query surprise mismatch: {surp_diff}"

    # ---- Check 3: Invariant A1 (`g[:, 1:] == 0`) & 3-Timescale Half-Life Band Separation ----
    cfg_m2 = Pi05MemConfig.tiny_test_preset("p2_m2_deltamem")
    pol_m2 = DeltaMemPi05Policy(cfg_m2, device=dev, precision="float32")
    assert pol_m2.mem_core.gdn is not None
    z_test = torch.randn((2, 7, cfg_m2.num_write_slots, cfg_m2.d_mem), device=dev)
    _, _, g_flat, _, _ = pol_m2.mem_core.gdn.layer1.project_slots(z_test)
    g_unflat = g_flat.view(2, 7, cfg_m2.num_write_slots, cfg_m2.gdn_heads)
    a1_slot_leak = float(g_unflat[:, :, 1:, :].abs().max().item())
    assert a1_slot_leak == 0.0, f"Invariant A1 violated: g[:, :, 1:] max={a1_slot_leak}"
    g_s0 = g_unflat[0, 0, 0]  # [Short(0..1), Mid(2..3), Long(4..5)]
    assert float(g_s0[:2].mean().item()) < float(g_s0[2:4].mean().item()) < float(g_s0[4:].mean().item()), (
        f"3-timescale decay ordering violated: {g_s0.tolist()}"
    )

    # ---- Check 4: Boundary Gate Protects `demo -> exec` (`r == 0`) & Flushes `k=3` (`r > 0.99`) ----
    assert pol_m2.mem_core.writer is not None and pol_m2.mem_core.gdn.boundary_gate is not None
    f_seq = torch.randn((6, 64, cfg_m2.vis_dim), device=dev)
    s_seq_st = torch.zeros((6, 8), device=dev)
    # Frames 0,1 = demo; Frame 2 = exec start (even with large visual/pose shift, MUST NOT flush!);
    # Frame 4 = un-reset new episode with prompt jump -> MUST flush (`r == 1.0`)!
    s_seq_st[2, :7] = 2.5
    d_seq = torch.tensor([True, True, False, False, False, False], device=dev)
    pj_seq = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.9, 0.0], device=dev)
    kin_b = pol_m2.mem_core.writer.compute_frame_kinematics(f_seq, s_seq_st, d_seq)
    r_b = pol_m2.mem_core.gdn.boundary_gate(torch.randn((6, cfg_m2.d_mem), device=dev), kin_b, pj_seq)
    assert float(r_b[2].item()) == 0.0, f"CRITICAL BUG: demo->exec triggered boundary reset r={r_b[2].item()}!"
    assert float(r_b[4].item()) >= 0.99, f"Cross-session prompt jump failed to trigger reset: r={r_b[4].item()}"

    # ---- Check 5: Multi-Stage `Stage A -> Stage B (LoRA + Mem-CFG) -> Stage C (Hop-2 + CRN)` ----
    # Simulate pretrained non-zero AdaRMSNorm FFN gate (`g1_ffn = 1.0`, zero in un-loaded fresh init)
    for blk in pol_m2.pi05.paligemma_with_expert.layers:
        assert blk.expert.post_attention_layernorm.dense is not None
        blk.expert.post_attention_layernorm.dense.bias.data[2 * cfg_m2.expert_width :].fill_(1.0)
    # Stage A
    pol_m2.configure_stage("stage_a")
    frozen_vlm_before = pol_m2.pi05.paligemma_with_expert.layers[0].vlm.q_proj.weight.detach().clone()
    opt_a = torch.optim.AdamW(pol_m2.build_optimizer_param_groups(lr_mem=1e-3, lr_gate=3e-3, lr_policy=1e-4))
    ema = TrainableFP32EMA(pol_m2, decay=0.95)
    pol_m2.train()
    out_a1 = pol_m2.forward_training_episode_batch([synth_ds[0], synth_ds[1]], max_chunks_per_ep=3)
    assert torch.isfinite(out_a1["loss"]), f"Stage-A non-finite loss: {out_a1}"
    out_a1["loss"].backward()
    # Verify non-zero grad on both `mem_core` and `pi05` memory hooks!
    hook_grad_norm = sum(float(p.grad.norm().item()) for p in pol_m2.pi05.memory_hook_parameters() if p.grad is not None)
    mem_grad_norm = sum(float(p.grad.norm().item()) for p in pol_m2.mem_core.parameters() if p.grad is not None)
    assert hook_grad_norm > 0.0 and mem_grad_norm > 0.0, (
        f"Zero gradient in Stage A: hook_grad={hook_grad_norm}, mem_grad={mem_grad_norm}"
    )
    opt_a.step()
    opt_a.zero_grad(set_to_none=True)
    ema.update(pol_m2)
    vlm_drift_a = float(
        (pol_m2.pi05.paligemma_with_expert.layers[0].vlm.q_proj.weight - frozen_vlm_before).abs().max().item()
    )
    assert vlm_drift_a == 0.0, f"Frozen VLM weight drifted during Stage A: {vlm_drift_a}"

    # Verify Empty-Mask `Mem-CFG` safety AFTER non-zero Stage-A update (`all-False mask == None`)
    pol_m2.eval()
    with torch.no_grad():
        kv_chk = pol_m2.pi05.prefill_prefix_kv(lt, lm, image_embs=[f_cur, w_cur])
        fake_mem = torch.randn((2, 20, cfg_m2.mem_token_dim), device=dev)
        empty_mask = torch.zeros((2, 20), device=dev, dtype=torch.bool)
        v_none = pol_m2.pi05.denoise_step_from_kv(kv_chk, x_t, t_vec, mem_tokens=None, mem_mask=None)  # type: ignore[arg-type]
        v_empty = pol_m2.pi05.denoise_step_from_kv(kv_chk, x_t, t_vec, mem_tokens=fake_mem, mem_mask=empty_mask)  # type: ignore[arg-type]
        empty_mask_diff = float((v_none - v_empty).abs().max().item())
        assert empty_mask_diff == 0.0, f"All-False mem_mask leaked non-zero modulation: {empty_mask_diff}"

    # Stage B (inject VLM LoRA -> verify Step-0 `0.0` diff vs post-Stage-A before stepping!)
    pol_m2.cfg.use_vlm_lora = True
    with torch.no_grad():
        v_pre_lora = pol_m2.pi05.denoise_step_from_kv(
            pol_m2.pi05.prefill_prefix_kv(lt, lm, image_embs=[f_cur, w_cur]),  # type: ignore[arg-type]
            x_t,
            t_vec,
        )
    pol_m2.configure_stage("stage_b", freeze_mem_in_stage_b=True, enable_grad_ckpt=True)
    pol_m2.eval()
    with torch.no_grad():
        v_post_lora = pol_m2.pi05.denoise_step_from_kv(
            pol_m2.pi05.prefill_prefix_kv(lt, lm, image_embs=[f_cur, w_cur]),  # type: ignore[arg-type]
            x_t,
            t_vec,
        )
    lora_init_diff = float((v_pre_lora - v_post_lora).abs().max().item())
    assert lora_init_diff == 0.0, f"VLM LoRA injection broke step-0 parity: {lora_init_diff}"

    pol_m2.train()
    ema.refresh_tracked_params(pol_m2)
    opt_b = torch.optim.AdamW(pol_m2.build_optimizer_param_groups(lr_mem=1e-4, lr_gate=2e-4, lr_policy=5e-4))
    out_b = pol_m2.forward_training_episode_batch([synth_ds[0], synth_ds[2]], max_chunks_per_ep=2)
    assert torch.isfinite(out_b["loss"])
    out_b["loss"].backward()
    opt_b.step()
    opt_b.zero_grad(set_to_none=True)
    ema.update(pol_m2)

    # Stage C (Zero-Init Residual Hop-2 -> verify `gamma_hop2 == 0 => L=2 == L=1` bit-exact!)
    pol_m2.configure_stage("stage_c")
    pol_m2.eval()
    with torch.no_grad():
        t_emb = pol_m2.pi05.paligemma_with_expert.embed_language(lt)
        p_pool = (t_emb * lm.float().unsqueeze(-1)).sum(dim=1) / lm.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        mc_out = pol_m2.mem_core.forward_episode_chunks(
            ep_front_emb=ep0["front_emb"].to(dev),
            ep_wrist_emb=ep0["wrist_emb"].to(dev),
            ep_state_norm=ep0["state_norm"].to(dev),
            ep_is_demo=ep0["is_demo"].to(dev),
            exec_start_idx=int(ep0["exec_start_idx"]),
            chunk_step_indices=ep0["chunk_step_indices"][:2].to(dev),
            readout_cutoff_excl=ep0["readout_cutoff_excl"][:2].to(dev),
            curr_front_emb=f_cur,
            curr_wrist_emb=w_cur,
            curr_state_norm=ep0["state_norm"][ep0["chunk_step_indices"][:2]].to(dev),
            prompt_pool=p_pool,
        )
        hop2_init_diff = float((mc_out["mem_tokens_l2"] - mc_out["mem_tokens_l1"]).abs().max().item())
        assert hop2_init_diff == 0.0, f"Stage-C Hop-2 violated zero-init guarantee: {hop2_init_diff}"

    pol_m2.train()
    ema.refresh_tracked_params(pol_m2)
    opt_c = torch.optim.AdamW(pol_m2.build_optimizer_param_groups(lr_mem=5e-4, lr_gate=1e-3, lr_policy=1e-4))
    out_c = pol_m2.forward_training_episode_batch([synth_ds[0]], max_chunks_per_ep=2)
    assert torch.isfinite(out_c["loss"]) and "gate_loss" in out_c
    out_c["loss"].backward()
    opt_c.step()
    opt_c.zero_grad(set_to_none=True)
    ema.update(pol_m2)

    # Also verify `T²Mem` (`p2_b1_t2mem`) full Stage-A forward + backward
    pol_t2 = DeltaMemPi05Policy(Pi05MemConfig.tiny_test_preset("p2_b1_t2mem"), device=dev, precision="float32")
    pol_t2.configure_stage("stage_a")
    pol_t2.train()
    out_t2 = pol_t2.forward_training_episode_batch([synth_ds[0], synth_ds[1]], max_chunks_per_ep=2)
    assert torch.isfinite(out_t2["loss"])
    out_t2["loss"].backward()

    # ---- Check 6: Atomic Checkpoint Save + `build_eval_adapter` Online Streaming + `Mem-CFG` ----
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_p = Path(tmpdir) / "latest.pt"
        atomic_torch_save(
            {
                "step": 3,
                "model_delta": pol_m2.export_delta_state_dict(),
                "ema_delta": ema.export_ema_delta(pol_m2),
            },
            ckpt_p,
        )
        fresh_pi05 = Pi05Model(build_pi05_config_from_mem_cfg(base_cfg)).to(dev)
        fresh_pi05.load_state_dict(ref_sd, strict=True)
        adapter = build_eval_adapter(ckpt_p, fresh_pi05, device=dev, hop_mode="l2")

        # Stream demo -> execution -> un-reset cross-session task switch (`k=2`) + `Mem-CFG = 1.25`!
        f_demo = torch.randn((10, cfg_m2.front_tokens_curr, cfg_m2.vis_dim), device=dev)
        w_demo = torch.randn((10, cfg_m2.wrist_tokens_curr, cfg_m2.vis_dim), device=dev)
        st_np = np.zeros((10, 8), dtype=np.float32)
        adapter.ingest_frames(f_demo, w_demo, st_np, st_np, task_goal="pick red cube", is_demo=True)
        adapter.ingest_frames(f_demo[:1], w_demo[:1], st_np[:1], st_np[:1], task_goal="pick red cube", is_demo=False)
        mt_eval, mm_eval, info_eval = adapter.read_memory(
            prefix_hidden=None,
            front_tokens_256=f_demo[:1],
            wrist_tokens_256=w_demo[:1],
            raw_state=st_np[0],
            norm_state=st_np[0],
            task_goal="pick red cube",
            step_idx=0,
            exec_start_idx=10,
        )
        assert mt_eval is not None and mm_eval is not None
        assert info_eval["loop_depth"] == 2 and "pred_subgoal" in info_eval

        # Test un-reset cross-session switch without calling `adapter.reset()`!
        adapter.ingest_frames(
            f_demo[:2], w_demo[:2], st_np[:2], st_np[:2], task_goal="press blue stop button", is_demo=False
        )
        mt_k2, mm_k2, info_k2 = adapter.read_memory(
            prefix_hidden=None,
            front_tokens_256=f_demo[:1],
            wrist_tokens_256=w_demo[:1],
            raw_state=st_np[0],
            norm_state=st_np[0],
            task_goal="press blue stop button",
            step_idx=1,
            exec_start_idx=0,
        )
        assert info_k2["boundary_reset_max"] >= 0.99, f"Online k=2 boundary reset missed: {info_k2}"

        act_cfg = fresh_pi05.sample_actions(
            lt[:1],
            lm[:1],
            image_embs=[f_cur[:1], w_cur[:1]],
            num_steps=3,
            mem_tokens=mt_k2,
            mem_mask=mm_k2,
            mem_cfg_scale=1.25,
        )
        assert act_cfg.shape == (1, cfg_m2.action_horizon, cfg_m2.padded_action_dim)
        assert torch.isfinite(act_cfg).all()

        # ---- Check 11: Deferred On-Device `vision_tower.head` (`siglip_width -> vis_dim`) Parity ----
        sig_w = pol_m2.pi05.vision_tower.head.in_features
        x_unproj = torch.randn((3, 16, sig_w), device=dev, dtype=torch.bfloat16)
        proj_fast = pol_m2._project_vis_if_needed(x_unproj)
        proj_ref = pol_m2.pi05.vision_tower.head(x_unproj.to(pol_m2.pi05.vision_tower.head.weight.dtype))
        on_dev_proj_diff = float((proj_fast.float() - proj_ref.float()).abs().max().item())
        assert on_dev_proj_diff == 0.0 and proj_fast.shape == (3, 16, cfg_m2.vis_dim)

    # ---- Check 12: End-to-End `bfloat16` Forward + Backward Across All 6 Archs & Stages A/B/C ----
    ctx_bf16 = torch.backends.mkldnn.flags(enabled=False) if dev.type == "cpu" else contextlib.nullcontext()
    with ctx_bf16:
        for arch_bf in ALL_PI05_MEM_ARCHS:
            cfg_bf = Pi05MemConfig.tiny_test_preset(arch_bf, use_vlm_lora=True)
            pol_bf = DeltaMemPi05Policy(cfg_bf, device=dev, precision="bfloat16")
            pol_bf.configure_stage("stage_a")
            pol_bf.train()
            out_bfa = pol_bf.forward_training_episode_batch([synth_ds[0], synth_ds[1]], max_chunks_per_ep=2)
            assert torch.isfinite(out_bfa["loss"]), f"bf16 stage_a non-finite on {arch_bf}"
            if out_bfa["loss"].requires_grad:
                out_bfa["loss"].backward()
            if arch_bf == "p2_m2_deltamem":
                pol_bf.zero_grad(set_to_none=True)
                pol_bf.configure_stage("stage_b")
                out_bfb = pol_bf.forward_training_episode_batch([synth_ds[0], synth_ds[1]], max_chunks_per_ep=2)
                assert torch.isfinite(out_bfb["loss"])
                out_bfb["loss"].backward()
                pol_bf.zero_grad(set_to_none=True)
                pol_bf.configure_stage("stage_c")
                out_bfc = pol_bf.forward_training_episode_batch([synth_ds[0], synth_ds[1]], max_chunks_per_ep=2)
                assert torch.isfinite(out_bfc["loss"])
                out_bfc["loss"].backward()
                with tempfile.TemporaryDirectory() as td_bf:
                    ck_bf = Path(td_bf) / "bf16_ckpt.pt"
                    torch.save({"ema_delta": pol_bf.export_delta_state_dict()}, ck_bf)
                    pi05_bf_cfg = build_pi05_config_from_mem_cfg(cfg_bf)
                    pi05_bf_cfg.precision = "bfloat16"
                    pi05_bf = Pi05Model(pi05_bf_cfg).to(dev)
                    ad_bf = build_eval_adapter(ck_bf, pi05_bf, dev)
                    ad_bf.ingest_frames(
                        f_cur.to(torch.bfloat16),
                        w_cur.to(torch.bfloat16),
                        np.zeros((2, 8), dtype=np.float32),
                        np.zeros((2, 8), dtype=np.float32),
                        task_goal="test bf16 eval",
                        is_demo=True,
                    )
                    ad_bf.ingest_frames(
                        f_cur[:1].to(torch.bfloat16),
                        w_cur[:1].to(torch.bfloat16),
                        np.zeros((1, 8), dtype=np.float32),
                        np.zeros((1, 8), dtype=np.float32),
                        task_goal="test bf16 eval",
                        is_demo=False,
                    )
                    mt_bf, mm_bf, _ = ad_bf.read_memory(
                        prefix_hidden=None,
                        front_tokens_256=f_cur[:1].to(torch.bfloat16),
                        wrist_tokens_256=w_cur[:1].to(torch.bfloat16),
                        raw_state=np.zeros((8,), dtype=np.float32),
                        norm_state=np.zeros((8,), dtype=np.float32),
                        task_goal="test bf16 eval",
                        step_idx=2,
                        exec_start_idx=2,
                    )
                    assert mt_bf is not None and mm_bf is not None
                    assert mt_bf.shape == (1, cfg_bf.max_mem_tokens, cfg_bf.mem_token_dim)
                    assert mm_bf.shape == (1, cfg_bf.max_mem_tokens)

    # ---- Check 13: Single-Pass Interleaved Zero-Write Readout vs Segment-by-Segment Parity (`< 1e-5`) ----
    pol_m2.eval()
    assert pol_m2.mem_core.hop2_refiner is not None
    with torch.no_grad():
        pol_m2.mem_core.hop2_refiner.gamma_hop2.fill_(0.5)
        interleaved_max_diff = 0.0
        for ep_chk in (synth_ds[0], synth_ds[1]):
            ms = int(ep_chk["chunk_step_indices"].numel())
            kw_common = dict(
                ep_front_emb=ep_chk["front_emb"].to(dev),
                ep_wrist_emb=ep_chk["wrist_emb"].to(dev),
                ep_state_norm=ep_chk["state_norm"].to(dev),
                ep_is_demo=ep_chk["is_demo"].to(dev),
                exec_start_idx=int(ep_chk["exec_start_idx"]),
                chunk_step_indices=ep_chk["chunk_step_indices"].to(dev),
                readout_cutoff_excl=ep_chk["readout_cutoff_excl"].to(dev),
                curr_front_emb=ep_chk["exec_front_256"][:ms].to(dev),
                curr_wrist_emb=ep_chk["exec_wrist_256"][:ms].to(dev),
                curr_state_norm=ep_chk["state_norm"][ep_chk["chunk_step_indices"]].to(dev),
                ep_prompt_cos_jump=ep_chk["prompt_cos_jump"].to(dev),
            )
            o_int = pol_m2.mem_core.forward_episode_chunks(**kw_common, use_interleaved_readout=True)
            o_seg = pol_m2.mem_core.forward_episode_chunks(**kw_common, use_interleaved_readout=False)
            d_l1 = float((o_int["mem_tokens_l1"] - o_seg["mem_tokens_l1"]).abs().max().item())
            d_l2 = float((o_int["mem_tokens_l2"] - o_seg["mem_tokens_l2"]).abs().max().item())
            d_mask = int((o_int["mem_mask"] != o_seg["mem_mask"]).sum().item())
            interleaved_max_diff = max(interleaved_max_diff, d_l1, d_l2)
            assert d_mask == 0 and max(d_l1, d_l2) < 1e-5, (
                f"Interleaved readout mismatch: d_l1={d_l1}, d_l2={d_l2}, d_mask={d_mask}"
            )

    # ---- Check 14: `FP32MasterAdamW` Sub-ULP Precision on `bfloat16` Parameters (`#241`) ----
    # For w = 0.1 in bf16, half-ULP is ~3.9e-4 > lr = 3e-5: naive bf16 AdamW rounds 100% of updates to 0.0!
    toy_mod = nn.Linear(4, 4, bias=False).to(device=dev, dtype=torch.bfloat16)
    nn.init.constant_(toy_mod.weight, 0.1)
    w_init_bf16 = toy_mod.weight.detach().float().clone()
    m_opt = FP32MasterAdamW(toy_mod, [{"name": "policy_decay", "params": [toy_mod.weight], "lr": 3e-5, "weight_decay": 1e-10}])
    m_ema = TrainableFP32EMA(toy_mod, decay=0.9)
    for _ in range(20):
        m_opt.zero_grad()
        toy_mod.weight.grad = torch.ones_like(toy_mod.weight)
        m_opt.clip_and_step(grad_clip=1.0)
        m_ema.update(toy_mod, master_override=m_opt.master_named_tensors())
    master_delta = float((m_opt.master_named_tensors()["weight"] - w_init_bf16).abs().mean().item())
    ema_delta_val = float((m_ema.shadow["weight"] - w_init_bf16).abs().mean().item())
    assert master_delta > 4e-4 and ema_delta_val > 2e-4, (
        f"FP32MasterAdamW failed to accumulate sub-ULP bf16 steps: master_delta={master_delta}, ema={ema_delta_val}"
    )

    # ---- Check 15: Track-2 `p2_m5_hybrid_surprise_gdn` Step-0 `0.0` Parity, Bias Guard, & `k=2` Online Parity (`#248`/`#249`) ----
    pos_np = PosEmb3D(dim=768)(np.array([0, 2, 17, 145], dtype=np.int64), 4)
    pos_th = TorchPosEmb3D(dim=768).to(dev)(torch.tensor([0, 2, 17, 145], device=dev, dtype=torch.long))
    posemb3d_diff = float((pos_th.cpu() - torch.from_numpy(pos_np)).abs().max().item())
    assert posemb3d_diff < 1e-6, f"TorchPosEmb3D mismatch vs PosEmb3D: {posemb3d_diff}"

    cfg_m5 = Pi05MemConfig.tiny_test_preset("p2_m5_hybrid_surprise_gdn")
    pi05_m5 = Pi05Model(build_pi05_config_from_mem_cfg(cfg_m5)).to(dev)
    # Simulate 80k-step pretrained `perceptual_framesamp_modul_pt` with non-zero `mem_rms_norm_ffn.dense.bias`
    for blk in pi05_m5.paligemma_with_expert.layers:
        assert blk.mem_rms_norm_ffn is not None and blk.mem_rms_norm_ffn.dense is not None
        nn.init.normal_(blk.mem_rms_norm_ffn.dense.weight, std=0.05)
        nn.init.constant_(blk.mem_rms_norm_ffn.dense.bias, 0.37)
        assert blk.expert.post_attention_layernorm.dense is not None
        blk.expert.post_attention_layernorm.dense.bias.data[2 * cfg_m5.expert_width :].fill_(1.0)

    pol_m5 = DeltaMemPi05Policy(cfg_m5, pi05_model=pi05_m5, device=dev, precision="float32")
    pol_m5.configure_stage("stage_a", freeze_hooks_in_stage_a=True)
    # Catch #2: pretrained `mem_rms_norm_ffn.dense.bias` MUST NOT be wiped to 0 on `mme_vla_exact`!
    bias_after_cfg = float(
        pol_m5.pi05.paligemma_with_expert.layers[0].mem_rms_norm_ffn.dense.bias.abs().mean().item()  # type: ignore[union-attr]
    )
    assert abs(bias_after_cfg - 0.37) < 1e-6, f"CRITICAL: mme_vla_exact bias wiped: {bias_after_cfg}"

    # Verify Step-0 `0.0` exact diff vs pure `pi05.mem_encoder` on non-distractor episode `synth_ds[1]`!
    ep_clean = synth_ds[1]
    pol_m5.eval()
    with torch.no_grad():
        assert pol_m5.mem_core.hybrid_bridge is not None and pol_m5.pi05.mem_encoder is not None
        m5_out = pol_m5.mem_core.forward_episode_chunks(
            ep_front_emb=ep_clean["front_emb"].to(dev),
            ep_wrist_emb=ep_clean["wrist_emb"].to(dev),
            ep_state_norm=ep_clean["state_norm"].to(dev),
            ep_is_demo=ep_clean["is_demo"].to(dev),
            exec_start_idx=int(ep_clean["exec_start_idx"]),
            chunk_step_indices=ep_clean["chunk_step_indices"][:2].to(dev),
            readout_cutoff_excl=ep_clean["readout_cutoff_excl"][:2].to(dev),
            curr_front_emb=ep_clean["exec_front_256"][:2].to(dev),
            curr_wrist_emb=ep_clean["exec_wrist_256"][:2].to(dev),
            curr_state_norm=ep_clean["state_norm"][ep_clean["chunk_step_indices"][:2]].to(dev),
            ep_prompt_cos_jump=ep_clean["prompt_cos_jump"].to(dev),
            perceptual_mem_encoder=pol_m5.pi05.mem_encoder,
        )
        # Pure `PerceptualMemoryEncoder` reference at chunk 0 (`step_idx == 0` -> frame 0, `local_pos == 0`)
        w_idx_c = select_episode_write_indices(
            int(ep_clean["front_emb"].shape[0]), int(ep_clean["exec_start_idx"]), cfg_m5, device=dev
        )
        f0_4x4 = pool_spatial_tokens_to_square(ep_clean["front_emb"].to(dev)[w_idx_c[:1]], 16)
        p0_768 = pol_m5.mem_core.hybrid_bridge.pos_emb_3d(torch.zeros((1,), device=dev, dtype=torch.long))
        ref_tok0 = pol_m5.pi05.mem_encoder(f0_4x4.reshape(16, -1), p0_768.reshape(16, -1), None)
        m5_step0_diff = float((m5_out["mem_tokens_l1"][0, :16] - ref_tok0).abs().max().item())
        assert m5_step0_diff == 0.0 and int(m5_out["mem_mask"][0].sum().item()) == 16, (
            f"p2_m5 Step-0 parity violated: diff={m5_step0_diff}, active={int(m5_out['mem_mask'][0].sum().item())}"
        )
        # Also verify compact 96-token canvas w/ `q_rope_offset=128` matches 128-padded HistoryBlock `0.0` bit-for-bit!
        kv_m5 = pol_m5.pi05.prefill_prefix_kv(
            ep_clean["lang_tokens"].unsqueeze(0).to(dev),
            ep_clean["lang_masks"].unsqueeze(0).to(dev),
            image_embs=[ep_clean["exec_front_256"][:1].to(dev), ep_clean["exec_wrist_256"][:1].to(dev)],
        )
        v_compact = pol_m5.pi05.denoise_step_from_kv(
            kv_m5,  # type: ignore[arg-type]
            x_t[:1],
            t_vec[:1],
            mem_tokens=m5_out["mem_tokens_l1"][:1],
            mem_mask=m5_out["mem_mask"][:1],
        )
        pad_128_tok = torch.zeros((1, 128, cfg_m5.mem_token_dim), device=dev)
        pad_128_mask = torch.zeros((1, 128), device=dev, dtype=torch.bool)
        pad_128_tok[0, :16] = ref_tok0
        pad_128_mask[0, :16] = True
        v_pad128 = pol_m5.pi05.denoise_step_from_kv(
            kv_m5,  # type: ignore[arg-type]
            x_t[:1],
            t_vec[:1],
            mem_tokens=pad_128_tok,
            mem_mask=pad_128_mask,
        )
        m5_rope_diff = float((v_compact - v_pad128).abs().max().item())
        assert m5_rope_diff < 1e-5, f"q_rope_offset=128 compact canvas mismatch vs 128-pad: {m5_rope_diff}"
        assert bool(pol_m5.pi05.paligemma_with_expert.layers[0].mem_attn.k_rope_span_frames) is True  # type: ignore[union-attr]
        # `#259` Bug #2: chunk 1 (`t=16`) must include recent `[0..16]` frames (`> 16` active tokens, not `excl=16` blinded!)
        assert int(m5_out["mem_mask"][1].sum().item()) > 16
        # `#260` Bug #4: even under 50x SwiGLU weight inflation, `|tanh(gamma)|=0.02` perturbation is bounded to ~2%!
        orig_wgu = pol_m5.mem_core.hybrid_bridge.ffn.w_gate_up.weight.data.clone()
        pol_m5.mem_core.hybrid_bridge.ffn.w_gate_up.weight.data.mul_(50.0)
        pol_m5.mem_core.hybrid_bridge.gamma_hybrid.data.fill_(float(math.atanh(0.02)))
        dummy_canvas = torch.randn((2, cfg_m5.hybrid_canvas_tokens, cfg_m5.mem_token_dim), device=dev) * 100.0
        dummy_cmask = torch.ones((2, cfg_m5.hybrid_canvas_tokens), device=dev, dtype=torch.bool)
        dummy_dm = torch.randn((2, 13, cfg_m5.mem_token_dim), device=dev)
        dummy_dmask = torch.ones((2, 13), device=dev, dtype=torch.bool)
        dummy_ctx = torch.randn((2, 8, cfg_m5.d_mem), device=dev)
        fused_50x = pol_m5.mem_core.hybrid_bridge(dummy_canvas, dummy_cmask, dummy_dm, dummy_dmask, dummy_ctx)
        rel_pert = float((fused_50x - dummy_canvas).pow(2).mean().sqrt() / dummy_canvas.pow(2).mean().sqrt())
        assert 0.018 <= rel_pert <= 0.022, f"#260 SwiGLU RMS bound violated under 50x weights: rel_pert={rel_pert}"
        pol_m5.mem_core.hybrid_bridge.ffn.w_gate_up.weight.data.copy_(orig_wgu)
        pol_m5.mem_core.hybrid_bridge.gamma_hybrid.data.zero_()

    # Verify Step-0 backward gives non-zero grads to `gamma_hybrid`, `gamma_detail`, AND `DeltaMem`!
    pol_m5.train()
    out_m5_tr = pol_m5.forward_training_episode_batch([synth_ds[0], synth_ds[1]], max_chunks_per_ep=2)
    out_m5_tr["loss"].backward()
    assert pol_m5.mem_core.hybrid_bridge.gamma_hybrid.grad is not None
    assert float(pol_m5.mem_core.hybrid_bridge.gamma_hybrid.grad.abs().max().item()) > 0.0
    gdn_m5_gn = sum(
        float(p.grad.norm().item()) for p in pol_m5.mem_core.gdn.parameters() if p.grad is not None  # type: ignore[union-attr]
    )
    assert gdn_m5_gn > 0.0, f"Expected non-zero GDN grad via aux heads on Step 0, got {gdn_m5_gn}"
    pol_m5.zero_grad(set_to_none=True)

    # Verify online `DeltaMemEvalAdapter` works cleanly on `p2_m5_hybrid_surprise_gdn` including `k=2` reset!
    with tempfile.TemporaryDirectory() as td_m5:
        ck_m5 = Path(td_m5) / "m5_ckpt.pt"
        pol_m5.mem_core.hybrid_bridge.gamma_hybrid.data.fill_(0.25)
        pol_m5.mem_core.hybrid_bridge.gamma_detail.data.fill_(0.15)
        torch.save({"ema_delta": pol_m5.export_delta_state_dict()}, ck_m5)
        ad_m5 = build_eval_adapter(ck_m5, pol_m5.pi05, dev)
        st_z = np.zeros((6, 8), dtype=np.float32)
        f_stream = torch.randn((6, cfg_m5.front_tokens_curr, cfg_m5.vis_dim), device=dev)
        w_stream = torch.randn((6, cfg_m5.wrist_tokens_curr, cfg_m5.vis_dim), device=dev)
        ad_m5.ingest_frames(f_stream, w_stream, st_z, st_z, task_goal="session one goal", is_demo=True)
        ad_m5.ingest_frames(f_stream[:1], w_stream[:1], st_z[:1], st_z[:1], task_goal="session one goal", is_demo=False)
        mt_m5_1, mm_m5_1, _ = ad_m5.read_memory(
            prefix_hidden=None,
            front_tokens_256=f_stream[:1],
            wrist_tokens_256=w_stream[:1],
            raw_state=st_z[0],
            norm_state=st_z[0],
            task_goal="session one goal",
            step_idx=0,
            exec_start_idx=6,
        )
        assert mt_m5_1 is not None and mm_m5_1 is not None
        assert mt_m5_1.shape == (1, cfg_m5.hybrid_canvas_tokens, cfg_m5.mem_token_dim)
        # Trigger un-reset `k=2` new session -> must flush old demo keyframes down to 16 active tokens (`local_pos=0`)!
        st_k2 = np.ones((1, 8), dtype=np.float32) * 1.5
        ad_m5.ingest_frames(f_stream[2:3], w_stream[2:3], st_k2, st_k2, task_goal="completely new task", is_demo=False)
        _, mm_m5_k2, info_m5_k2 = ad_m5.read_memory(
            prefix_hidden=None,
            front_tokens_256=f_stream[2:3],
            wrist_tokens_256=w_stream[2:3],
            raw_state=st_k2[0],
            norm_state=st_k2[0],
            task_goal="completely new task",
            step_idx=0,
            exec_start_idx=0,
        )
        assert info_m5_k2["boundary_reset_max"] >= 0.99 and int(mm_m5_k2.sum().item()) == 16  # type: ignore[union-attr]

    # ---- Check 16: `Stratified16TaskBatchSampler` & Catch #1 `hooks` Optimizer Group LR > 0 (`#248`/`#249`) ----
    synth_32 = SyntheticPi05EpisodeDataset(cfg_m2, num_episodes=32, seed=99)
    strat_samp = Stratified16TaskBatchSampler(synth_32, batch_size=4, seed=42)
    strat_batches = list(strat_samp)
    assert len(strat_batches) == 8
    first_macro_tasks = {synth_32[i]["task_name"] for b in strat_batches[:4] for i in b}
    second_macro_tasks = {synth_32[i]["task_name"] for b in strat_batches[4:8] for i in b}
    assert len(first_macro_tasks) == 16 and len(second_macro_tasks) == 16, (
        f"Stratified16TaskBatchSampler failed 16-task coverage: {len(first_macro_tasks)}, {len(second_macro_tasks)}"
    )

    pol_m2.configure_stage("stage_b", freeze_mem_in_stage_b=True, enable_grad_ckpt=False)
    grps_b = pol_m2.build_optimizer_param_groups(
        lr_mem=2e-4, lr_gate=5e-4, lr_policy=1.2e-5, lr_hooks=8e-5, stage_b_mem_lr_scale=0.0
    )
    hook_lrs = [g["lr"] for g in grps_b if g["name"].startswith("hooks_")]
    assert hook_lrs and all(lr == 8e-5 for lr in hook_lrs), f"Catch #1 regression: hook_lrs={hook_lrs}"

    # ---- Check 17: Regularized `ActionExpert LoRA` + `hybrid_use_landmarks=True` + `anc=1.0` Roundtrip (`#274`/`#275`) ----
    cfg_m5_peft = Pi05MemConfig.tiny_test_preset(
        "p2_m5_hybrid_surprise_gdn",
        train_stage="stage_b",
        use_vlm_lora=False,
        use_expert_lora=True,
        expert_lora_rank=8,
        expert_lora_alpha=8.0,
        expert_lora_dropout=0.05,
        hybrid_use_landmarks=True,
        hybrid_anchor_ratio=1.0,
    )
    assert cfg_m5_peft.use_landmark_bank is True
    pol_peft = DeltaMemPi05Policy(cfg_m5_peft, device=dev, precision="float32")
    assert pol_peft.mem_core.landmark_bank is not None
    # Emulate pretrained `perceptual_framesamp_modul_pt` non-zero `HistoryBlock` / AdaRMS gates before `ActionExpert LoRA`
    for blk in pol_peft.pi05.paligemma_with_expert.layers:
        assert blk.mem_rms_norm_ffn is not None and blk.mem_rms_norm_ffn.dense is not None
        nn.init.normal_(blk.mem_rms_norm_ffn.dense.weight, std=0.05)
        nn.init.constant_(blk.mem_rms_norm_ffn.dense.bias, 0.37)
        assert blk.expert.post_attention_layernorm.dense is not None
        blk.expert.post_attention_layernorm.dense.bias.data[2 * cfg_m5_peft.expert_width :].fill_(1.0)
    lt_p = ep_clean["lang_tokens"].unsqueeze(0).to(dev)
    lm_p = ep_clean["lang_masks"].unsqueeze(0).to(dev)
    fc_p = ep_clean["exec_front_256"][:1].to(dev)
    wc_p = ep_clean["exec_wrist_256"][:1].to(dev)
    ns_p = x_t[:1].clone()
    pol_peft.eval()
    with torch.no_grad():
        kv_peft = pol_peft.pi05.prefill_prefix_kv(lt_p, lm_p, image_embs=[fc_p, wc_p])
        act_pre_lora = pol_peft.pi05.sample_actions_from_kv(kv_peft, noise=ns_p, num_steps=4)  # type: ignore[arg-type]

    pol_peft.configure_stage("stage_b", freeze_mem_in_stage_b=False, enable_grad_ckpt=False)
    assert pol_peft.has_expert_lora and not pol_peft.has_vlm_lora
    assert all(not p.requires_grad for n, p in pol_peft.pi05.named_parameters() if ".expert." in n and ".base." in n)
    assert any(p.requires_grad for n, p in pol_peft.pi05.named_parameters() if ".expert." in n and "lora_a" in n)
    pol_peft.eval()
    with torch.no_grad():
        act_post_lora0 = pol_peft.pi05.sample_actions_from_kv(kv_peft, noise=ns_p, num_steps=4)  # type: ignore[arg-type]
    expert_lora_step0_diff = float((act_pre_lora - act_post_lora0).abs().max().item())
    assert expert_lora_step0_diff == 0.0, f"Expert LoRA Step-0 non-zero diff: {expert_lora_step0_diff}"

    pol_peft.train()
    grps_peft = pol_peft.build_optimizer_param_groups(
        lr_mem=2e-4, lr_gate=2e-3, lr_policy=4e-5, lr_hooks=2e-5, stage_b_mem_lr_scale=0.3
    )
    opt_peft = FP32MasterAdamW(pol_peft, grps_peft)
    ema_peft = TrainableFP32EMA(pol_peft, decay=0.99)
    synth_peft = SyntheticPi05EpisodeDataset(cfg_m5_peft, num_episodes=4, seed=21)
    for _ in range(2):
        opt_peft.zero_grad()
        out_p = pol_peft.forward_training_episode_batch([synth_peft[0], synth_peft[1]], max_chunks_per_ep=3)
        out_p["loss"].backward()
        opt_peft.clip_and_step(grad_clip=1.0)
        ema_peft.update(pol_peft, master_override=opt_peft.master_named_tensors())
    assert pol_peft.mem_core.hybrid_bridge is not None
    gh_after2 = float(torch.tanh(pol_peft.mem_core.hybrid_bridge.gamma_hybrid.detach().float()).abs().mean().item())
    assert gh_after2 > 1e-4, f"Expected gamma_hybrid to grow with lr_gate=2e-3, got {gh_after2}"

    with tempfile.TemporaryDirectory() as tmp_d:
        ck_p = Path(tmp_d) / "peft_ema.pt"
        ed_p = ema_peft.export_ema_delta(pol_peft)
        assert ed_p["has_expert_lora"] is True and not any(".base." in k for k in ed_p["pi05_delta"])
        torch.save({"step": 2, "ema_delta": ed_p}, ck_p)
        fresh_pi05 = Pi05Model(build_pi05_config_from_mem_cfg(cfg_m5_peft)).to(dev)
        fresh_pi05.load_state_dict(
            {k.replace(".base.", "."): v for k, v in pol_peft.pi05.state_dict().items() if "lora_" not in k},
            strict=False,
        )
        ad_peft = build_eval_adapter(ck_p, fresh_pi05, device=dev)
        assert ad_peft.cfg.use_landmark_bank is True and ad_peft.cfg.hybrid_anchor_ratio == 1.0
        pol_peft_ema_check = DeltaMemPi05Policy(cfg_m5_peft, pi05_model=fresh_pi05, device=dev, precision="float32")
        pol_peft_ema_check.load_delta_state_dict(ed_p, strict_mem=True)
        pol_peft_ema_check.eval()
        with torch.no_grad():
            kv_f = pol_peft_ema_check.pi05.prefill_prefix_kv(lt_p, lm_p, image_embs=[fc_p, wc_p])
            act_reloaded = pol_peft_ema_check.pi05.sample_actions_from_kv(kv_f, noise=ns_p, num_steps=4)  # type: ignore[arg-type]
        assert torch.isfinite(act_reloaded).all()

    # ---- Check 18: `DeltaGround-π0.5` `<row_y, col_x>` Reversible Codec, 3-Hop Looped Pointer, & `predict_subgoal` (`#289`/`#290`/`#292`) ----
    sg_two = "hook the cube at <114, 95> to the target at <168, 143>"
    tmpl_2, coords_2 = decompose_grounded_subgoal(sg_two)
    assert tmpl_2 == "hook the cube at <bbox> to the target at <bbox>"
    assert coords_2 == [(95.0, 114.0), (143.0, 168.0)]  # Canonical (x_col, y_row)!
    xy1_n = pixel_xy_to_norm_xy(torch.tensor(coords_2[0]), align_corners=False)
    xy2_n = pixel_xy_to_norm_xy(torch.tensor(coords_2[1]), align_corners=False)
    rebuilt_2 = render_grounded_subgoal_from_norm_xy(tmpl_2, xy1_n, xy2_n)
    assert rebuilt_2 == sg_two, f"Axis-swap or rounding error in DeltaGround codec: {rebuilt_2!r} != {sg_two!r}"

    dg_vocab = ["", "pick up the red cube at <bbox>", tmpl_2, "put down the container"]
    dg_cfg = DeltaGroundConfig.tiny_test_preset(num_templates=len(dg_vocab), num_hops=3)
    dg_head_w = torch.randn((dg_cfg.vis_dim, dg_cfg.raw_siglip_dim), device=dev) * 0.02
    dg_head_b = torch.zeros((dg_cfg.vis_dim,), device=dev)
    dg_model = DeltaGroundModel(
        dg_cfg, template_vocab=dg_vocab, siglip_head_weight=dg_head_w, siglip_head_bias=dg_head_b
    ).to(dev)
    dg_ep = {
        "exec_start_idx": 4,
        "lang_tokens": torch.randint(1, 100, (dg_cfg.max_prompt_len,), device=dev),
        "lang_masks": torch.ones((dg_cfg.max_prompt_len,), dtype=torch.bool, device=dev),
        "front_emb": torch.randn((10, dg_cfg.front_tokens_hist, dg_cfg.raw_siglip_dim), device=dev),
        "wrist_emb": torch.randn((10, dg_cfg.wrist_tokens_hist, dg_cfg.raw_siglip_dim), device=dev),
        "state_norm": torch.randn((10, 8), device=dev) * 0.2,
        "is_demo": torch.tensor([True] * 4 + [False] * 6, device=dev),
        "prompt_cos_jump": torch.zeros((10,), device=dev),
        "chunk_step_indices": torch.tensor([4, 7, 9], dtype=torch.long, device=dev),
        "exec_front_256": torch.randn((3, dg_cfg.front_tokens_curr, dg_cfg.raw_siglip_dim), device=dev),
        "exec_wrist_256": torch.randn((3, dg_cfg.wrist_tokens_curr, dg_cfg.raw_siglip_dim), device=dev),
        "template_id": torch.tensor([1, 2, 3], dtype=torch.long, device=dev),
        "xy1_m11": torch.stack([xy1_n.to(dev), xy1_n.to(dev), torch.zeros(2, device=dev)], dim=0),
        "has_xy1": torch.tensor([True, True, False], device=dev),
        "xy2_m11": torch.stack([torch.zeros(2, device=dev), xy2_n.to(dev), torch.zeros(2, device=dev)], dim=0),
        "has_xy2": torch.tensor([False, True, False], device=dev),
        "completed_count": torch.tensor([0, 1, 1], dtype=torch.long, device=dev),
        "remaining_count": torch.tensor([2, 1, 0], dtype=torch.long, device=dev),
        "stage_id": torch.tensor([5, 11, 4], dtype=torch.long, device=dev),
        "color_id": torch.tensor([1, 0, 0], dtype=torch.long, device=dev),
        "ordinal_id": torch.tensor([1, 0, 0], dtype=torch.long, device=dev),
    }
    dg_hops = dg_model.forward_episode(dg_ep)
    assert len(dg_hops) == 3 and dg_hops[2]["template_logits"].shape == (3, len(dg_vocab))
    dg_loss = dg_model.compute_deep_supervision_loss(dg_hops, dg_ep)
    dg_loss["loss"].backward()
    assert dg_model.coda.gamma_loop.grad is not None
    assert float(dg_model.coda.gamma_loop.grad.abs().max().item()) > 0.0

    with tempfile.TemporaryDirectory() as td_dg:
        ck_dg = Path(td_dg) / "dg.pt"
        torch.save(
            {
                "ema_deltaground": {
                    "deltaground_config": dg_cfg.to_dict(),
                    "template_vocab": dg_vocab,
                    "state_dict": {k: v.detach().cpu() for k, v in dg_model.state_dict().items()},
                }
            },
            ck_dg,
        )
        ad_dg = build_deltaground_adapter(ck_dg, pol_m5.pi05, device=dev, hop_mode="auto")
        f_dg_s = torch.randn((4, dg_cfg.front_tokens_curr, dg_cfg.vis_dim), device=dev)
        w_dg_s = torch.randn((4, dg_cfg.wrist_tokens_curr, dg_cfg.vis_dim), device=dev)
        st_dg_s = np.zeros((4, 8), dtype=np.float32)
        ad_dg.ingest_frames(f_dg_s, w_dg_s, st_dg_s, st_dg_s, task_goal="pick up the red cube", is_demo=True)
        ad_dg.ingest_frames(f_dg_s[:1], w_dg_s[:1], st_dg_s[:1], st_dg_s[:1], task_goal="pick up the red cube", is_demo=False)
        _ = ad_dg.predict_subgoal(
            front_tokens_256=f_dg_s[:1],
            wrist_tokens_256=w_dg_s[:1],
            raw_state=st_dg_s[0],
            norm_state=st_dg_s[0],
            task_goal="pick up the red cube",
            step_idx=0,
            exec_start_idx=4,
        )
        mt_dg, mm_dg, info_dg = ad_dg.read_memory(
            prefix_hidden=None,
            front_tokens_256=f_dg_s[:1],
            wrist_tokens_256=w_dg_s[:1],
            raw_state=st_dg_s[0],
            norm_state=st_dg_s[0],
            task_goal="pick up the red cube",
            step_idx=0,
            exec_start_idx=4,
        )
        assert mt_dg is None and mm_dg is None and info_dg["loop_depth"] == 3

        # ---- Check 19: `P2-M7` Unified Grounded-Hybrid (`P2-M7A` Grafted Step-0 `0.0` Diff + `UnifiedGroundedHybridEvalAdapter` Union Pass + `#315` Twins) ----
        tw_pairs = build_occlusion_twin_template_pairs(["", "pick up the cube", "pick up the cube at <bbox>"])
        assert tw_pairs == [(1, 2)]
        lg_test = torch.tensor([[0.0, 2.0, 1.8]])
        lg_fold = fold_occlusion_twin_logits(lg_test, tw_pairs)
        assert int(lg_fold.argmax(dim=-1).item()) == 2

        cfg_m7a = Pi05MemConfig.tiny_test_preset(
            "p2_m7_grounded_hybrid",
            modulator_mode="grafted_residual",
            graft_gate_init=0.0,
            hybrid_use_landmarks=True,
            use_expert_lora=True,
        )
        sym_base_cfg = build_pi05_config_from_mem_cfg(Pi05MemConfig.tiny_test_preset("p2_b0_pi05_base", max_token_len=cfg_m7a.max_token_len))
        sym_base = Pi05Model(sym_base_cfg).to(dev).eval()
        for blk_s in sym_base.paligemma_with_expert.layers:
            assert blk_s.expert.post_attention_layernorm.dense is not None
            blk_s.expert.post_attention_layernorm.dense.bias.data[2 * cfg_m7a.expert_width :].fill_(1.0)
        nn.init.normal_(sym_base.action_out_proj.weight, std=0.05)
        sym_sd = {k: v.detach().clone() for k, v in sym_base.state_dict().items()}
        pol_m7a = DeltaMemPi05Policy(cfg_m7a, pi05_model=Pi05Model(sym_base_cfg).to(dev), device=dev, precision="float32")
        pol_m7a.pi05.load_state_dict(sym_sd, strict=False)
        # Non-zero pretrained-like dense weights inside GraftedMemoryRMSNorm: graft_gate == 0.0 must still yield EXACT 0.0 diff!
        for blk in pol_m7a.pi05.paligemma_with_expert.layers:
            assert isinstance(blk.mem_rms_norm_ffn, GraftedMemoryRMSNorm)
            nn.init.normal_(blk.mem_rms_norm_ffn.dense.weight, std=0.05)
            nn.init.normal_(blk.mem_rms_norm_ffn.dense.bias, std=0.05)
        pol_m7a.configure_stage("stage_b")
        pol_m7a.eval()

        ds_m7 = SyntheticPi05EpisodeDataset(cfg_m7a, num_episodes=4, seed=77)
        ep_m7 = ds_m7[0]
        q0_m7 = ep_m7["chunk_vlm_tokens"][:1].to(dev)
        m0_m7 = ep_m7["chunk_vlm_masks"][:1].to(dev)
        f0_m7 = ep_m7["exec_front_256"][:1].to(dev)
        w0_m7 = ep_m7["exec_wrist_256"][:1].to(dev)
        ns_m7 = torch.randn((1, cfg_m7a.action_horizon, cfg_m7a.padded_action_dim), device=dev)
        with torch.no_grad():
            a_sym = sym_base.sample_actions(q0_m7, m0_m7, image_embs=[f0_m7, w0_m7], noise=ns_m7, num_steps=3)
            mo_m7 = pol_m7a.mem_core.forward_episode_chunks(
                ep_front_emb=ep_m7["front_emb"].to(dev),
                ep_wrist_emb=ep_m7["wrist_emb"].to(dev),
                ep_state_norm=ep_m7["state_norm"].to(dev),
                ep_is_demo=ep_m7["is_demo"].to(dev),
                exec_start_idx=int(ep_m7["exec_start_idx"]),
                chunk_step_indices=ep_m7["chunk_step_indices"][:1].to(dev),
                readout_cutoff_excl=ep_m7["readout_cutoff_excl"][:1].to(dev),
                curr_front_emb=f0_m7,
                curr_wrist_emb=w0_m7,
                curr_state_norm=ep_m7["state_norm"][ep_m7["chunk_step_indices"][:1]].to(dev),
                perceptual_mem_encoder=pol_m7a.pi05.mem_encoder,
            )
            a_m7 = pol_m7a.pi05.sample_actions(
                q0_m7, m0_m7, image_embs=[f0_m7, w0_m7], noise=ns_m7, num_steps=3,
                mem_tokens=mo_m7["mem_tokens_l1"], mem_mask=mo_m7["mem_mask"],
            )
        m7a_step0_diff = float((a_sym - a_m7).abs().max().item())
        assert m7a_step0_diff == 0.0, f"P2-M7A Step-0 diff {m7a_step0_diff} != 0.0!"

        # Verify Step-1 backward flows non-zero gradient into `graft_gate`!
        pol_m7a.train()
        out_m7 = pol_m7a.forward_training_episode_batch([ds_m7[0], ds_m7[1]], max_chunks_per_ep=2)
        out_m7["loss"].backward()
        gg_grad = max(
            float(blk.mem_rms_norm_ffn.graft_gate.grad.abs().max().item())  # type: ignore[union-attr]
            for blk in pol_m7a.pi05.paligemma_with_expert.layers
        )
        assert gg_grad > 0.0, "Expected non-zero Step-1 gradient into GraftedMemoryRMSNorm.graft_gate!"

        # Verify bundled save -> `build_grounded_hybrid_adapter` single-pass union ingest & `k=2` reset!
        ck_m7 = Path(td_dg) / "m7_bundled.pt"
        torch.save(
            {
                "ema_delta": pol_m7a.export_delta_state_dict(),
                "deltaground": {
                    "deltaground_config": dg_cfg.to_dict(),
                    "template_vocab": dg_vocab,
                    "state_dict": {k: v.detach().cpu() for k, v in dg_model.state_dict().items()},
                },
            },
            ck_m7,
        )
        ad_m7 = build_grounded_hybrid_adapter(ck_m7, Pi05Model(sym_base_cfg).to(dev), device=dev, hop_mode="l3")
        assert isinstance(ad_m7, UnifiedGroundedHybridEvalAdapter)
        u_sel = ad_m7.select_write_indices(8, task_goal="pick up the red cube", is_demo=True, total_demo_len=8)
        assert len(u_sel) > 0
        ad_m7.ingest_selected_frames(
            f_dg_s[:1].expand(len(u_sel), -1, -1),
            w_dg_s[:1].expand(len(u_sel), -1, -1),
            np.zeros((len(u_sel), 8), dtype=np.float32),
            np.zeros((len(u_sel), 8), dtype=np.float32),
            task_goal="pick up the red cube",
            is_demo=True,
        )
        ad_m7.ingest_frames(f_dg_s[:1], w_dg_s[:1], st_dg_s[:1], st_dg_s[:1], task_goal="pick up the red cube", is_demo=False)
        _ = ad_m7.predict_subgoal(
            front_tokens_256=f_dg_s[:1],
            wrist_tokens_256=w_dg_s[:1],
            raw_state=st_dg_s[0],
            norm_state=st_dg_s[0],
            task_goal="pick up the red cube",
            step_idx=8,
            exec_start_idx=8,
        )
        mt_u, mm_u, info_u = ad_m7.read_memory(
            prefix_hidden=None,
            front_tokens_256=f_dg_s[:1],
            wrist_tokens_256=w_dg_s[:1],
            raw_state=st_dg_s[0],
            norm_state=st_dg_s[0],
            task_goal="pick up the red cube",
            step_idx=8,
            exec_start_idx=8,
        )
        assert mt_u is not None and mt_u.shape[1] == cfg_m7a.hybrid_canvas_tokens and info_u["loop_depth"] == 3
        ad_m7.ingest_frames(f_dg_s[:1], w_dg_s[:1], st_dg_s[:1], st_dg_s[:1], task_goal="press blue button", is_demo=False)
        _ = ad_m7.predict_subgoal(
            front_tokens_256=f_dg_s[:1],
            wrist_tokens_256=w_dg_s[:1],
            raw_state=st_dg_s[0],
            norm_state=st_dg_s[0],
            task_goal="press blue button",
            step_idx=0,
            exec_start_idx=0,
        )
        _, _, info_u_k2 = ad_m7.read_memory(
            prefix_hidden=None,
            front_tokens_256=f_dg_s[:1],
            wrist_tokens_256=w_dg_s[:1],
            raw_state=st_dg_s[0],
            norm_state=st_dg_s[0],
            task_goal="press blue button",
            step_idx=0,
            exec_start_idx=0,
        )
        assert info_u_k2["boundary_reset_max"] >= 0.99

        # ---- Check 20: `P2-M8` VLM-Prefix GDN-Surprise Reveal Keyframes (`#329`/`#330`/`#331`) ----
        cfg_m8 = Pi05MemConfig.tiny_test_preset(
            "p2_m8_vlmprefix_surprise_hybrid",
            hybrid_use_landmarks=True,
            use_vlm_lora=True,
            use_expert_lora=True,
            vlm_prefix_drop_prob=0.0,
        )
        pol_m8 = DeltaMemPi05Policy(cfg_m8, device=dev, precision="float32")
        for blk_8 in pol_m8.pi05.paligemma_with_expert.layers:
            assert blk_8.expert.input_layernorm.dense is not None
            assert blk_8.expert.post_attention_layernorm.dense is not None
            blk_8.expert.input_layernorm.dense.bias.data[2 * cfg_m8.expert_width :].fill_(1.0)
            blk_8.expert.post_attention_layernorm.dense.bias.data[2 * cfg_m8.expert_width :].fill_(1.0)
        nn.init.normal_(pol_m8.pi05.action_out_proj.weight, std=0.05)
        pol_m8.configure_stage("stage_b")
        assert pol_m8.mem_core.vlm_surprise_selector is not None

        ds_m8 = SyntheticPi05EpisodeDataset(cfg_m8, num_episodes=4, seed=91)
        ep_m8 = ds_m8[0]
        q_m8 = ep_m8["lang_tokens"].unsqueeze(0).to(dev)
        m_m8 = ep_m8["lang_masks"].unsqueeze(0).to(dev)
        f_m8 = ep_m8["exec_front_256"][:1].to(dev)
        w_m8 = ep_m8["exec_wrist_256"][:1].to(dev)

        # Verify exact unchanged RoPE positions for `[front, wrist, lang]` when `extra_prefix_embs` is appended!
        with torch.no_grad():
            _, pm_base, _ = pol_m8.pi05.embed_prefix(q_m8, m_m8, image_embs=[f_m8, w_m8])
            pos_base = torch.cumsum(pm_base.long(), dim=1) - 1
            mo_m8 = pol_m8.mem_core.forward_episode_chunks(
                ep_front_emb=ep_m8["front_emb"].to(dev),
                ep_wrist_emb=ep_m8["wrist_emb"].to(dev),
                ep_state_norm=ep_m8["state_norm"].to(dev),
                ep_is_demo=ep_m8["is_demo"].to(dev),
                exec_start_idx=int(ep_m8["exec_start_idx"]),
                chunk_step_indices=ep_m8["chunk_step_indices"][:1].to(dev),
                readout_cutoff_excl=ep_m8["readout_cutoff_excl"][:1].to(dev),
                curr_front_emb=f_m8,
                curr_wrist_emb=w_m8,
                curr_state_norm=ep_m8["state_norm"][ep_m8["chunk_step_indices"][:1]].to(dev),
                perceptual_mem_encoder=pol_m8.pi05.mem_encoder,
            )
            vp_tok = mo_m8["vlm_prefix_tokens"]
            vp_mask = mo_m8["vlm_prefix_mask"]
            assert vp_tok is not None and vp_mask is not None
            _, pm_ext, ar_ext = pol_m8.pi05.embed_prefix(
                q_m8, m_m8, image_embs=[f_m8, w_m8], extra_prefix_embs=vp_tok, extra_prefix_mask=vp_mask
            )
            pos_ext = torch.cumsum(pm_ext.long(), dim=1) - 1
            assert torch.equal(pos_ext[:, : pm_base.shape[1]], pos_base), "RoPE positions of front/wrist/lang shifted!"
            assert not bool(ar_ext.any()), "VLM prefix must stay 100% bidirectional (prefix_ar_mask == 0)!"

            # Verify forward_joint == prefill_prefix_kv + compute_loss_from_kv parity (`< 1e-5`)!
            pol_m8.eval()
            act_m8 = ep_m8["target_action_chunks"][:1].to(dev)
            ns_m8 = torch.randn((1, cfg_m8.action_horizon, cfg_m8.padded_action_dim), device=dev)
            tm_m8 = torch.tensor([0.42], device=dev, dtype=torch.float32)
            kv_m8 = pol_m8.pi05.prefill_prefix_kv(
                q_m8, m_m8, image_embs=[f_m8, w_m8], extra_prefix_embs=vp_tok, extra_prefix_mask=vp_mask
            )
            l_kv = pol_m8.pi05.compute_loss_from_kv(
                kv_m8, act_m8, noise=ns_m8, time=tm_m8, mem_tokens=mo_m8["mem_tokens_l1"], mem_mask=mo_m8["mem_mask"]  # type: ignore[arg-type]
            )
            l_jt = pol_m8.pi05.forward_joint(
                act_m8, q_m8, m_m8, image_embs=[f_m8, w_m8],
                extra_prefix_embs=vp_tok, extra_prefix_mask=vp_mask,
                noise=ns_m8, time=tm_m8, mem_tokens=mo_m8["mem_tokens_l1"], mem_mask=mo_m8["mem_mask"],
            )
            m8_joint_kv_diff = float((l_kv - l_jt).abs().max().item())
            assert m8_joint_kv_diff < 1e-5, f"P2-M8 joint vs KV loss diff {m8_joint_kv_diff}"

        # Verify backward flows non-zero gradients into `vlm_hist_slot_emb` & `vlm` LoRA!
        pol_m8.train()
        out_m8 = pol_m8.forward_training_episode_batch([ds_m8[0], ds_m8[1]], max_chunks_per_ep=2)
        out_m8["loss"].backward()
        slot_grad = float(pol_m8.mem_core.vlm_surprise_selector.vlm_hist_slot_emb.grad.abs().max().item())  # type: ignore[union-attr]
        vlm_lora_grad = max(
            float(p.grad.abs().max().item())
            for n, p in pol_m8.pi05.named_parameters()
            if ".vlm." in n and "lora_b" in n and p.grad is not None
        )
        assert slot_grad > 0.0 and vlm_lora_grad > 0.0, f"Zero P2-M8 grad: slot={slot_grad} vlm_lora={vlm_lora_grad}"

        # Verify `DeltaMemEvalAdapter.predict_vlm_prefix_tokens` online streaming & `k=2` boundary isolation!
        ck_m8 = Path(td_dg) / "m8_ckpt.pt"
        torch.save({"ema_delta": pol_m8.export_delta_state_dict()}, ck_m8)
        ad_m8 = build_eval_adapter(ck_m8, Pi05Model(build_pi05_config_from_mem_cfg(cfg_m8)).to(dev), device=dev)
        ad_m8.ingest_frames(
            f_m8.expand(8, -1, -1), w_m8.expand(8, -1, -1), np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32),
            task_goal="pick up container hiding green cube", is_demo=True, total_demo_len=8,
        )
        ad_m8.ingest_frames(
            f_m8, w_m8, np.zeros((1, 8), dtype=np.float32), np.zeros((1, 8), dtype=np.float32),
            task_goal="pick up container hiding green cube", is_demo=False,
        )
        ep_emb_8, ep_msk_8 = ad_m8.predict_vlm_prefix_tokens(
            front_tokens_256=f_m8, wrist_tokens_256=w_m8, raw_state=np.zeros((8,), dtype=np.float32),
            norm_state=np.zeros((8,), dtype=np.float32), task_goal="pick up container hiding green cube",
            step_idx=8, exec_start_idx=8,
        )
        assert ep_emb_8 is not None and ep_msk_8 is not None
        assert ep_emb_8.shape == (1, cfg_m8.vlm_prefix_surprise_keyframes * cfg_m8.vlm_prefix_tokens_per_frame, cfg_m8.vlm_width)
        assert bool(ep_msk_8.all())

    elapsed = time.perf_counter() - t0
    summary = {
        "status": "ALL_PASS",
        "device": str(dev),
        "has_fla": HAS_FLA,
        "step0_max_abs_diff_track1_archs": max_step0_diff,
        "p2_m5_hybrid_step0_diff": m5_step0_diff,
        "p2_m7a_grafted_step0_diff": m7a_step0_diff,
        "posemb3d_max_diff": posemb3d_diff,
        "stratified_macro_unique_tasks": len(first_macro_tasks),
        "shifted_query_surprise_max_diff": surp_diff,
        "interleaved_vs_segment_max_diff": interleaved_max_diff,
        "a1_intra_frame_decay_leak": a1_slot_leak,
        "empty_mask_cfg_max_diff": empty_mask_diff,
        "lora_step0_max_diff": lora_init_diff,
        "stage_c_hop2_step0_max_diff": hop2_init_diff,
        "on_device_siglip_head_diff": on_dev_proj_diff,
        "elapsed_sec": round(elapsed, 3),
    }
    print(f"[Self-Test] PASS in {elapsed:.2f}s:\n" + json.dumps(summary, indent=2), flush=True)
    return summary


# =============================================================================
# 3. Main Resumable Training Loop (`Stage A`, `Stage B`, `Stage C`)
# =============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Phase-2 DeltaMem-pi0.5 & T2Mem models on Miyabi GH200.")
    p.add_argument("--self-test", action="store_true", help="Run 18-check verification suite and exit.")
    p.add_argument("--train-deltaground", action="store_true", help="Train standalone <45M DeltaGround-pi0.5 (#289/#292).")
    p.add_argument("--train-deltaground-vlm", action="store_true", help="Train P2-M9 DeltaGround-VLM on shared PaliGemma-2B (#334).")
    p.add_argument("--dg-v2-mode", action="store_true", default=False, help="Enable P2-M9-v2 (#367/#368): shortcut-free stage-entry xy loss, K=12 swap-complete reveal, shared 2D coords & RevealCrossAttn.")
    p.add_argument("--dg-v2c-mode", action="store_true", default=False, help="Enable P2-M9-v2c (#376 Action B): restores 1D exec clock + 65%% interior stage transitions + unblurred raw-SigLIP reveal residual while keeping xy_loss 100%% neutral-arm stage-entry!")
    p.add_argument("--dg-v2d-mode", action="store_true", default=False, help="Enable P2-M9-v2d (#391(1)): K=14 Reference-Apex Delta_ref(t) NMS peaks (5 apex + 6 uniform + 3 GDN-surprise) fixing static container lift apexes on VideoUnmask*!")
    p.add_argument("--dg-v3-hr16-mode", action="store_true", default=False, help="Enable P2-M9-Hyb-v3-HR16 (#395(1)): unpooled 256-token (16x16, 14x14px) SigLIP reveal keyframes + shared 16x16 2d_sincos grid + Reference-Apex Delta_ref(t)!")
    p.add_argument("--dg-v4-pyramid-mode", action="store_true", default=False, help="Enable Wave-15 (#410) P2-M9v4 Hybrid-Pyramid Pointer: 4x256t (16x16) Apex-Reveal + 12x64t (8x8) Swap-Motion = 1,792 tokens!")
    p.add_argument("--dg-v5-advect-mode", action="store_true", default=False, help="Enable Wave-16 (#426) P2-M9v5 on top of v4 pyramid: two-sided transient apex prominence (Bug #1) + causal 8x8 patch-flow reveal advection onto the current front (Bug #2).")
    p.add_argument("--dg-prominence-tau", type=int, default=8, help="Write-step offset tau for two-sided transient apex prominence in --dg-v5-advect-mode.")
    p.add_argument("--dg-wrist-tokens", type=int, default=16, help="Current wrist spatial pool tokens for P2-M9 (16=4x4 default; 64=8x8 close-up wrist for Wave-15 stage tracker!).")
    p.add_argument("--dg-xy-loss-weight", type=float, default=1.0, help="Scalar multiplier on xy_loss in P2-M9 (1.0 default; 0.2 for dedicated stage-transition tracker!).")
    p.add_argument("--dg-reveal-tokens-per-frame", type=int, default=0, help="Reveal tokens per keyframe (0=auto: 256 in v3-hr16 else 64).")
    p.add_argument("--extract-hr256-cache", action="store_true", default=False, help="Extract 6-task unpooled 256-token write-step front features from LeRobot dataset into --hr-cache-dir and exit (#395(1a)).")
    p.add_argument("--lerobot-root", type=Path, default=Path("/work/gj26/b20098/datasets/robomme_data_lerobot"), help="Path to RoboMME LeRobot dataset for --extract-hr256-cache.")
    p.add_argument("--hr-cache-dir", type=Path, default=None, help="Directory of unpooled 256-token (16x16) specialist write-step front shards (#395(1)).")
    p.add_argument("--dg-apex-peaks", type=int, default=0, help="Number of Reference-State Patch Deviation Delta_ref(t) NMS peaks (5 in v2d, 4 in v3-hr16).")
    p.add_argument("--init-dg-vlm-ckpt", type=Path, default=None, help="Optional warm-start DeltaGround-VLM ckpt (e.g. v2c -> v2d with K=12->14 slot interpolation).")
    p.add_argument("--dg-reveal-keyframes", type=int, default=6, help="Reveal keyframes K for P2-M9 (6 for v1, 12 for v2, 14 for v2d).")
    p.add_argument("--dg-uniform-anchors", type=int, default=7, help="Uniform trajectory anchors inside K reveal keyframes for P2-M9-v2.")
    p.add_argument("--dg-no-mlp-lora", action="store_true", default=False, help="Attach dg_lora only to VLM q/k/v/o attention projections (blocks seat memorization in v2, #367).")
    p.add_argument("--dg-lora-rank", type=int, default=16)
    p.add_argument("--dg-lora-dropout", type=float, default=0.10)
    p.add_argument("--eval-every", type=int, default=200, help="Unseen ep 90..99 validation interval for P2-M9 (#334).")
    p.add_argument("--bundle-into-m7b-ckpt", type=Path, default=None, help="Optional P2-M7B s300 ckpt to bundle with best P2-M9 DeltaGround-VLM.")
    p.add_argument("--bundled-output-path", type=Path, default=None, help="Output path for 1-file P2-M7B+P2-M9 zero-retraining bundle.")
    p.add_argument("--deltaground-hops", type=int, default=3)
    p.add_argument("--online-subgoal-prob", type=float, default=0.85)
    p.add_argument("--smoke-synthetic", action="store_true", help="Run synthetic training smoke loop.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", type=str, default="bfloat16", choices=("bfloat16", "float32"))
    p.add_argument("--seed", type=int, default=42)

    # Architecture & Stage
    p.add_argument("--arch", type=str, default="p2_m2_deltamem", choices=ALL_PI05_MEM_ARCHS)
    p.add_argument("--stage", type=str, default="stage_a", choices=("stage_a", "stage_b", "stage_c"))
    p.add_argument("--exp-name", type=str, default="p2_m2_deltamem_stageA0")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "checkpoints" / "phase2")
    p.add_argument("--pi05-ckpt", type=Path, default=Path(DEFAULT_MIYABI_PI05_CKPT))
    p.add_argument("--cache-dir", type=Path, default=Path(DEFAULT_MIYABI_PI05_CACHE))
    p.add_argument("--init-delta-ckpt", type=Path, default=None, help="Warm-start delta ckpt (e.g. A0 -> B1 -> C).")
    p.add_argument("--resume", type=str, default="auto", choices=("auto", "none"))

    # Dataset & Batching
    p.add_argument("--tasks", type=str, default="all")
    p.add_argument("--stats-variant", type=str, default="pi05_baseline", choices=("pi05_baseline", "mme_vla", "framesamp_modul"))
    p.add_argument("--episodes-per-step", type=int, default=4)
    p.add_argument("--chunks-per-episode", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--stratified-16task", action="store_true", default=True, help="Stratify 16 RoboMME tasks across macro-steps.")
    p.add_argument("--no-stratified-16task", action="store_false", dest="stratified_16task")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--distractor-prob", type=float, default=0.25, help="Cross-session k=1..3 prefix prob in Stage A.")

    # Optimization & Schedule (`docs/SURVEY_PHASE2.md` §3 Prescriptions)
    p.add_argument("--max-steps", type=int, default=10_000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--lr-mem", type=float, default=2e-4)
    p.add_argument("--lr-gate", type=float, default=5e-4)
    p.add_argument("--lr-policy", type=float, default=3e-5)
    p.add_argument("--lr-hooks", type=float, default=None, help="Dedicated LR for 18L mem_attn/hooks (#249 Catch #1).")
    p.add_argument("--freeze-hooks-in-stage-a", action="store_true", default=False)
    p.add_argument("--stage-b-mem-lr-scale", type=float, default=0.0, help="0.0 freezes slow mem in Stage B (T2Mem).")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--warmup-ema", action="store_true", default=False, help="Enable (1+n)/(10+n) EMA warmup for short VLM-LoRA runs (#361).")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-vlm-lora", action="store_true", default=True)
    p.add_argument("--no-vlm-lora", action="store_false", dest="use_vlm_lora")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--use-expert-lora", action="store_true", default=False, help="Regularized LoRA on ActionExpert-300M (#274).")
    p.add_argument("--expert-lora-rank", type=int, default=16)
    p.add_argument("--expert-lora-dropout", type=float, default=0.05)
    p.add_argument("--freeze-dense-expert-in-stage-b", action="store_true", default=False)
    p.add_argument("--freeze-all-except-vlm-lora", action="store_true", default=False, help="P2-M7D (#361): freeze 100%% of ActionExpert/hooks/HybridBridge/mem_encoder, training ONLY PaliGemma VLM LoRA.")
    p.add_argument("--no-grad-ckpt", action="store_true", default=False, help="Disable VLM gradient checkpointing on 96GB GH200 for ~1.8x faster Stage B.")

    # Ablation & Track-2/3 Hybrid + Unified P2-M7 knobs
    p.add_argument("--modulator-mode", type=str, default=None, choices=("modul_zero_init", "modul_residual", "mme_vla_exact", "grafted_residual"))
    p.add_argument("--graft-history-ckpt", type=Path, default=None, help="Stream-load mem_encoder + 18L HistoryBlock from framesamp ckpt onto symbolic VLM (P2-M7A).")
    p.add_argument("--graft-gate-init", type=float, default=0.0, help="Initial per-layer GraftedMemoryRMSNorm scalar gate (`0.0` = bit-exact symbolic VLM).")
    p.add_argument("--skip-init-lora", action="store_true", default=False, help="Load mem_core + HistoryBlock from --init-delta-ckpt while keeping LoRA fresh zero-init (P2-M7A).")
    p.add_argument("--deltaground-ckpt", type=Path, default=None, help="Trained DeltaGround checkpoint for P2-M7 subgoal precomputation & checkpoint bundling.")
    p.add_argument("--deltaground-vlm-ckpt", type=Path, default=None, help="Trained P2-M9 DeltaGround-VLM ckpt for P2-M7C selective precompute & bundling (#342).")
    p.add_argument("--subgoal-selective-filter", action="store_true", default=False, help="Apply #339/#343 selective abstention (twin-summed conf>=0.60 + episode-level counting/place-order abstention) in P2-M7C.")
    p.add_argument("--subgoal-keep-place-order", action="store_true", default=False, help="Keep VideoPlaceOrder grounded when --subgoal-selective-filter is active.")
    p.add_argument("--subgoal-min-conf", type=float, default=0.60, help="Confidence threshold when --subgoal-selective-filter is active.")
    p.add_argument("--subgoal-cache-file", type=Path, default=None, help="Disk cache path for precomputed 1,600-ep DeltaGround subgoal strings.")
    p.add_argument("--subgoal-dg-prob", type=float, default=0.65, help="Prob of conditioning VLM prefix on DeltaGround predicted subgoals in P2-M7.")
    p.add_argument("--subgoal-oracle-prob", type=float, default=0.15, help="Prob of conditioning VLM prefix on jittered grounded_subgoals_online in P2-M7.")
    p.add_argument("--subgoal-jitter-px", type=int, default=6)
    p.add_argument("--use-continuous-vlm-ground", action="store_true", default=False, help="P2-M11 (#395(2)): inject continuous P2-M9-HR16 heatmaps + target ROIs + stage/color/xy tokens into HistoryBlock perc_canvas with max_token_len=64 ALWAYS!")
    p.add_argument("--cont-vlm-cache-dir", type=Path, default=None, help="Directory of precomputed P2-M11 continuous VLM grounding shards (#395(2)).")
    p.add_argument("--cont-vlm-jitter-cells", type=float, default=0.85, help="Training spatial heatmap shift + xy jitter std (in 16x16 patch cells) for P2-M11.")
    p.add_argument("--reactive-precision-mode", action="store_true", default=False, help="P2-M12 (#405): zero-oracle StopCube (excl=1, stride=1, 3x trigger oversampling) + InsertPeg (L=2 hop2 + demo climax landmarks + 3x contact/hole oversampling)!")
    p.add_argument("--train-ep-range", type=str, default=None, help="Optional half-open episode range 'start,end' (e.g. '0,90' for P2-M12 train split).")
    p.add_argument("--val-ep-range", type=str, default=None, help="Optional half-open episode range 'start,end' (e.g. '90,100' for unseen FM val split).")
    p.add_argument("--vlm-prefix-keyframes", type=int, default=4, help="K_vlm GDN-surprise reveal keyframes in PaliGemma prefix (P2-M8, #329).")
    p.add_argument("--vlm-prefix-tokens-per-frame", type=int, default=64, choices=(16, 64), help="64 (8x8 -> 256t) or 16 (4x4 -> 64t, #331).")
    p.add_argument("--vlm-prefix-demo-quota", type=int, default=3)
    p.add_argument("--vlm-prefix-early-quota", type=int, default=2)
    p.add_argument("--vlm-prefix-drop-prob", type=float, default=0.15, help="Stage-B whole-history extra_prefix_mask=False dropout (#331).")
    p.add_argument("--timescale-mode", type=str, default="three_band_consolidate")
    p.add_argument("--landmark-encoder", type=str, default="multiscale_pyramid", choices=("multiscale_pyramid", "perceiver"))
    p.add_argument("--num-landmarks", type=int, default=8)
    p.add_argument("--hybrid-num-keyframes", type=int, default=32)
    p.add_argument("--hybrid-canvas-tokens", type=int, default=512)
    p.add_argument("--hybrid-anchor-ratio", type=float, default=None, help="Override hybrid_anchor_ratio (e.g. 1.0 or 0.9375).")
    p.add_argument("--hybrid-use-landmarks", action="store_true", default=False, help="Enable K=8 Multi-Scale Surprise Landmarks in HybridBridge KV (#275).")
    p.add_argument("--no-self-subgoal-tokens", action="store_false", dest="use_self_subgoal_tokens", default=True)
    p.add_argument("--mem-cfg-dropout", type=float, default=0.10)
    p.add_argument("--stage-b-aux-scale", type=float, default=0.02)
    p.add_argument("--fm-loss-scale", type=float, default=4.0)
    p.add_argument("--loss-on-valid-8d-only", action="store_true", default=False)

    # Resumability & Miyabi 2h `interact-g` Walltime Guard
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=2500)
    p.add_argument("--keep-every", type=int, default=5000)
    p.add_argument("--max-wall-minutes", type=float, default=112.0, help="Graceful save+exit before 120m PBS limit.")
    return p


def infinite_episode_iterator(loader: DataLoader) -> Iterable[List[Dict[str, Any]]]:
    while True:
        for batch in loader:
            yield batch


def run_deltaground_training(args: argparse.Namespace) -> None:
    """Fast standalone trainer for `<45M`-param `DeltaGround-π0.5` (`< 10 min` for 15 epochs on GH200!)."""
    wall_start = time.perf_counter()
    set_all_seeds(args.seed)
    dev = torch.device(args.device)

    train_ds = DeltaGroundEpisodeDataset(
        cache_dir=args.cache_dir,
        tasks=parse_task_filter(args.tasks),
        max_steps_per_episode=args.chunks_per_episode,
        online_subgoal_prob=args.online_subgoal_prob,
        distractor_prob=args.distractor_prob,
        seed=args.seed,
    )
    dg_cfg = DeltaGroundConfig(
        num_hops=int(args.deltaground_hops),
        num_landmarks=max(8, int(args.num_landmarks)),
        num_templates=len(train_ds.template_vocab),
    )
    model = DeltaGroundModel(
        dg_cfg,
        template_vocab=train_ds.template_vocab,
        siglip_head_weight=train_ds.siglip_head_weight,
        siglip_head_bias=train_ds.siglip_head_bias,
    ).to(dev)
    if args.precision == "bfloat16":
        model.cast_bf16_keep_fp32_gates()

    gate_params: List[nn.Parameter] = []
    mem_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("gamma_loop", "salience_gate", "boundary_gate", "A_log", "dt_bias")):
            gate_params.append(p)
        else:
            mem_params.append(p)
    param_groups = [
        {"name": "gates", "params": gate_params, "lr": float(args.lr_gate), "weight_decay": 0.0},
        {"name": "memory", "params": mem_params, "lr": float(args.lr_mem), "weight_decay": float(args.weight_decay)},
    ]
    base_lrs = [float(g["lr"]) for g in param_groups]
    optimizer = FP32MasterAdamW(model, param_groups, betas=(0.9, 0.95), eps=1e-8)
    ema = TrainableFP32EMA(model, decay=min(float(args.ema_decay), 0.985), warmup_ema=True)

    exp_dir = args.output_dir / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = exp_dir / "latest.pt"
    metrics_jsonl = exp_dir / "metrics.jsonl"
    from robomme_eval.interface import MME_VLA_NORM_STATS

    ns_file = Path(args.cache_dir) / "norm_stats.json"
    if ns_file.exists():
        raw_ns = json.loads(ns_file.read_text())
        state_stats_dict = raw_ns.get("norm_stats", raw_ns)
    else:
        state_stats_dict = MME_VLA_NORM_STATS

    start_step = 0
    best_loss = float("inf")
    if args.resume == "auto" and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["deltaground"]["state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"], device=dev)
        if "ema_shadow" in ckpt:
            ema.load_state_dict(ckpt["ema_shadow"], device=dev)
        start_step = int(ckpt["step"])
        best_loss = float(ckpt.get("best_loss", float("inf")))
        print(f"[DeltaGround Resume] step={start_step}/{args.max_steps}", flush=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[DeltaGround Train] templates={len(train_ds.template_vocab)} hops={dg_cfg.num_hops} "
        f"landmarks={dg_cfg.num_landmarks} trainable={n_params:,} steps={start_step}->{args.max_steps} "
        f"online_prob={args.online_subgoal_prob} (HAS_FLA={HAS_FLA})",
        flush=True,
    )

    eff_workers = max(0, int(args.num_workers))
    b_sampler = Stratified16TaskBatchSampler(train_ds, batch_size=args.episodes_per_step, seed=args.seed + start_step)
    loader = DataLoader(
        train_ds,
        batch_sampler=b_sampler,
        num_workers=eff_workers,
        collate_fn=collate_episode_list,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(eff_workers > 0),
        prefetch_factor=(2 if eff_workers > 0 else None),
    )
    batch_iter = iter(infinite_episode_iterator(loader))

    def _export_dg_payload(use_ema: bool) -> Dict[str, Any]:
        backups: Dict[str, torch.Tensor] = {}
        pmap = dict(model.named_parameters())
        try:
            if use_ema:
                for n, ev in ema.shadow.items():
                    if n in pmap:
                        backups[n] = pmap[n].data.clone()
                        pmap[n].data.copy_(ev.to(device=pmap[n].device, dtype=pmap[n].dtype))
            return {
                "deltaground_config": dg_cfg.to_dict(),
                "template_vocab": list(train_ds.template_vocab),
                "state_stats": state_stats_dict,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
        finally:
            for n, orig in backups.items():
                pmap[n].data.copy_(orig)

    def _save_dg(step_done: int, is_periodic: bool = False) -> None:
        online_dg = _export_dg_payload(use_ema=False)
        ema_dg = _export_dg_payload(use_ema=True)
        full_p = {
            "step": step_done,
            "best_loss": best_loss,
            "deltaground": online_dg,
            "ema_deltaground": ema_dg,
            "ema_shadow": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        atomic_torch_save(full_p, latest_ckpt_path)
        atomic_torch_save({"step": step_done, "ema_deltaground": ema_dg}, exp_dir / "latest_ema.pt")
        atomic_torch_save({"step": step_done, "deltaground": online_dg}, exp_dir / "latest_online.pt")
        if is_periodic and args.keep_every > 0 and (step_done % args.keep_every == 0):
            atomic_torch_save({"step": step_done, "ema_deltaground": ema_dg}, exp_dir / f"step_{step_done:06d}_ema.pt")
            atomic_torch_save({"step": step_done, "deltaground": online_dg}, exp_dir / f"step_{step_done:06d}_online.pt")

    model.train()
    running: Dict[str, float] = {}
    running_n = 0
    t_log = time.perf_counter()

    for step in range(start_step, args.max_steps):
        lr_mult = cosine_lr_multiplier(step, args.max_steps, args.warmup_steps)
        for grp, b_lr in zip(optimizer.param_groups, base_lrs):
            grp["lr"] = b_lr * lr_mult

        optimizer.zero_grad(set_to_none=True)
        step_metrics: Dict[str, float] = {}
        tot_episodes = max(1, args.grad_accum_steps) * max(1, args.episodes_per_step)

        for _ in range(max(1, args.grad_accum_steps)):
            ep_batch = next(batch_iter)
            # Concatenate all episodes in sub-batch into one batched loss after per-episode `O(W)` GDN scan!
            hop_lists: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(dg_cfg.num_hops)]
            tgt_cat: Dict[str, List[torch.Tensor]] = {
                k: []
                for k in (
                    "template_id",
                    "xy1_m11",
                    "has_xy1",
                    "xy2_m11",
                    "has_xy2",
                    "completed_count",
                    "remaining_count",
                    "stage_id",
                    "color_id",
                    "ordinal_id",
                )
            }
            for ep in ep_batch:
                ep_hops = model.forward_episode(ep)
                for hi, hp in enumerate(ep_hops):
                    hop_lists[hi].append(hp)
                for tk in tgt_cat:
                    tgt_cat[tk].append(ep[tk])

            merged_hops: List[Dict[str, torch.Tensor]] = [
                {pk: torch.cat([ d[pk] for d in h_list ], dim=0) for pk in h_list[0]}
                for h_list in hop_lists
            ]
            merged_tgts = {tk: torch.cat(t_list, dim=0) for tk, t_list in tgt_cat.items()}
            loss_dict = model.compute_deep_supervision_loss(merged_hops, merged_tgts)
            loss = loss_dict["loss"] / float(max(1, args.grad_accum_steps))
            loss.backward()

            for k, v in loss_dict.items():
                val = float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
                step_metrics[k] = step_metrics.get(k, 0.0) + val / float(max(1, args.grad_accum_steps))

        gn_mem, _ = optimizer.clip_and_step(grad_clip=args.grad_clip)
        ema.update(model, master_override=optimizer.master_named_tensors())
        step_done = step + 1
        running_n += 1
        step_metrics["grad_norm"] = gn_mem
        for k, v in step_metrics.items():
            running[k] = running.get(k, 0.0) + v

        if step_done % args.log_every == 0 or step_done == args.max_steps:
            dt = max(1e-6, time.perf_counter() - t_log)
            it_s = running_n / dt
            avg = {k: v / float(running_n) for k, v in running.items()}
            vram_gb = torch.cuda.max_memory_allocated(dev) / (1024**3) if dev.type == "cuda" else 0.0
            g_loop = [round(float(torch.tanh(x).item()), 3) for x in model.coda.gamma_loop.detach()[:2]]
            rec = {"step": step_done, "it_per_sec": round(it_s, 3), "gamma_loop": g_loop, **{k: round(v, 5) for k, v in avg.items()}}
            print(
                f"[DG Step {step_done:05d}/{args.max_steps}] loss={avg.get('loss', 0.0):.4f} "
                f"acc(h1->h3)={avg.get('tmpl_acc_h1', 0.0)*100:.1f}%->{avg.get('tmpl_acc_h3', 0.0)*100:.1f}% "
                f"px(h1->h3)={avg.get('px_err_h1', 0.0):.1f}->{avg.get('px_err_h3', 0.0):.1f}px "
                f"g_loop={g_loop} ({it_s:.2f} it/s, VRAM={vram_gb:.2f}GB)",
                flush=True,
            )
            with metrics_jsonl.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec) + "\n")
            if avg.get("loss", float("inf")) < best_loss:
                best_loss = avg["loss"]
            running.clear()
            running_n = 0
            t_log = time.perf_counter()

        if step_done % args.save_every == 0 or step_done == args.max_steps:
            _save_dg(step_done, is_periodic=True)
            print(f"[DG Ckpt] Saved step {step_done} -> {latest_ckpt_path}", flush=True)

        elapsed_min = (time.perf_counter() - wall_start) / 60.0
        if elapsed_min >= args.max_wall_minutes and step_done < args.max_steps:
            _save_dg(step_done, is_periodic=True)
            print(f"[DG Walltime Guard] Saved cleanly at step {step_done}/{args.max_steps}", flush=True)
            return

    print(f"[DeltaGround Done] {args.max_steps} steps in {(time.perf_counter() - wall_start)/60.0:.2f} min -> {exp_dir}", flush=True)


def run_deltaground_vlm_training(args: argparse.Namespace) -> None:
    """Train `P2-M9: DeltaGround-VLM` on shared `PaliGemma-2B` with strict `ep 0..89 / ep 90..99` unseen val (`#334`)."""
    from pi05 import load_pi05_checkpoint
    from pi05_mem.deltaground_vlm import DeltaGroundVLMConfig, DeltaGroundVLMSystem

    wall_start = time.perf_counter()
    set_all_seeds(args.seed)
    dev = torch.device(args.device)
    tasks_use = parse_task_filter(args.tasks)

    # Reuse saved 133-template vocab if present (`0.1s`), else scan full 100-ep closed grammar once!
    v2_vocab_ckpt = args.output_dir / "p2_m6_deltaground_v2_on100" / "step_001000_online.pt"
    if v2_vocab_ckpt.exists():
        raw_v2 = torch.load(v2_vocab_ckpt, map_location="cpu", weights_only=False)
        p_v2 = raw_v2.get("ema_deltaground") or raw_v2.get("deltaground") or raw_v2
        full_vocab = list(p_v2["template_vocab"])
        del raw_v2, p_v2
    else:
        full_vocab_ds = DeltaGroundEpisodeDataset(
            cache_dir=args.cache_dir,
            tasks=tasks_use,
            max_steps_per_episode=1,
            online_subgoal_prob=1.0,
            distractor_prob=0.0,
            seed=args.seed,
        )
        full_vocab = list(full_vocab_ds.template_vocab)
        del full_vocab_ds

    v5_adv_on = bool(args.dg_v5_advect_mode)
    v4_pyr_on = bool(args.dg_v4_pyramid_mode or v5_adv_on)
    v3_hr16_on = bool(args.dg_v3_hr16_mode)
    v2d_on = bool(args.dg_v2d_mode or v3_hr16_on or v4_pyr_on)
    v2c_on = bool(args.dg_v2c_mode or v2d_on)
    v2_on = bool(args.dg_v2_mode or v2c_on)
    if v4_pyr_on and int(args.dg_reveal_keyframes) == 6:
        eff_rev_k = 16
    elif v3_hr16_on and int(args.dg_reveal_keyframes) == 6:
        eff_rev_k = 9
    elif v2d_on and int(args.dg_reveal_keyframes) == 6:
        eff_rev_k = 14
    elif v2_on and int(args.dg_reveal_keyframes) == 6:
        eff_rev_k = 12
    else:
        eff_rev_k = int(args.dg_reveal_keyframes)
    if int(args.dg_reveal_tokens_per_frame) > 0:
        eff_rev_tok = int(args.dg_reveal_tokens_per_frame)
    else:
        eff_rev_tok = 256 if (v3_hr16_on and not v4_pyr_on) else 64
    if v4_pyr_on and int(args.dg_uniform_anchors) == 7:
        eff_uni_anchors = 8
    elif v3_hr16_on and int(args.dg_uniform_anchors) == 7:
        eff_uni_anchors = 3
    elif v2d_on and int(args.dg_uniform_anchors) == 7:
        eff_uni_anchors = 6
    else:
        eff_uni_anchors = int(args.dg_uniform_anchors)
    if (v3_hr16_on or v4_pyr_on) and int(args.dg_apex_peaks) <= 0:
        eff_apex_peaks = 4
    elif v2d_on and int(args.dg_apex_peaks) <= 0:
        eff_apex_peaks = 5
    else:
        eff_apex_peaks = int(args.dg_apex_peaks)
    eff_pyr_apex = eff_apex_peaks if v4_pyr_on else 0
    val_on_prob = float(args.online_subgoal_prob) if (v2c_on or float(args.online_subgoal_prob) < 1.0) else 1.0

    train_ds = DeltaGroundEpisodeDataset(
        cache_dir=args.cache_dir,
        hr_cache_dir=args.hr_cache_dir,
        tasks=tasks_use,
        max_steps_per_episode=args.chunks_per_episode,
        online_subgoal_prob=args.online_subgoal_prob,
        distractor_prob=args.distractor_prob,
        template_vocab=full_vocab,
        episode_range=(0, 90),
        stage_entry_focus_prob=(0.35 if v2c_on else (0.70 if v2_on else 0.0)),
        arm_neutral_sub_prob=(0.0 if v2c_on else (0.65 if v2_on else 0.0)),
        seed=args.seed,
    )
    val_ds = DeltaGroundEpisodeDataset(
        cache_dir=args.cache_dir,
        hr_cache_dir=args.hr_cache_dir,
        tasks=tasks_use,
        max_steps_per_episode=4,
        online_subgoal_prob=val_on_prob,
        distractor_prob=0.0,
        template_vocab=full_vocab,
        episode_range=(90, 100),
        shards_override=train_ds.shards,
        hr_shards_override=train_ds.hr_shards,
        seed=args.seed + 999,
    )
    val_entry_ds = (
        DeltaGroundEpisodeDataset(
            cache_dir=args.cache_dir,
            hr_cache_dir=args.hr_cache_dir,
            tasks=tasks_use,
            max_steps_per_episode=6,
            online_subgoal_prob=val_on_prob,
            distractor_prob=0.0,
            template_vocab=full_vocab,
            episode_range=(90, 100),
            eval_stage_entry_only=True,
            shards_override=train_ds.shards,
            hr_shards_override=train_ds.hr_shards,
            seed=args.seed + 1999,
        )
        if v2_on
        else None
    )

    pi05_base, _ = load_pi05_checkpoint(
        args.pi05_ckpt,
        device=dev,
        precision="bfloat16" if args.precision == "bfloat16" else "float32",
    )
    dgv_cfg = DeltaGroundVLMConfig(
        wrist_tokens_curr_pool=int(args.dg_wrist_tokens),
        reveal_keyframes=eff_rev_k,
        reveal_tokens_per_frame=eff_rev_tok,
        dg_lora_rank=int(args.dg_lora_rank),
        dg_lora_alpha=float(args.dg_lora_rank),
        dg_lora_dropout=float(args.dg_lora_dropout),
        dg_lora_include_mlp=(not bool(args.dg_no_mlp_lora)),
        num_templates=len(full_vocab),
        v2_mode=v2_on,
        v2_uniform_anchors=eff_uni_anchors,
        v2_keep_exec_clock=v2c_on,
        v2_raw_siglip_reveal=v2c_on,
        v2_apex_peaks=eff_apex_peaks,
        v2_hr16_sincos=bool(v3_hr16_on or v4_pyr_on),
        v2_pyramid_apex_hr16=eff_pyr_apex,
        xy_loss_weight=float(args.dg_xy_loss_weight),
        v2_reveal_prominence_tau=(int(args.dg_prominence_tau) if v5_adv_on else 0),
        v2_advect_reveal=v5_adv_on,
    )
    system = DeltaGroundVLMSystem(
        dgv_cfg,
        pi05_base,
        template_vocab=full_vocab,
        enable_grad_ckpt=not bool(args.no_grad_ckpt),
    )
    if args.init_dg_vlm_ckpt is not None:
        init_dg_raw = torch.load(args.init_dg_vlm_ckpt, map_location="cpu", weights_only=False)
        init_dg_pay = init_dg_raw.get("ema_deltaground_vlm") or init_dg_raw.get("deltaground_vlm") or init_dg_raw
        system.load_payload(init_dg_pay, strict=True, allow_slot_interp=True)
        print(
            f"[DG-VLM WarmStart] Loaded {args.init_dg_vlm_ckpt} (K_new={eff_rev_k}, apex={eff_apex_peaks}, uni={eff_uni_anchors})",
            flush=True,
        )
        del init_dg_raw, init_dg_pay

    gate_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []
    lora_params: List[nn.Parameter] = []
    for name, p in system.named_parameters():
        if not p.requires_grad:
            continue
        if "dg_lora_" in name:
            lora_params.append(p)
        elif any(k in name for k in ("A_log", "dt_bias", "salience_gate", "boundary_gate", "wrist_view_emb", "vlm_hist_slot_emb", "log_prox")):
            gate_params.append(p)
        else:
            head_params.append(p)

    param_groups = [
        {"name": "gates", "params": gate_params, "lr": float(args.lr_gate), "weight_decay": 0.0},
        {"name": "memory", "params": head_params, "lr": float(args.lr_mem), "weight_decay": float(args.weight_decay)},
        {"name": "dg_vlm_lora", "params": lora_params, "lr": float(args.lr_policy), "weight_decay": 0.01 if v2_on else 0.0},
    ]
    base_lrs = [float(g["lr"]) for g in param_groups]
    optimizer = FP32MasterAdamW(system, param_groups, betas=(0.9, 0.95), eps=1e-8)
    ema = TrainableFP32EMA(system, decay=min(float(args.ema_decay), 0.985), warmup_ema=(args.init_dg_vlm_ckpt is None))

    exp_dir = args.output_dir / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = exp_dir / "latest.pt"
    metrics_jsonl = exp_dir / "metrics.jsonl"
    from robomme_eval.interface import MME_VLA_NORM_STATS

    ns_file = Path(args.cache_dir) / "norm_stats.json"
    state_stats_dict = json.loads(ns_file.read_text()).get("norm_stats", MME_VLA_NORM_STATS) if ns_file.exists() else MME_VLA_NORM_STATS

    start_step = 0
    best_unseen_score = -1e9
    if args.resume == "auto" and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        system.load_payload(ckpt["deltaground_vlm"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"], device=dev)
        if "ema_shadow" in ckpt:
            ema.load_state_dict(ckpt["ema_shadow"], device=dev)
        start_step = int(ckpt["step"])
        best_unseen_score = float(ckpt.get("best_unseen_score", -1e9))
        print(f"[DG-VLM Resume] step={start_step}/{args.max_steps} best_unseen_score={best_unseen_score:.4f}", flush=True)

    n_train = sum(p.numel() for p in system.parameters() if p.requires_grad)
    print(
        f"[DG-VLM Train] train_eps={len(train_ds)} unseen_val_eps={len(val_ds)} templates={len(full_vocab)} "
        f"dg_lora_r={dgv_cfg.dg_lora_rank} trainable={n_train:,} steps={start_step}->{args.max_steps} (HAS_FLA={HAS_FLA})",
        flush=True,
    )

    eff_workers = max(0, int(args.num_workers))
    b_sampler = Stratified16TaskBatchSampler(train_ds, batch_size=args.episodes_per_step, seed=args.seed + start_step)
    loader = DataLoader(
        train_ds,
        batch_sampler=b_sampler,
        num_workers=eff_workers,
        collate_fn=collate_episode_list,
        pin_memory=(dev.type == "cuda"),
        persistent_workers=(eff_workers > 0),
        prefetch_factor=(2 if eff_workers > 0 else None),
    )
    batch_iter = iter(infinite_episode_iterator(loader))

    def _export_dgv(use_ema: bool) -> Dict[str, Any]:
        backups: Dict[str, torch.Tensor] = {}
        pmap = dict(system.named_parameters())
        try:
            if use_ema:
                for n, ev in ema.shadow.items():
                    if n in pmap:
                        backups[n] = pmap[n].data.clone()
                        pmap[n].data.copy_(ev.to(device=pmap[n].device, dtype=pmap[n].dtype))
            return system.export_payload(state_stats=state_stats_dict)
        finally:
            for n, orig in backups.items():
                pmap[n].data.copy_(orig)

    @torch.no_grad()
    def _eval_unseen(use_ema: bool = True) -> Dict[str, Any]:
        from pi05_mem.deltaground import apply_peak_window_to_preds
        from analysis.subgoal_grounding import norm_xy_to_pixel_xy

        backups: Dict[str, torch.Tensor] = {}
        pmap = dict(system.named_parameters())
        was_train = system.training
        try:
            if use_ema:
                for n, ev in ema.shadow.items():
                    if n in pmap:
                        backups[n] = pmap[n].data.clone()
                        pmap[n].data.copy_(ev.to(device=pmap[n].device, dtype=pmap[n].dtype))
            system.eval()
            tot_n = 0
            tot_acc = 0.0
            tot_px_n = 0
            tot_px = 0.0
            suite_acc: Dict[str, List[float]] = {}
            suite_px: Dict[str, List[float]] = {}
            bs_val = 6
            for i0 in range(0, len(val_ds), bs_val):
                eps_v = [val_ds[j] for j in range(i0, min(len(val_ds), i0 + bs_val))]
                preds_v, tgts_v = system.forward_episodes(eps_v)
                ld_v = system.compute_loss(preds_v, tgts_v)
                n_c = int(tgts_v["template_id"].numel())
                tot_n += n_c
                tot_acc += float(ld_v["tmpl_acc"].item()) * n_c
                cur_off = 0
                pred_ids = preds_v["template_logits"].argmax(dim=-1)
                fin_pv = apply_peak_window_to_preds(preds_v, s_side=system.head.coda.front_side, radius=2)
                px_p = norm_xy_to_pixel_xy(fin_pv["xy_pri"], align_corners=False)
                px_t = norm_xy_to_pixel_xy(tgts_v["xy1_m11"].float(), align_corners=False)
                px_d = torch.linalg.vector_norm(px_p - px_t, ord=2, dim=-1)
                hx_all = tgts_v["has_xy1"].bool()
                if bool(hx_all.any()):
                    tot_px_n += int(hx_all.sum().item())
                    tot_px += float(px_d[hx_all].sum().item())
                for ep_item in eps_v:
                    m_e = int(ep_item["chunk_step_indices"].numel())
                    st_name = str(ep_item["suite"])
                    sl = slice(cur_off, cur_off + m_e)
                    cur_off += m_e
                    eq = (pred_ids[sl] == tgts_v["template_id"][sl].long()).float().tolist()
                    suite_acc.setdefault(st_name, []).extend(eq)
                    hx = tgts_v["has_xy1"][sl].bool()
                    if bool(hx.any()):
                        suite_px.setdefault(st_name, []).extend(px_d[sl][hx].float().tolist())

            out_ev: Dict[str, Any] = {
                "unseen_acc": tot_acc / max(1, tot_n),
                "unseen_px": tot_px / max(1, tot_px_n),
                "unseen_perm_acc": float(np.mean(suite_acc.get("Permanence", [0.0]))),
                "unseen_perm_px": float(np.mean(suite_px.get("Permanence", [0.0]))),
                "unseen_ref_acc": float(np.mean(suite_acc.get("Reference", [0.0]))),
                "unseen_ref_px": float(np.mean(suite_px.get("Reference", [0.0]))),
            }

            if val_entry_ds is not None:
                twin_p = build_occlusion_twin_template_pairs(full_vocab)
                ent_acc_l: List[float] = []
                ent_px_l: List[float] = []
                step0_px_l: List[float] = []
                task_ent_px: Dict[str, List[float]] = {}
                for i0 in range(0, len(val_entry_ds), bs_val):
                    eps_e = [val_entry_ds[j] for j in range(i0, min(len(val_entry_ds), i0 + bs_val))]
                    preds_e, tgts_e = system.forward_episodes(eps_e)
                    t_log_e = (
                        fold_occlusion_twin_logits(preds_e["template_logits"], twin_p)
                        if twin_p
                        else preds_e["template_logits"]
                    )
                    tgt_fold = tgts_e["template_id"].long().clone()
                    for u_i, g_i in twin_p:
                        tgt_fold = torch.where(tgt_fold == u_i, torch.full_like(tgt_fold, g_i), tgt_fold)
                    pred_ids_e = t_log_e.argmax(dim=-1)
                    fin_pe = apply_peak_window_to_preds(preds_e, s_side=system.head.coda.front_side, radius=2)
                    px_pe = norm_xy_to_pixel_xy(fin_pe["xy_pri"], align_corners=False)
                    px_te = norm_xy_to_pixel_xy(tgts_e["xy1_m11"].float(), align_corners=False)
                    px_de = torch.linalg.vector_norm(px_pe - px_te, ord=2, dim=-1)
                    cur_e = 0
                    for ep_it in eps_e:
                        m_e = int(ep_it["chunk_step_indices"].numel())
                        t_name = str(ep_it["task_name"])
                        sl = slice(cur_e, cur_e + m_e)
                        cur_e += m_e
                        ent_acc_l.extend((pred_ids_e[sl] == tgt_fold[sl]).float().tolist())
                        hx_e = tgts_e["has_xy1"][sl].bool()
                        if bool(hx_e.any()):
                            d_vals = px_de[sl][hx_e].float().tolist()
                            ent_px_l.extend(d_vals)
                            task_ent_px.setdefault(t_name, []).extend(d_vals)
                        if m_e > 0 and bool(tgts_e["has_xy1"][sl.start].item()):
                            step0_px_l.append(float(px_de[sl.start].item()))

                out_ev["unseen_entry_acc"] = float(np.mean(ent_acc_l)) if ent_acc_l else 0.0
                out_ev["unseen_entry_px"] = float(np.mean(ent_px_l)) if ent_px_l else 99.0
                out_ev["unseen_entry_hit18"] = float(np.mean([1.0 if d < 18.0 else 0.0 for d in ent_px_l])) if ent_px_l else 0.0
                out_ev["unseen_step0_px"] = float(np.mean(step0_px_l)) if step0_px_l else 99.0
                out_ev["unseen_step0_hit18"] = float(np.mean([1.0 if d < 18.0 else 0.0 for d in step0_px_l])) if step0_px_l else 0.0
                out_ev["per_task_entry"] = {
                    tn: {
                        "px": round(float(np.mean(vs)), 2),
                        "hit18": round(float(np.mean([1.0 if d < 18.0 else 0.0 for d in vs])) * 100.0, 1),
                    }
                    for tn, vs in sorted(task_ent_px.items())
                }
            return out_ev
        finally:
            for n, orig in backups.items():
                pmap[n].data.copy_(orig)
            if was_train:
                system.train()

    def _maybe_write_m7b_bundle(ema_dgv: Dict[str, Any], step_done: int) -> None:
        if args.bundle_into_m7b_ckpt is None or not Path(args.bundle_into_m7b_ckpt).exists():
            return
        out_b = args.bundled_output_path or (args.output_dir / "p2_m7b_s300_plus_m9_dgvlm" / "best_bundled.pt")
        m7b_raw = torch.load(args.bundle_into_m7b_ckpt, map_location="cpu", weights_only=False)
        m7b_delta = m7b_raw.get("ema_delta") or m7b_raw.get("model_delta") or m7b_raw
        atomic_torch_save(
            {"step": step_done, "ema_delta": m7b_delta, "ema_deltaground_vlm": ema_dgv},
            Path(out_b),
        )
        print(f"[DG-VLM Bundle] Exported 1-file P2-M7B+P2-M9 bundle -> {out_b}", flush=True)

    def _save_dgv(step_done: int, is_periodic: bool = False, is_best: bool = False) -> None:
        on_p = _export_dgv(use_ema=False)
        ema_p = _export_dgv(use_ema=True)
        atomic_torch_save(
            {
                "step": step_done,
                "best_unseen_score": best_unseen_score,
                "deltaground_vlm": on_p,
                "ema_deltaground_vlm": ema_p,
                "ema_shadow": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            latest_ckpt_path,
        )
        atomic_torch_save({"step": step_done, "ema_deltaground_vlm": ema_p}, exp_dir / "latest_ema.pt")
        atomic_torch_save({"step": step_done, "deltaground_vlm": on_p}, exp_dir / "latest_online.pt")
        if is_best:
            atomic_torch_save({"step": step_done, "ema_deltaground_vlm": ema_p}, exp_dir / "best_unseen_ema.pt")
            atomic_torch_save({"step": step_done, "deltaground_vlm": on_p}, exp_dir / "best_unseen_online.pt")
            _maybe_write_m7b_bundle(ema_p, step_done)
        if is_periodic and args.keep_every > 0 and (step_done % args.keep_every == 0):
            atomic_torch_save({"step": step_done, "ema_deltaground_vlm": ema_p}, exp_dir / f"step_{step_done:06d}_ema.pt")
            atomic_torch_save({"step": step_done, "deltaground_vlm": on_p}, exp_dir / f"step_{step_done:06d}_online.pt")

    if args.init_dg_vlm_ckpt is not None and start_step == 0:
        ev0 = _eval_unseen(use_ema=True)
        if v2_on and "unseen_entry_hit18" in ev0:
            score0 = (
                0.45 * float(ev0["unseen_entry_hit18"])
                + 0.45 * float(ev0["unseen_step0_hit18"])
                - 0.008 * float(ev0["unseen_entry_px"])
                + 0.25 * float(ev0["unseen_entry_acc"])
                + (0.35 * float(ev0["unseen_acc"]) if v2c_on else 0.0)
            )
            best_unseen_score = score0
            pt0 = " ".join(f"{k}:{v['hit18']:.0f}%({v['px']:.1f}px)" for k, v in ev0.get("per_task_entry", {}).items())
            print(
                f"[DG-VLMv2 Unseen @ 0 (WarmStart)] STEP0: hit18={ev0['unseen_step0_hit18']*100:.1f}% px={ev0['unseen_step0_px']:.2f} | "
                f"ENTRY: hit18={ev0['unseen_entry_hit18']*100:.1f}% px={ev0['unseen_entry_px']:.2f} acc={ev0['unseen_entry_acc']*100:.1f}% | "
                f"TRAJ: acc={ev0['unseen_acc']*100:.1f}% px={ev0['unseen_px']:.2f}\n"
                f"  -> Tasks: {pt0}",
                flush=True,
            )
            _save_dgv(0, is_periodic=False, is_best=True)
        else:
            score0 = float(ev0["unseen_acc"]) - 0.004 * float(args.dg_xy_loss_weight) * float(ev0["unseen_px"])
            best_unseen_score = score0
            print(
                f"[DG-VLM Unseen ep90..99 @ 0 (WarmStart)] acc={ev0['unseen_acc']*100:.1f}% px={ev0['unseen_px']:.2f} | "
                f"Perm: acc={ev0['unseen_perm_acc']*100:.1f}% | Ref: acc={ev0['unseen_ref_acc']*100:.1f}%",
                flush=True,
            )
            _save_dgv(0, is_periodic=False, is_best=True)

    system.train()
    running: Dict[str, float] = {}
    running_n = 0
    t_log = time.perf_counter()

    for step in range(start_step, args.max_steps):
        lr_mult = cosine_lr_multiplier(step, args.max_steps, args.warmup_steps)
        for grp, b_lr in zip(optimizer.param_groups, base_lrs):
            grp["lr"] = b_lr * lr_mult

        optimizer.zero_grad(set_to_none=True)
        step_m: Dict[str, float] = {}
        ga = max(1, args.grad_accum_steps)
        for _ in range(ga):
            ep_batch = next(batch_iter)
            preds_b, tgts_b = system.forward_episodes(ep_batch)
            ld = system.compute_loss(preds_b, tgts_b)
            (ld["loss"] / float(ga)).backward()
            for k, v in ld.items():
                step_m[k] = step_m.get(k, 0.0) + float(v.detach().item()) / float(ga)

        gn_m, gn_p = optimizer.clip_and_step(grad_clip=args.grad_clip)
        ema.update(system, master_override=optimizer.master_named_tensors())
        step_done = step + 1
        running_n += 1
        step_m["gn_m"] = gn_m
        step_m["gn_p"] = gn_p
        for k, v in step_m.items():
            running[k] = running.get(k, 0.0) + v

        if step_done % args.log_every == 0 or step_done == args.max_steps:
            dt = max(1e-6, time.perf_counter() - t_log)
            it_s = running_n / dt
            avg = {k: v / float(running_n) for k, v in running.items()}
            vram_gb = torch.cuda.max_memory_allocated(dev) / (1024**3) if dev.type == "cuda" else 0.0
            print(
                f"[DG-VLM {step_done:05d}/{args.max_steps}] loss={avg['loss']:.4f} tmpl={avg['tmpl_loss']:.4f} "
                f"xy={avg['xy_loss']:.4f} acc={avg['tmpl_acc']*100:.1f}% px={avg['px_err']:.2f} "
                f"({it_s:.2f} it/s, VRAM={vram_gb:.2f}GB)",
                flush=True,
            )
            with metrics_jsonl.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps({"step": step_done, "it_s": round(it_s, 3), **{k: round(v, 5) for k, v in avg.items()}}) + "\n")
            running.clear()
            running_n = 0
            t_log = time.perf_counter()

        if (args.eval_every > 0 and step_done % args.eval_every == 0) or step_done == args.max_steps:
            ev = _eval_unseen(use_ema=True)
            if v2_on and "unseen_entry_hit18" in ev:
                score = (
                    0.45 * float(ev["unseen_entry_hit18"])
                    + 0.45 * float(ev["unseen_step0_hit18"])
                    - 0.008 * float(ev["unseen_entry_px"])
                    + 0.25 * float(ev["unseen_entry_acc"])
                    + (0.35 * float(ev["unseen_acc"]) if v2c_on else 0.0)
                )
            else:
                score = float(ev["unseen_acc"]) - 0.004 * float(args.dg_xy_loss_weight) * float(ev["unseen_px"])
            is_best = score > best_unseen_score
            if is_best:
                best_unseen_score = score
            if v2_on and "unseen_entry_hit18" in ev:
                pt_str = " ".join(
                    f"{k}:{v['hit18']:.0f}%({v['px']:.1f}px)" for k, v in ev.get("per_task_entry", {}).items()
                )
                print(
                    f"[DG-VLMv2 Unseen @ {step_done}] STEP0: hit18={ev['unseen_step0_hit18']*100:.1f}% px={ev['unseen_step0_px']:.2f} | "
                    f"ENTRY: hit18={ev['unseen_entry_hit18']*100:.1f}% px={ev['unseen_entry_px']:.2f} acc={ev['unseen_entry_acc']*100:.1f}% | "
                    f"TRAJ: acc={ev['unseen_acc']*100:.1f}% px={ev['unseen_px']:.2f} {'(NEW BEST!)' if is_best else ''}\n"
                    f"  -> Tasks: {pt_str}",
                    flush=True,
                )
            else:
                print(
                    f"[DG-VLM Unseen ep90..99 @ {step_done}] acc={ev['unseen_acc']*100:.1f}% px={ev['unseen_px']:.2f} | "
                    f"Perm: acc={ev['unseen_perm_acc']*100:.1f}% px={ev['unseen_perm_px']:.2f} | "
                    f"Ref: acc={ev['unseen_ref_acc']*100:.1f}% px={ev['unseen_ref_px']:.2f} "
                    f"{'(NEW BEST!)' if is_best else ''}",
                    flush=True,
                )
            with metrics_jsonl.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps({"step": step_done, "eval_unseen": True, **ev}) + "\n")
            _save_dgv(step_done, is_periodic=(step_done % args.save_every == 0 or step_done == args.max_steps), is_best=is_best)
        elif step_done % args.save_every == 0:
            _save_dgv(step_done, is_periodic=True, is_best=False)

        elapsed_min = (time.perf_counter() - wall_start) / 60.0
        if elapsed_min >= args.max_wall_minutes and step_done < args.max_steps:
            _save_dgv(step_done, is_periodic=True, is_best=False)
            print(f"[DG-VLM Walltime Guard] Saved cleanly at step {step_done}/{args.max_steps}", flush=True)
            return

    print(f"[DG-VLM Done] {args.max_steps} steps in {(time.perf_counter() - wall_start)/60.0:.2f} min -> {exp_dir}", flush=True)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.self_test:
        run_pi05_mem_self_tests(device_str=args.device)
        return
    if args.extract_hr256_cache:
        from pi05_mem.deltaground_vlm import extract_specialist_hr256_cache

        if args.hr_cache_dir is None:
            raise ValueError("--hr-cache-dir is required when --extract-hr256-cache is set")
        tasks_hr = parse_task_filter(args.tasks) or (
            "VideoUnmask",
            "VideoUnmaskSwap",
            "ButtonUnmask",
            "ButtonUnmaskSwap",
            "PickHighlight",
            "VideoRepick",
        )
        extract_specialist_hr256_cache(
            dataset_root=args.lerobot_root,
            base_cache_dir=args.cache_dir,
            hr_output_dir=args.hr_cache_dir,
            tasks=tasks_hr,
            device=str(args.device),
        )
        return
    if args.train_deltaground:
        run_deltaground_training(args)
        return
    if args.train_deltaground_vlm:
        run_deltaground_vlm_training(args)
        return

    wall_start = time.perf_counter()
    set_all_seeds(args.seed)
    dev = torch.device(args.device)

    # Auto-align `stats_variant` across Track-1 (`pi05_baseline`), Track-2 (`framesamp_modul`), & `symbolic` (`mme_vla`)!
    eff_stats_variant = args.stats_variant
    if "symbolic_grounded_subgoal" in str(args.pi05_ckpt) and eff_stats_variant == "pi05_baseline":
        eff_stats_variant = "mme_vla"
    elif (
        args.arch in ("p2_m4_deltamem_aligned", "p2_m5_hybrid_surprise_gdn")
        or "framesamp_modul" in str(args.pi05_ckpt)
    ) and eff_stats_variant == "pi05_baseline":
        eff_stats_variant = "framesamp_modul"

    bundled_dg_payload: Optional[Dict[str, Any]] = None
    bundled_dgv_payload: Optional[Dict[str, Any]] = None
    pred_subgoals_by_task: Optional[Dict[str, List[Dict[int, str]]]] = None
    eff_cont_vlm_cache_dir: Optional[Path] = args.cont_vlm_cache_dir
    if args.use_continuous_vlm_ground and not args.smoke_synthetic and args.deltaground_vlm_ckpt is not None:
        from pi05_mem.deltaground_vlm import precompute_continuous_vlm_grounding

        eff_cont_vlm_cache_dir = args.cont_vlm_cache_dir or (
            args.output_dir / f"cont_vlm_cache_{args.deltaground_vlm_ckpt.parent.name}_{args.deltaground_vlm_ckpt.stem}"
        )
        t_pre0 = time.perf_counter()
        print(
            f"[P2-M11] Loading/precomputing Continuous VLM Grounding from {args.deltaground_vlm_ckpt} -> {eff_cont_vlm_cache_dir}...",
            flush=True,
        )
        bundled_dgv_payload = precompute_continuous_vlm_grounding(
            args.deltaground_vlm_ckpt,
            args.pi05_ckpt,
            args.cache_dir,
            eff_cont_vlm_cache_dir,
            hr_cache_dir=args.hr_cache_dir,
            device=dev,
        )
        print(f"[P2-M11] Ready in {time.perf_counter() - t_pre0:.2f}s (max_token_len=64, 0 ASCII tokens!)", flush=True)
    elif args.arch == "p2_m7_grounded_hybrid" and not args.smoke_synthetic and args.deltaground_vlm_ckpt is not None:
        from pi05_mem.deltaground_vlm import precompute_deltaground_vlm_subgoals

        sg_cache_p = args.subgoal_cache_file or (
            args.output_dir / f"subgoal_vlm_cache_{args.deltaground_vlm_ckpt.parent.name}_{args.deltaground_vlm_ckpt.stem}.pt"
        )
        t_pre0 = time.perf_counter()
        print(f"[P2-M7C] Loading/precomputing DeltaGround-VLM subgoals from {args.deltaground_vlm_ckpt} -> {sg_cache_p}...", flush=True)
        pred_subgoals_by_task, bundled_dgv_payload = precompute_deltaground_vlm_subgoals(
            args.deltaground_vlm_ckpt,
            args.pi05_ckpt,
            args.cache_dir,
            output_path=sg_cache_p,
            tasks=parse_task_filter(args.tasks),
            device=dev,
            min_template_confidence=float(args.subgoal_min_conf) if args.subgoal_selective_filter else 0.0,
            abstain_ordinal_counting=bool(args.subgoal_selective_filter),
            abstain_place_order=(not bool(args.subgoal_keep_place_order)),
        )
        print(f"[P2-M7C] Ready in {time.perf_counter() - t_pre0:.2f}s ({len(pred_subgoals_by_task)} tasks)", flush=True)
    elif args.arch == "p2_m7_grounded_hybrid" and not args.smoke_synthetic and args.deltaground_ckpt is not None:
        sg_cache_p = args.subgoal_cache_file or (args.output_dir / f"subgoal_cache_{args.deltaground_ckpt.parent.name}_{args.deltaground_ckpt.stem}.pt")
        t_pre0 = time.perf_counter()
        print(f"[P2-M7] Loading/precomputing DeltaGround subgoals from {args.deltaground_ckpt} -> {sg_cache_p}...", flush=True)
        pred_subgoals_by_task, bundled_dg_payload = precompute_deltaground_subgoals(
            args.deltaground_ckpt,
            args.cache_dir,
            output_path=sg_cache_p,
            tasks=parse_task_filter(args.tasks),
            device=dev,
            num_hops=args.deltaground_hops,
        )
        print(f"[P2-M7] Ready in {time.perf_counter() - t_pre0:.2f}s ({len(pred_subgoals_by_task)} tasks)", flush=True)

    if args.smoke_synthetic:
        extra_kw: Dict[str, Any] = {}
        if args.modulator_mode is not None:
            extra_kw["modulator_mode"] = args.modulator_mode
        if args.hybrid_anchor_ratio is not None:
            extra_kw["hybrid_anchor_ratio"] = float(args.hybrid_anchor_ratio)
        cfg = Pi05MemConfig.tiny_test_preset(
            args.arch,  # type: ignore[arg-type]
            train_stage=args.stage,  # type: ignore[arg-type]
            use_vlm_lora=args.use_vlm_lora,
            use_expert_lora=args.use_expert_lora,
            expert_lora_rank=args.expert_lora_rank,
            expert_lora_alpha=float(args.expert_lora_rank),
            expert_lora_dropout=args.expert_lora_dropout,
            freeze_dense_expert_in_stage_b=args.freeze_dense_expert_in_stage_b,
            hybrid_use_landmarks=args.hybrid_use_landmarks,
            timescale_mode=args.timescale_mode,  # type: ignore[arg-type]
            landmark_encoder=args.landmark_encoder,  # type: ignore[arg-type]
            use_self_subgoal_tokens=args.use_self_subgoal_tokens,
            use_continuous_vlm_ground=args.use_continuous_vlm_ground,
            cont_vlm_jitter_cells=args.cont_vlm_jitter_cells,
            use_reactive_precision_mode=args.reactive_precision_mode,
            use_residual_hop2=bool(args.reactive_precision_mode or (args.stage == "stage_c")),
            mem_cfg_dropout=args.mem_cfg_dropout,
            **extra_kw,
        )
        policy = DeltaMemPi05Policy(cfg, device=dev, precision=args.precision)
        train_ds: Any = SyntheticPi05EpisodeDataset(cfg, num_episodes=16, seed=args.seed)
        val_fm_ds: Optional[RealisticCrossSessionEpisodeDataset] = None
    else:
        cfg_kw: Dict[str, Any] = dict(
            arch=args.arch,
            train_stage=args.stage,
            use_vlm_lora=args.use_vlm_lora,
            lora_rank=args.lora_rank,
            lora_alpha=float(args.lora_rank),
            use_expert_lora=args.use_expert_lora,
            expert_lora_rank=args.expert_lora_rank,
            expert_lora_alpha=float(args.expert_lora_rank),
            expert_lora_dropout=args.expert_lora_dropout,
            freeze_dense_expert_in_stage_b=args.freeze_dense_expert_in_stage_b,
            timescale_mode=args.timescale_mode,
            landmark_encoder=args.landmark_encoder,
            num_landmarks=args.num_landmarks,
            hybrid_num_keyframes=args.hybrid_num_keyframes,
            hybrid_canvas_tokens=args.hybrid_canvas_tokens,
            hybrid_use_landmarks=args.hybrid_use_landmarks,
            use_self_subgoal_tokens=args.use_self_subgoal_tokens,
            use_continuous_vlm_ground=args.use_continuous_vlm_ground,
            cont_vlm_jitter_cells=args.cont_vlm_jitter_cells,
            use_reactive_precision_mode=args.reactive_precision_mode,
            use_residual_hop2=bool(args.reactive_precision_mode or (args.stage == "stage_c")),
            mem_cfg_dropout=args.mem_cfg_dropout,
            stage_b_aux_scale=args.stage_b_aux_scale,
            fm_loss_scale=args.fm_loss_scale,
            loss_on_valid_8d_only=args.loss_on_valid_8d_only,
            graft_gate_init=args.graft_gate_init,
            subgoal_dg_prob=args.subgoal_dg_prob,
            subgoal_oracle_prob=args.subgoal_oracle_prob,
            subgoal_jitter_px=args.subgoal_jitter_px,
            vlm_prefix_surprise_keyframes=args.vlm_prefix_keyframes,
            vlm_prefix_tokens_per_frame=args.vlm_prefix_tokens_per_frame,
            vlm_prefix_demo_quota=args.vlm_prefix_demo_quota,
            vlm_prefix_early_quota=args.vlm_prefix_early_quota,
            vlm_prefix_drop_prob=args.vlm_prefix_drop_prob,
        )
        if args.modulator_mode is not None:
            cfg_kw["modulator_mode"] = args.modulator_mode
        elif args.arch == "p2_m7_grounded_hybrid":
            cfg_kw["modulator_mode"] = (
                "mme_vla_exact" if "framesamp_modul" in str(args.pi05_ckpt) else "grafted_residual"
            )
        if args.hybrid_anchor_ratio is not None:
            cfg_kw["hybrid_anchor_ratio"] = float(args.hybrid_anchor_ratio)
        if args.use_continuous_vlm_ground:
            cfg_kw["max_token_len"] = 64
        cfg = Pi05MemConfig(**cfg_kw)  # type: ignore[arg-type]
        policy = DeltaMemPi05Policy(
            cfg,
            pi05_ckpt_dir=args.pi05_ckpt,
            device=dev,
            precision=args.precision,
        )
        dist_p = args.distractor_prob if args.stage == "stage_a" else 0.05
        use_vlm_sg = (cfg.arch == "p2_m7_grounded_hybrid") and (not bool(cfg.use_continuous_vlm_ground))
        tr_ep_idx: Optional[List[int]] = None
        if args.train_ep_range:
            s_tr, e_tr = [int(x.strip()) for x in args.train_ep_range.split(",")]
            tr_ep_idx = list(range(s_tr, e_tr))
        train_ds = RealisticCrossSessionEpisodeDataset(
            cache_dir=args.cache_dir,
            tasks=parse_task_filter(args.tasks),
            episode_indices_per_task=tr_ep_idx,
            stats_variant=eff_stats_variant,  # type: ignore[arg-type]
            max_chunks_per_episode=args.chunks_per_episode,
            defer_siglip_head_to_device=True,
            distractor_prob=dist_p,
            pred_subgoals_by_task=pred_subgoals_by_task if use_vlm_sg else None,
            cont_vlm_cache_dir=eff_cont_vlm_cache_dir if bool(cfg.use_continuous_vlm_ground) else None,
            reactive_precision_sampling=bool(args.reactive_precision_mode),
            vlm_max_token_len=int(cfg.max_token_len),
            subgoal_dg_prob=float(args.subgoal_dg_prob) if use_vlm_sg else 0.0,
            subgoal_oracle_prob=float(args.subgoal_oracle_prob) if use_vlm_sg else 0.0,
            subgoal_jitter_px=int(args.subgoal_jitter_px),
            seed=args.seed,
        )
        val_fm_ds: Optional[RealisticCrossSessionEpisodeDataset] = None
        if args.val_ep_range:
            s_vl, e_vl = [int(x.strip()) for x in args.val_ep_range.split(",")]
            val_fm_ds = RealisticCrossSessionEpisodeDataset(
                cache_dir=args.cache_dir,
                tasks=parse_task_filter(args.tasks),
                episode_indices_per_task=list(range(s_vl, e_vl)),
                stats_variant=eff_stats_variant,  # type: ignore[arg-type]
                max_chunks_per_episode=8,
                defer_siglip_head_to_device=True,
                distractor_prob=0.0,
                reactive_precision_sampling=bool(args.reactive_precision_mode),
                vlm_max_token_len=int(cfg.max_token_len),
                seed=args.seed + 999,
            )

    exp_dir = args.output_dir / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = exp_dir / "latest.pt"
    metrics_jsonl = exp_dir / "metrics.jsonl"

    if args.graft_history_ckpt is not None and not (args.resume == "auto" and latest_ckpt_path.exists()):
        n_graft = policy.graft_history_from_checkpoint(args.graft_history_ckpt)
        print(f"[Graft] Stream-loaded {n_graft} HistoryBlock/PerceptualMem tensors from {args.graft_history_ckpt}", flush=True)

    frozen_base_vlm_lora: Optional[Dict[str, torch.Tensor]] = None
    if args.init_delta_ckpt is not None and args.freeze_all_except_vlm_lora:
        init_for_lora = torch.load(args.init_delta_ckpt, map_location="cpu", weights_only=False)
        id_d = init_for_lora.get("ema_delta") or init_for_lora.get("model_delta") or init_for_lora
        frozen_base_vlm_lora = {
            k: v.detach().cpu().clone()
            for k, v in id_d.get("pi05_delta", {}).items()
            if "lora_" in k and ".vlm." in k
        }
        del init_for_lora, id_d

    # Optional warm-start from prior stage (`A0 -> B1 -> A1 -> B2 -> C` or `v5 -> P2-M7`) before resume check
    if args.init_delta_ckpt is not None and not (args.resume == "auto" and latest_ckpt_path.exists()):
        print(f"[Init] Loading prior stage delta weights from {args.init_delta_ckpt} (skip_lora={args.skip_init_lora})...", flush=True)
        init_raw = torch.load(args.init_delta_ckpt, map_location="cpu", weights_only=False)
        init_delta = init_raw.get("ema_delta") or init_raw.get("model_delta") or init_raw
        policy.load_delta_state_dict(init_delta, strict_mem=False, skip_lora=bool(args.skip_init_lora))

    def _freeze_to_vlm_lora_only() -> Dict[str, int]:
        allow_exp = bool(args.use_expert_lora)
        allow_cont_br = bool(args.use_continuous_vlm_ground)
        allow_reac_h2 = bool(args.reactive_precision_mode)
        for name, p in policy.named_parameters():
            is_lora = ("lora_" in name) and ((".vlm." in name) or (allow_exp and ".expert." in name))
            is_cont_br = allow_cont_br and ("continuous_vlm_bridge." in name)
            is_reac_h2 = allow_reac_h2 and ("hop2_refiner." in name)
            p.requires_grad = bool(is_lora or is_cont_br or is_reac_h2)
        tr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in policy.parameters())
        return {"trainable_params": tr, "total_params": tot}

    freeze_mem_b = args.stage_b_mem_lr_scale <= 0.0
    use_gckpt = (args.stage == "stage_b") and bool(args.use_vlm_lora) and not bool(args.no_grad_ckpt)
    p_counts = policy.configure_stage(
        args.stage,
        freeze_mem_in_stage_b=freeze_mem_b,
        freeze_hooks_in_stage_a=args.freeze_hooks_in_stage_a,
        enable_grad_ckpt=use_gckpt,
    )
    if args.freeze_all_except_vlm_lora:
        p_counts = _freeze_to_vlm_lora_only()
    eff_mem_lr_scale = (
        1.0
        if (
            (bool(args.use_continuous_vlm_ground) or bool(args.reactive_precision_mode))
            and float(args.stage_b_mem_lr_scale) <= 0.0
        )
        else max(0.0, float(args.stage_b_mem_lr_scale))
    )
    param_groups = policy.build_optimizer_param_groups(
        lr_mem=args.lr_mem,
        lr_gate=args.lr_gate,
        lr_policy=args.lr_policy,
        lr_hooks=args.lr_hooks,
        stage_b_mem_lr_scale=eff_mem_lr_scale,
        weight_decay=args.weight_decay,
    )
    base_lrs = [float(g["lr"]) for g in param_groups]
    optimizer = FP32MasterAdamW(policy, param_groups, betas=(0.9, 0.95), eps=1e-8)
    ema = TrainableFP32EMA(
        policy,
        decay=args.ema_decay,
        warmup_ema=bool(args.warmup_ema or args.freeze_all_except_vlm_lora),
    )

    start_step = 0
    best_loss = float("inf")
    if args.resume == "auto" and latest_ckpt_path.exists():
        print(f"[Resume] Restoring training state from {latest_ckpt_path}...", flush=True)
        ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        policy.load_delta_state_dict(ckpt["model_delta"], strict_mem=False)
        policy.configure_stage(
            args.stage,
            freeze_mem_in_stage_b=freeze_mem_b,
            freeze_hooks_in_stage_a=args.freeze_hooks_in_stage_a,
            enable_grad_ckpt=use_gckpt,
        )
        if args.freeze_all_except_vlm_lora:
            p_counts = _freeze_to_vlm_lora_only()
        optimizer.load_state_dict(ckpt["optimizer"], device=dev)
        if "ema_shadow" in ckpt:
            ema.load_state_dict(ckpt["ema_shadow"], device=dev)
        start_step = int(ckpt["step"])
        best_loss = float(ckpt.get("best_loss", float("inf")))
        print(f"[Resume] Resumed cleanly at step={start_step}/{args.max_steps} (best_loss={best_loss:.5f})", flush=True)

    grp_summary = ", ".join(f"{g['name']}:lr={g['lr']:.2e}" for g in param_groups)
    print(
        f"[Train] arch={cfg.arch} stage={args.stage} stats={eff_stats_variant} max_mem_tokens={cfg.max_mem_tokens} "
        f"gdn_state_bytes={cfg.gdn_state_bytes} trainable={p_counts['trainable_params']:,}/"
        f"{p_counts['total_params']:,} steps={start_step}->{args.max_steps} "
        f"stratified={args.stratified_16task} groups=[{grp_summary}] (HAS_FLA={HAS_FLA})",
        flush=True,
    )

    eff_workers = 0 if args.smoke_synthetic else max(0, int(args.num_workers))
    if args.stratified_16task and len(train_ds) >= args.episodes_per_step:
        b_sampler = Stratified16TaskBatchSampler(
            train_ds, batch_size=args.episodes_per_step, seed=args.seed + start_step
        )
        loader = DataLoader(
            train_ds,
            batch_sampler=b_sampler,
            num_workers=eff_workers,
            collate_fn=collate_episode_list,
            pin_memory=(dev.type == "cuda" and not args.smoke_synthetic),
            persistent_workers=(eff_workers > 0),
            prefetch_factor=(2 if eff_workers > 0 else None),
        )
    else:
        loader = DataLoader(
            train_ds,
            batch_size=args.episodes_per_step,
            shuffle=True,
            num_workers=eff_workers,
            collate_fn=collate_episode_list,
            drop_last=len(train_ds) >= args.episodes_per_step,
            pin_memory=(dev.type == "cuda" and not args.smoke_synthetic),
            persistent_workers=(eff_workers > 0),
            prefetch_factor=(2 if eff_workers > 0 else None),
        )
    batch_iter = iter(infinite_episode_iterator(loader))

    policy.train()
    running: Dict[str, float] = {}
    running_n = 0
    t_log = time.perf_counter()

    def _save_state(step_done: int, is_periodic: bool = False, is_best: bool = False) -> None:
        dg_bundle: Dict[str, Any] = {}
        if bundled_dg_payload is not None:
            dg_bundle["deltaground"] = bundled_dg_payload
        if bundled_dgv_payload is not None:
            dg_bundle["ema_deltaground_vlm"] = bundled_dgv_payload
        if args.freeze_all_except_vlm_lora:
            dg_bundle["is_m7d_dynamic_lora"] = True
            if frozen_base_vlm_lora:
                dg_bundle["base_vlm_lora"] = frozen_base_vlm_lora
        payload = {
            "step": step_done,
            "best_loss": best_loss,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "model_delta": policy.export_delta_state_dict(),
            "ema_delta": ema.export_ema_delta(policy),
            "ema_shadow": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            **dg_bundle,
        }
        atomic_torch_save(payload, latest_ckpt_path)
        atomic_torch_save(
            {"step": step_done, "ema_delta": payload["ema_delta"], **dg_bundle},
            exp_dir / "latest_ema.pt",
        )
        if is_best:
            atomic_torch_save(
                {"step": step_done, "ema_delta": payload["ema_delta"], **dg_bundle},
                exp_dir / "best_ema.pt",
            )
        if is_periodic and args.keep_every > 0 and (step_done % args.keep_every == 0):
            atomic_torch_save(
                {"step": step_done, "ema_delta": payload["ema_delta"], **dg_bundle},
                exp_dir / f"step_{step_done:06d}_ema.pt",
            )

    for step in range(start_step, args.max_steps):
        lr_mult = cosine_lr_multiplier(step, args.max_steps, args.warmup_steps)
        for grp, b_lr in zip(optimizer.param_groups, base_lrs):
            grp["lr"] = b_lr * lr_mult

        optimizer.zero_grad(set_to_none=True)
        step_metrics: Dict[str, float] = {}

        for _ in range(max(1, args.grad_accum_steps)):
            ep_batch = next(batch_iter)
            out = policy.forward_training_episode_batch(
                ep_batch, max_chunks_per_ep=args.chunks_per_episode
            )
            loss = out["loss"] / float(max(1, args.grad_accum_steps))
            loss.backward()

            for k, v in out.items():
                if k == "loss":
                    val = float(out["loss"].detach().item())
                elif isinstance(v, torch.Tensor):
                    val = float(v.detach().item())
                else:
                    val = float(v)
                step_metrics[k] = step_metrics.get(k, 0.0) + val / float(max(1, args.grad_accum_steps))

        gn_mem, gn_pol = optimizer.clip_and_step(grad_clip=args.grad_clip)
        ema.update(policy, master_override=optimizer.master_named_tensors())

        step_done = step + 1
        running_n += 1
        step_metrics["grad_norm_mem"] = gn_mem
        step_metrics["grad_norm_pol"] = gn_pol
        for k, v in step_metrics.items():
            running[k] = running.get(k, 0.0) + v

        if step_done % args.log_every == 0 or step_done == args.max_steps:
            dt = max(1e-6, time.perf_counter() - t_log)
            it_s = running_n / dt
            avg = {k: v / float(running_n) for k, v in running.items()}
            vram_gb = (
                torch.cuda.max_memory_allocated(dev) / (1024**3) if dev.type == "cuda" else 0.0
            )
            gh_extra = ""
            if policy.mem_core.hybrid_bridge is not None:
                gh_mean = float(
                    torch.tanh(policy.mem_core.hybrid_bridge.gamma_hybrid.detach().float()).abs().mean().item()
                )
                avg["tanh_gamma_hybrid"] = gh_mean
                gh_extra = f" g_hyb={gh_mean:.4f}"
            if policy.mem_core.continuous_vlm_bridge is not None:
                gv_a = float(
                    torch.tanh(policy.mem_core.continuous_vlm_bridge.gamma_vlm.detach().float()).abs().mean().item()
                )
                gv_d = float(
                    torch.tanh(policy.mem_core.continuous_vlm_bridge.gamma_vlm_direct.detach().float()).abs().mean().item()
                )
                avg["tanh_gamma_vlm"] = gv_a
                avg["tanh_gamma_vlm_direct"] = gv_d
                gh_extra += f" g_vlm={gv_a:.4f}/{gv_d:.4f}"
            rec = {
                "step": step_done,
                "lr_mult": round(lr_mult, 5),
                "it_per_sec": round(it_s, 3),
                "peak_vram_gb": round(vram_gb, 3),
                **{k: round(v, 6) for k, v in avg.items()},
            }
            print(
                f"[Step {step_done:05d}/{args.max_steps}] fm={rec.get('fm_loss', 0.0):.5f} "
                f"aux={rec.get('aux_total', 0.0):.5f} xy={rec.get('xy_loss', 0.0):.5f} "
                f"cnt={rec.get('count_loss', 0.0):.5f}{gh_extra} ({it_s:.2f} it/s, VRAM={vram_gb:.2f}GB)",
                flush=True,
            )
            with metrics_jsonl.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec) + "\n")

            if val_fm_ds is None:
                is_new_best = avg.get("fm_loss", float("inf")) < best_loss
                if is_new_best:
                    best_loss = avg["fm_loss"]
                if is_new_best and step_done >= min(500, args.max_steps):
                    _save_state(step_done, is_periodic=False, is_best=True)

            running.clear()
            running_n = 0
            t_log = time.perf_counter()

        if val_fm_ds is not None and ((args.eval_every > 0 and step_done % args.eval_every == 0) or step_done == args.max_steps):
            backups_v: Dict[str, torch.Tensor] = {}
            pmap_v = dict(policy.named_parameters())
            was_tr = policy.training
            rng_cpu = torch.get_rng_state()
            rng_cuda = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None
            try:
                for n_v, ev_v in ema.shadow.items():
                    if n_v in pmap_v:
                        backups_v[n_v] = pmap_v[n_v].data.clone()
                        pmap_v[n_v].data.copy_(ev_v.to(device=pmap_v[n_v].device, dtype=pmap_v[n_v].dtype))
                policy.eval()
                torch.manual_seed(args.seed + 7777)
                if dev.type == "cuda":
                    torch.cuda.manual_seed_all(args.seed + 7777)
                val_fm_ds.rng = np.random.default_rng(args.seed + 7777)
                task_fms: Dict[str, List[float]] = {}
                with torch.no_grad():
                    for vi in range(len(val_fm_ds)):
                        ep_v = val_fm_ds[vi]
                        out_v = policy.forward_training_episode_batch([ep_v], max_chunks_per_ep=8)
                        task_fms.setdefault(str(ep_v["task_name"]), []).append(float(out_v["fm_loss"].item()))
                per_t_v = {tk: round(float(np.mean(vs)), 6) for tk, vs in sorted(task_fms.items())}
                mean_v = float(np.mean([v for vs in task_fms.values() for v in vs])) if task_fms else float("inf")
                is_best_v = mean_v < best_loss
                if is_best_v:
                    best_loss = mean_v
                    _save_state(step_done, is_periodic=False, is_best=True)
                pt_v = " ".join(f"{tk}={vv:.5f}" for tk, vv in per_t_v.items())
                print(
                    f"[Unseen Val FM @ {step_done:05d}] mean_fm={mean_v:.5f} ({pt_v}) {'(NEW BEST!)' if is_best_v else ''}",
                    flush=True,
                )
                with metrics_jsonl.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps({"step": step_done, "eval_unseen_val_fm": True, "val_mean_fm": round(mean_v, 6), "per_task_val_fm": per_t_v}) + "\n")
            finally:
                for n_v, orig_v in backups_v.items():
                    pmap_v[n_v].data.copy_(orig_v)
                torch.set_rng_state(rng_cpu)
                if rng_cuda is not None and dev.type == "cuda":
                    torch.cuda.set_rng_state(rng_cuda, dev)
                if was_tr:
                    policy.train()

        if step_done % args.save_every == 0 or step_done == args.max_steps:
            _save_state(step_done, is_periodic=True, is_best=(step_done == args.max_steps and best_loss == float("inf")))
            print(f"[Ckpt] Saved atomic resumable checkpoint at step {step_done} -> {latest_ckpt_path}", flush=True)

        elapsed_min = (time.perf_counter() - wall_start) / 60.0
        if elapsed_min >= args.max_wall_minutes and step_done < args.max_steps:
            _save_state(step_done, is_periodic=True, is_best=False)
            print(
                f"[Walltime Guard] Reached {elapsed_min:.1f} min >= {args.max_wall_minutes:.1f} min at step "
                f"{step_done}/{args.max_steps}; saved `{latest_ckpt_path}` cleanly for chained `interact-g` resume!",
                flush=True,
            )
            return

    print(f"[Done] Completed {args.max_steps} steps in {(time.perf_counter() - wall_start)/60.0:.2f} min -> {exp_dir}", flush=True)


if __name__ == "__main__":
    main()
