#!/usr/bin/env python3
"""Cycle 1 Synthetic Mechanistic & Ablation Runner for DeltaLoop-VLA.

Executes shortcut-free 3-seed evaluations on RTX 5080 (VRAM <= 6 GB):
  1. `1a_online`: Online Container Swap State Tracking (ID k=1..4, OOD k=5..6)
  2. `1a_retro`:  Retroactive Multi-Hop Relational Pointer Chaining
                  (Full (k, L) matrix, Forward vs Reverse causal ordering,
                   Untied Deep Readout vs Weight-Tied DeltaLoop, S_prev vs S_curr,
                   2-Stage Adaptive Halting, Query-Key Hop Alignment Probe)
  3. `1b`:        T=2048 Key-Collision Multi-Timescale Interference Benchmark
                  (Strictly Iso-State = 1536 floats across 8 variants)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.loop.halting import AdaptiveHaltingHead
from src.loop.looped_core import Arch1A, DeltaLoopSequenceModel, MultiTimescaleSequenceModel
from src.memory.multi_timescale_gdn import Variant1B
from src.synthetic import (
    MultiTimescaleDataset,
    OnlineSwapDataset,
    RetroChainDataset,
    audit_multiscale_dataset,
    audit_online_swap_dataset,
    audit_retro_chain_dataset,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clip_and_step(model: nn.Module, opt: torch.optim.Optimizer, max_norm: float = 1.0) -> None:
    """Clip temporal-write parameters and recurrent-depth parameters independently.

    Prevents L-step depth unrolling in RecurrentCore from throttling gradient updates
    to single-pass Temporal Write (GDN) parameters under global norm clipping!
    """
    write_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        if any(tag in name for tag in ("gdn", "write_loop", "encoder", "memory")):
            write_params.append(p)
        else:
            other_params.append(p)
    if write_params:
        torch.nn.utils.clip_grad_norm_(write_params, max_norm)
    if other_params:
        torch.nn.utils.clip_grad_norm_(other_params, max_norm)
    opt.step()


def summarize_seeds(rows: List[Dict[str, Any]], metric_keys: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in metric_keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] is not None]
        if vals:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0)),
                "per_seed": vals,
            }
    return out


# =============================================================================
# Task 1A Evaluation Helpers
# =============================================================================


@torch.no_grad()
def eval_1a_split(
    model: DeltaLoopSequenceModel,
    X: torch.Tensor,
    Y: torch.Tensor,
    K: torch.Tensor,
    k_list: Sequence[int],
    max_loops: int = 6,
    order_types: torch.Tensor | None = None,
    hop_timesteps: torch.Tensor | None = None,
    batch_size: int = 512,
) -> Dict[str, Any]:
    model.eval()
    N = len(X)
    L_eval = max_loops if model.is_multi_depth else 1

    all_logits_d = []
    all_halt_p_tilde = []
    all_dh = []
    hop_hit_counts = np.zeros((max_loops, 4), dtype=np.float64)
    hop_totals = np.zeros(4, dtype=np.float64)

    need_probes = hop_timesteps is not None and model.arch == "deltaloop_b2"

    for i in range(0, N, batch_size):
        xb = X[i : i + batch_size]
        out = model(xb, num_loops=L_eval, return_probes=need_probes)
        ld = out["logits_per_depth"]  # (b, L_eval, C)
        all_logits_d.append(ld)
        if "halt" in out:
            all_halt_p_tilde.append(out["halt"]["p_tilde"])
        if "dh_norms" in out:
            all_dh.append(out["dh_norms"])

        if need_probes and "qk_sim" in out and hop_timesteps is not None:
            # qk_sim: (b, L_eval, H, T) -> max abs cosine similarity across heads per (b, l, t)
            qk = out["qk_sim"].abs().amax(dim=2).cpu().numpy()  # (b, L_eval, T)
            ht = hop_timesteps[i : i + batch_size].cpu().numpy()  # (b, max_k)
            kb = K[i : i + batch_size].cpu().numpy()
            # Exclude final query step T-1 when ranking prefix memory keys 0..T-2
            prefix_top1 = qk[:, :, :-1].argmax(axis=-1)  # (b, L_eval)
            for bi in range(len(xb)):
                if kb[bi] == 4:
                    for h_idx in range(4):
                        t_true = ht[bi, h_idx]
                        if t_true >= 0:
                            hop_totals[h_idx] += 1.0
                            for l_idx in range(L_eval):
                                if prefix_top1[bi, l_idx] == t_true:
                                    hop_hit_counts[l_idx, h_idx] += 1.0

    logits_d = torch.cat(all_logits_d, dim=0)  # (N, L_eval, C)
    preds_d = logits_d.argmax(dim=-1)  # (N, L_eval)

    # Accuracy matrix: depth L in 1..L_eval vs hop count k
    matrix_k_L: Dict[str, List[float]] = {}
    overall_per_L: List[float] = []
    for l_idx in range(L_eval):
        overall_per_L.append(float((preds_d[:, l_idx] == Y).float().mean().item() * 100.0))

    for k_val in k_list:
        mask_k = K == k_val
        row = []
        for l_idx in range(L_eval):
            acc_kl = float((preds_d[mask_k, l_idx] == Y[mask_k]).float().mean().item() * 100.0)
            row.append(acc_kl)
        matrix_k_L[f"k={k_val}"] = row

    result: Dict[str, Any] = {
        "overall_per_L": overall_per_L,
        "final_L_overall": overall_per_L[-1],
        "matrix_k_L": matrix_k_L,
    }

    # Causal ordering breakdown at final depth (for Task 1A-Retro)
    if order_types is not None:
        ord_names = {0: "forward", 1: "reverse", 2: "mixed", 3: "single"}
        ord_acc: Dict[str, float] = {}
        ord_k_acc: Dict[str, Dict[str, float]] = {}
        for o_code, o_name in ord_names.items():
            mo = order_types == o_code
            if int(mo.sum().item()) > 0:
                ord_acc[o_name] = float(
                    (preds_d[mo, -1] == Y[mo]).float().mean().item() * 100.0
                )
                per_k = {}
                for k_val in k_list:
                    mok = mo & (K == k_val)
                    if int(mok.sum().item()) > 0:
                        per_k[f"k={k_val}"] = float(
                            (preds_d[mok, -1] == Y[mok]).float().mean().item() * 100.0
                        )
                ord_k_acc[o_name] = per_k
        result["order_breakdown_final_L"] = ord_acc
        result["order_by_k_final_L"] = ord_k_acc

    # Stage-2 Adaptive Halting evaluation
    if all_halt_p_tilde:
        p_tilde = torch.cat(all_halt_p_tilde, dim=0)
        halt_sel = AdaptiveHaltingHead.select_halt_step(
            {"p_tilde": p_tilde}, logits_d, threshold=0.5
        )
        halt_preds = halt_sel["logits"].argmax(dim=-1)
        halt_loops = halt_sel["halt_loops"]
        halt_by_k = {}
        for k_val in k_list:
            mk = K == k_val
            halt_by_k[f"k={k_val}"] = {
                "acc": float((halt_preds[mk] == Y[mk]).float().mean().item() * 100.0),
                "mean_loops": float(halt_loops[mk].mean().item()),
            }
        result["adaptive_halting"] = {
            "overall_acc": float((halt_preds == Y).float().mean().item() * 100.0),
            "mean_loops": float(halt_loops.mean().item()),
            "by_k": halt_by_k,
        }

    # Latent L2-convergence stopping baseline (||h^{(l)} - h^{(l-1)}||_2 < tau)
    if all_dh and L_eval > 1:
        dh_cat = torch.cat(all_dh, dim=0)  # (N, L_eval)
        # Calibrate median step-2..L delta threshold
        tau = float(torch.quantile(dh_cat[:, 1:], 0.35).item())
        hit_conv = dh_cat < tau
        hit_conv[:, 0] = False  # Require >= 1 transition comparison or allow after l>=1
        hit_conv[:, -1] = True
        conv_idx = torch.argmax(hit_conv.int(), dim=1)
        conv_logits = torch.gather(
            logits_d, 1, conv_idx.view(N, 1, 1).expand(N, 1, logits_d.shape[-1])
        ).squeeze(1)
        result["latent_l2_convergence"] = {
            "overall_acc": float((conv_logits.argmax(-1) == Y).float().mean().item() * 100.0),
            "mean_loops": float((conv_idx + 1).float().mean().item()),
        }

    if need_probes and hop_totals.min() > 0:
        # Row l_idx (loop 1..6), Col h_idx (true hop 1..4): top-1 key match rate (%)
        align_mat = (hop_hit_counts / hop_totals[None, :] * 100.0).tolist()
        result["qk_hop_top1_alignment_k4"] = align_mat

    return result


def train_and_eval_1a_model(
    task_name: str,
    arch: Arch1A,
    seed: int,
    datasets: Dict[str, Any],
    device: torch.device,
    steps: int = 2600,
    halting_steps: int = 450,
    max_loops: int = 6,
) -> Dict[str, Any]:
    set_seed(seed)
    num_ops = datasets["num_ops"]
    vocab_size = datasets["vocab_size"]
    num_classes = datasets["num_classes"]

    model = DeltaLoopSequenceModel(
        arch=arch,
        num_ops=num_ops,
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_loops=max_loops,
    ).to(device)

    # Freeze Stage-2 halting head during Stage-1 policy training
    if model.halting_head is not None:
        for p in model.halting_head.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-3,
        weight_decay=1e-2,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)

    X1, Y1 = datasets["X_c1"], datasets["Y_c1"]
    X2, Y2 = datasets["X_c2"], datasets["Y_c2"]
    X4, Y4 = datasets["X_train"], datasets["Y_train"]

    bs = 256
    curve: List[Dict[str, float]] = []
    t0 = time.time()

    for s in range(1, steps + 1):
        model.train()
        if s <= 350:
            X_cur, Y_cur = X1, Y1
        elif s <= 800:
            X_cur, Y_cur = X2, Y2
        else:
            X_cur, Y_cur = X4, Y4

        idx = torch.randint(0, len(X_cur), (bs,), device=device)
        out = model(X_cur[idx], num_loops=max_loops)
        ld = out["logits_per_depth"]
        if ld.shape[1] == 1:
            loss = F.cross_entropy(ld[:, 0], Y_cur[idx])
        else:
            # Uniform all-depth supervision + final-depth anchor (enables 1 model across L=1..6!)
            all_d = sum(F.cross_entropy(ld[:, l], Y_cur[idx]) for l in range(ld.shape[1])) / ld.shape[1]
            loss = 0.5 * F.cross_entropy(ld[:, -1], Y_cur[idx]) + 0.5 * all_d

        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_and_step(model, opt, max_norm=1.0)
        sched.step()

        if s in (400, 900, 1500, 2100, steps):
            model.eval()
            with torch.no_grad():
                val_logits = model(datasets["X_id"][:512], num_loops=max_loops)["logits"]
                val_acc = float((val_logits.argmax(-1) == datasets["Y_id"][:512]).float().mean().item() * 100.0)
            curve.append({"step": s, "loss": float(loss.item()), "id_acc_sample": val_acc})

    # Stage 2: Train Adaptive Halting Head on FROZEN policy!
    if model.halting_head is not None:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.halting_head.parameters():
            p.requires_grad_(True)

        opt_h = torch.optim.AdamW(model.halting_head.parameters(), lr=3e-3, weight_decay=1e-3)
        for hs in range(1, halting_steps + 1):
            idx = torch.randint(0, len(X4), (bs,), device=device)
            yb = Y4[idx]
            out = model(X4[idx], num_loops=max_loops)
            ld = out["logits_per_depth"].detach()  # (B, L, C)
            per_depth_ce = torch.stack(
                [F.cross_entropy(ld[:, l], yb, reduction="none") for l in range(ld.shape[1])],
                dim=1,
            )
            h_loss = model.halting_head.compute_stage2_loss(
                out["halt"],
                per_depth_ce,
                ponder_cost=0.06,
                kl_weight=0.02,
                lambda_prior=0.30,
            )
            opt_h.zero_grad(set_to_none=True)
            h_loss.backward()
            opt_h.step()

    train_sec = time.time() - t0

    # Full evaluation on ID, OOD-Hops (same T), and OOD-Length+Hops (longer T)
    id_res = eval_1a_split(
        model,
        datasets["X_id"],
        datasets["Y_id"],
        datasets["K_id"],
        k_list=(1, 2, 3, 4),
        max_loops=max_loops,
        order_types=datasets.get("Ord_id"),
        hop_timesteps=datasets.get("HopT_id"),
    )
    ood_hops_res = eval_1a_split(
        model,
        datasets["X_ood_hops"],
        datasets["Y_ood_hops"],
        datasets["K_ood_hops"],
        k_list=(5, 6),
        max_loops=max_loops,
        order_types=datasets.get("Ord_ood_hops"),
    )
    ood_len_res = eval_1a_split(
        model,
        datasets["X_ood_len"],
        datasets["Y_ood_len"],
        datasets["K_ood_len"],
        k_list=(5, 6),
        max_loops=max_loops,
        order_types=datasets.get("Ord_ood_len"),
    )

    peak_vram_mb = float(torch.cuda.max_memory_allocated() / 1e6)
    print(
        f"[{task_name}][seed={seed}] {arch:24s} | {train_sec:5.1f}s | "
        f"ID={id_res['final_L_overall']:5.1f}% | "
        f"OOD(k=5,6)={ood_hops_res['final_L_overall']:5.1f}% | "
        f"OOD-Len={ood_len_res['final_L_overall']:5.1f}% | VRAM={peak_vram_mb:.0f}MB",
        flush=True,
    )

    return {
        "arch": arch,
        "seed": seed,
        "core_params": model.core_parameter_count(),
        "state_numel": model.state_numel,
        "state_bytes_fp32": model.state_numel * 4 if model.state_numel > 0 else -1,
        "train_seconds": train_sec,
        "peak_vram_mb": peak_vram_mb,
        "learning_curve": curve,
        "id": id_res,
        "ood_hops": ood_hops_res,
        "ood_len_hops": ood_len_res,
        "id_acc": id_res["final_L_overall"],
        "ood_hops_acc": ood_hops_res["final_L_overall"],
        "ood_len_acc": ood_len_res["final_L_overall"],
    }


def _load_existing_runs(out_path: Path, keys: Sequence[str], resume: bool) -> Dict[str, List[Dict[str, Any]]]:
    all_runs: Dict[str, List[Dict[str, Any]]] = {k: [] for k in keys}
    if resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            prev_runs = prev.get("runs", {})
            for k in keys:
                if k in prev_runs and isinstance(prev_runs[k], list):
                    all_runs[k] = prev_runs[k]
        except Exception as exc:
            print(f"[WARN] Could not load existing {out_path}: {exc}", flush=True)
    return all_runs


def run_task_1a_online(
    seeds: Sequence[int], out_dir: Path, device: torch.device, resume: bool = False
) -> Dict[str, Any]:
    print("\n================ TASK 1A-ONLINE: CONTAINER SWAP STATE TRACKING ================", flush=True)
    ds_c1 = OnlineSwapDataset(16000, k_values=(1,), seq_len=40, seed=11)
    ds_c2 = OnlineSwapDataset(24000, k_values=(1, 2), seq_len=40, seed=12)
    ds_tr = OnlineSwapDataset(56000, k_values=(1, 2, 3, 4), seq_len=40, seed=13)
    ds_id = OnlineSwapDataset(2400, k_values=(1, 2, 3, 4), seq_len=40, seed=101)
    ds_ood_h = OnlineSwapDataset(2000, k_values=(5, 6), seq_len=40, seed=102)
    ds_ood_l = OnlineSwapDataset(2000, k_values=(5, 6), seq_len=56, total_swaps=16, seed=103)

    audit = {
        "id": audit_online_swap_dataset(ds_id),
        "ood_hops": audit_online_swap_dataset(ds_ood_h),
    }

    datasets = {
        "num_ops": OnlineSwapDataset.NUM_OPS,
        "vocab_size": OnlineSwapDataset.VOCAB_SIZE,
        "num_classes": OnlineSwapDataset.NUM_CUPS,
        "X_c1": ds_c1.tokens.to(device),
        "Y_c1": ds_c1.labels.to(device),
        "X_c2": ds_c2.tokens.to(device),
        "Y_c2": ds_c2.labels.to(device),
        "X_train": ds_tr.tokens.to(device),
        "Y_train": ds_tr.labels.to(device),
        "X_id": ds_id.tokens.to(device),
        "Y_id": ds_id.labels.to(device),
        "K_id": ds_id.k_hops.to(device),
        "X_ood_hops": ds_ood_h.tokens.to(device),
        "Y_ood_hops": ds_ood_h.labels.to(device),
        "K_ood_hops": ds_ood_h.k_hops.to(device),
        "X_ood_len": ds_ood_l.tokens.to(device),
        "Y_ood_len": ds_ood_l.labels.to(device),
        "K_ood_len": ds_ood_l.k_hops.to(device),
    }

    archs: List[Arch1A] = [
        "single_gdn_b1",
        "single_gdn_b2",
        "two_layer_gdn_b2",
        "write_loop_gdn_b2",
        "static_query_loop_b2",
        "parallel_multiquery_b2",
        "untied_deep_readout_b2",
        "deltaloop_b1",
        "deltaloop_b2",
        "softmax_transformer",
    ]

    out_path = out_dir / "task1a_online.json"
    all_runs = _load_existing_runs(out_path, archs, resume=resume)
    for a in archs:
        done_seeds = {int(r["seed"]) for r in all_runs[a]}
        for sd in seeds:
            if sd in done_seeds:
                print(f"[1A-Online][seed={sd}] {a:24s} | [RESUME] already completed, skipping", flush=True)
                continue
            res = train_and_eval_1a_model(
                task_name="1A-Online",
                arch=a,
                seed=sd,
                datasets=datasets,
                device=device,
                steps=2400,
                halting_steps=400,
                max_loops=6,
            )
            all_runs[a].append(res)
            summary = {
                arch_k: summarize_seeds(runs_v, ("id_acc", "ood_hops_acc", "ood_len_acc", "train_seconds"))
                for arch_k, runs_v in all_runs.items()
                if runs_v
            }
            out_path.write_text(
                json.dumps({"audit": audit, "summary": summary, "runs": all_runs}, indent=2)
            )

    return json.loads(out_path.read_text())


def run_task_1a_retro(
    seeds: Sequence[int], out_dir: Path, device: torch.device, resume: bool = False
) -> Dict[str, Any]:
    print("\n================ TASK 1A-RETRO: RETROACTIVE POINTER CHAINING ================", flush=True)
    ds_c1 = RetroChainDataset(16000, k_values=(1,), seq_len=44, seed=21)
    ds_c2 = RetroChainDataset(24000, k_values=(1, 2), seq_len=44, seed=22)
    ds_tr = RetroChainDataset(64000, k_values=(1, 2, 3, 4), seq_len=44, seed=23)
    ds_id = RetroChainDataset(2400, k_values=(1, 2, 3, 4), seq_len=44, seed=201)
    ds_ood_h = RetroChainDataset(2000, k_values=(5, 6), seq_len=44, seed=202)
    ds_ood_l = RetroChainDataset(2000, k_values=(5, 6), seq_len=56, seed=203)

    audit = {
        "id": audit_retro_chain_dataset(ds_id),
        "ood_hops": audit_retro_chain_dataset(ds_ood_h),
    }

    datasets = {
        "num_ops": RetroChainDataset.NUM_OPS,
        "vocab_size": RetroChainDataset.VOCAB_SIZE,
        "num_classes": RetroChainDataset.NUM_ZONES,
        "X_c1": ds_c1.tokens.to(device),
        "Y_c1": ds_c1.labels.to(device),
        "X_c2": ds_c2.tokens.to(device),
        "Y_c2": ds_c2.labels.to(device),
        "X_train": ds_tr.tokens.to(device),
        "Y_train": ds_tr.labels.to(device),
        "X_id": ds_id.tokens.to(device),
        "Y_id": ds_id.labels.to(device),
        "K_id": ds_id.k_hops.to(device),
        "Ord_id": ds_id.order_types.to(device),
        "HopT_id": ds_id.hop_timesteps.to(device),
        "X_ood_hops": ds_ood_h.tokens.to(device),
        "Y_ood_hops": ds_ood_h.labels.to(device),
        "K_ood_hops": ds_ood_h.k_hops.to(device),
        "Ord_ood_hops": ds_ood_h.order_types.to(device),
        "X_ood_len": ds_ood_l.tokens.to(device),
        "Y_ood_len": ds_ood_l.labels.to(device),
        "K_ood_len": ds_ood_l.k_hops.to(device),
        "Ord_ood_len": ds_ood_l.order_types.to(device),
    }

    archs: List[Arch1A] = [
        "single_gdn_b1",
        "single_gdn_b2",
        "two_layer_gdn_b2",
        "write_loop_gdn_b2",
        "static_query_loop_b2",
        "parallel_multiquery_b2",
        "untied_deep_readout_b2",
        "deltaloop_b1",
        "deltaloop_b2",
        "deltaloop_b2_scurr",
        "softmax_transformer",
    ]

    out_path = out_dir / "task1a_retro.json"
    all_runs = _load_existing_runs(out_path, archs, resume=resume)
    for a in archs:
        done_seeds = {int(r["seed"]) for r in all_runs[a]}
        for sd in seeds:
            if sd in done_seeds:
                print(f"[1A-Retro][seed={sd}] {a:24s} | [RESUME] already completed, skipping", flush=True)
                continue
            res = train_and_eval_1a_model(
                task_name="1A-Retro",
                arch=a,
                seed=sd,
                datasets=datasets,
                device=device,
                steps=2800,
                halting_steps=450,
                max_loops=6,
            )
            all_runs[a].append(res)
            summary = {
                arch_k: summarize_seeds(runs_v, ("id_acc", "ood_hops_acc", "ood_len_acc", "train_seconds"))
                for arch_k, runs_v in all_runs.items()
                if runs_v
            }
            out_path.write_text(
                json.dumps({"audit": audit, "summary": summary, "runs": all_runs}, indent=2)
            )

    return json.loads(out_path.read_text())


# =============================================================================
# Task 1B: Key-Collision Multi-Timescale Interference (T = 2048, Iso-State)
# =============================================================================


def build_1b_short_warmup(num_samples: int = 10000, seq_len: int = 128, seed: int = 77) -> Tuple[torch.Tensor, torch.Tensor]:
    """Short-horizon warmup stream with identical op/symbol semantics so channel keys form cleanly."""
    rng = np.random.default_rng(seed)
    tok = np.zeros((num_samples, seq_len, 3), dtype=np.int64)
    lab = np.zeros(num_samples, dtype=np.int64)
    for i in range(num_samples):
        sub = i % 5
        m = rng.random(seq_len - 1) < 0.18
        c = rng.integers(0, 4, size=seq_len - 1)
        v = rng.integers(0, 8, size=seq_len - 1)
        tok[i, :-1, 0] = np.where(m, 1, 0)
        tok[i, :-1, 1] = np.where(m, 1 + c, 0)
        tok[i, :-1, 2] = np.where(m, 7 + v, 0)

        if sub == 0:  # short
            qc = int(rng.integers(0, 4)); tv = int(rng.integers(0, 8))
            ts = int(rng.integers(seq_len - 25, seq_len - 6))
            tok[i, ts] = (1, 1 + qc, 7 + tv)
            for tt in range(ts + 1, seq_len - 1):
                if tok[i, tt, 0] == 1 and tok[i, tt, 1] == 1 + qc:
                    tok[i, tt] = (0, 0, 0)
            tok[i, -1] = (4, 1 + qc, 15 + 0); lab[i] = tv
        elif sub in (1, 2):  # mid/long persist
            qc = 5 if sub == 1 else 4
            sc = 1 if sub == 1 else 2
            tv = int(rng.integers(0, 8))
            ts = int(rng.integers(6, 50))
            tok[i, ts] = (1, 1 + qc, 7 + tv)
            tok[i, -1] = (4, 1 + qc, 15 + sc); lab[i] = tv
        else:  # mid/long collide
            qc = int(rng.integers(2, 4)) if sub == 3 else int(rng.integers(0, 2))
            sc = 1 if sub == 3 else 2
            lock_op = 2 if sub == 3 else 3
            tv = int(rng.integers(0, 8))
            ts = int(rng.integers(8, 55))
            tok[i, ts - 1] = (1, 1 + qc, 7 + tv)
            tok[i, ts] = (lock_op, 1 + qc, 0)
            tok[i, seq_len - 12] = (1, 1 + qc, 7 + ((tv + 3) % 8))
            tok[i, -1] = (4, 1 + qc, 15 + sc); lab[i] = tv
    return torch.from_numpy(tok), torch.from_numpy(lab)


def run_task_1b(
    seeds: Sequence[int], out_dir: Path, device: torch.device, resume: bool = False
) -> Dict[str, Any]:
    print("\n================ TASK 1B: T=2048 MULTI-TIMESCALE INTERFERENCE ================", flush=True)
    Xw_cpu, Yw_cpu = build_1b_short_warmup(10000, seq_len=128, seed=77)
    Xw, Yw = Xw_cpu.to(device), Yw_cpu.to(device)

    ds_tr = MultiTimescaleDataset(10000, seq_len=2048, seed=31)
    ds_va = MultiTimescaleDataset(2000, seq_len=2048, seed=301)
    audit = audit_multiscale_dataset(ds_va)

    Xtr, Ytr = ds_tr.tokens.to(device), ds_tr.labels.to(device)
    Xva, Yva = ds_va.tokens.to(device), ds_va.labels.to(device)
    Sva, Qva = ds_va.subtypes.to(device), ds_va.query_scales.to(device)

    variants: List[Variant1B] = [
        "1_single_mh",
        "1b_stacked_3layer",
        "2_3bank_same_fixed",
        "3_3bank_diverse_fixed",
        "4_3bank_learnable",
        "5a_3bank_consolidation",
        "5b_3bank_routing",
        "5c_3bank_full",
    ]

    out_path = out_dir / "task1b.json"
    out_path_alias = out_dir / "task1b_multiscale.json"
    load_path = out_path if out_path.exists() else out_path_alias
    all_runs = _load_existing_runs(load_path, variants, resume=resume)

    warm_steps = 350
    main_steps = 700
    total_steps = warm_steps + main_steps

    for v in variants:
        done_seeds = {int(r["seed"]) for r in all_runs[v]}
        for sd in seeds:
            if sd in done_seeds:
                print(f"[1B][seed={sd}] {v:24s} | [RESUME] already completed, skipping", flush=True)
                continue
            set_seed(sd)
            model = MultiTimescaleSequenceModel(
                variant=v,
                num_ops=MultiTimescaleDataset.NUM_OPS,
                vocab_size=MultiTimescaleDataset.VOCAB_SIZE,
                num_classes=MultiTimescaleDataset.NUM_VALUES,
                chunk_size=64,
            ).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps, eta_min=1.5e-4
            )
            curve: List[Dict[str, float]] = []
            t0 = time.time()

            for s in range(1, total_steps + 1):
                model.train()
                if s <= warm_steps:
                    idx = torch.randint(0, len(Xw), (256,), device=device)
                    out = model(Xw[idx])
                    loss = F.cross_entropy(out["logits"], Yw[idx])
                else:
                    idx = torch.randint(0, len(Xtr), (64,), device=device)
                    out = model(Xtr[idx])
                    loss = F.cross_entropy(out["logits"], Ytr[idx])

                opt.zero_grad(set_to_none=True)
                loss.backward()
                clip_and_step(model, opt, max_norm=1.0)
                sched.step()

                if s in (warm_steps, warm_steps + 200, warm_steps + 450, total_steps):
                    curve.append({"step": s, "loss": float(loss.item())})

            train_sec = time.time() - t0
            model.eval()
            preds_list = []
            pis_list = []
            alphas_list = []
            betas_list = []

            with torch.no_grad():
                for i in range(0, len(Xva), 128):
                    o_va = model(Xva[i : i + 128])
                    preds_list.append(o_va["logits"].argmax(dim=-1))
                    if o_va["pi"] is not None:
                        pis_list.append(o_va["pi"])
                    alphas_list.append(o_va["mem_diag"]["mean_alpha"])
                    betas_list.append(o_va["mem_diag"]["mean_beta"])

            preds = torch.cat(preds_list, dim=0)
            overall_acc = float((preds == Yva).float().mean().item() * 100.0)
            subtype_accs: Dict[str, float] = {}
            for s_idx, s_name in enumerate(MultiTimescaleDataset.SUBTYPE_NAMES):
                ms = Sva == s_idx
                subtype_accs[s_name] = float((preds[ms] == Yva[ms]).float().mean().item() * 100.0)

            routing_matrix = None
            if pis_list:
                pi_all = torch.cat(pis_list, dim=0)  # (N, 3)
                rmat = {}
                for q_sc, sc_name in enumerate(("query_short", "query_mid", "query_long")):
                    mq = Qva == q_sc
                    rmat[sc_name] = pi_all[mq].mean(dim=0).cpu().tolist()
                routing_matrix = rmat

            mean_alpha = torch.stack(alphas_list, dim=0).mean(dim=0).cpu().tolist()
            mean_beta = torch.stack(betas_list, dim=0).mean(dim=0).cpu().tolist()
            peak_vram_mb = float(torch.cuda.max_memory_allocated() / 1e6)

            print(
                f"[1B][seed={sd}] {v:24s} | {train_sec:5.1f}s | ALL={overall_acc:5.1f}% | "
                + " ".join(f"{k}:{val:5.1f}%" for k, val in subtype_accs.items())
                + f" | VRAM={peak_vram_mb:.0f}MB",
                flush=True,
            )

            run_rec: Dict[str, Any] = {
                "variant": v,
                "seed": sd,
                "params": sum(p.numel() for p in model.parameters()),
                "state_numel": model.state_numel,
                "state_bytes_fp32": model.state_numel * 4,
                "train_seconds": train_sec,
                "peak_vram_mb": peak_vram_mb,
                "overall_acc": overall_acc,
                **subtype_accs,
                "routing_matrix": routing_matrix,
                "mean_alpha_per_head": mean_alpha,
                "mean_beta_per_head": mean_beta,
                "learning_curve": curve,
            }
            all_runs[v].append(run_rec)

            metric_keys = ("overall_acc", *MultiTimescaleDataset.SUBTYPE_NAMES, "train_seconds")
            summary = {
                vk: summarize_seeds(rv, metric_keys)
                for vk, rv in all_runs.items()
                if rv
            }
            payload_1b = json.dumps({"audit": audit, "summary": summary, "runs": all_runs}, indent=2)
            out_path.write_text(payload_1b)
            out_path_alias.write_text(payload_1b)

    return json.loads(out_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cycle 1 Synthetic Mechanistic Suite.")
    parser.add_argument(
        "--task",
        choices=["1a_online", "1a_retro", "1b", "all"],
        default="all",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "synthetic"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed (arch/variant, seed) entries in output JSON files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running on device={device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}), "
        f"seeds={args.seeds}, resume={args.resume}",
        flush=True,
    )

    if args.task in ("1a_retro", "all"):
        run_task_1a_retro(args.seeds, out_dir, device, resume=args.resume)
    if args.task in ("1a_online", "all"):
        run_task_1a_online(args.seeds, out_dir, device, resume=args.resume)
    if args.task in ("1b", "all"):
        run_task_1b(args.seeds, out_dir, device, resume=args.resume)


if __name__ == "__main__":
    main()
