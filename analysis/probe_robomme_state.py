#!/usr/bin/env python3
"""Mechanistic State & Multi-Hop Loop Probe Suite for DeltaLoop-VLA (`RESEARCH_PLAN.md` v2.3 §4.3).

Evaluates frozen representations and recurrent memory states of trained `DeltaLoopVLAPolicy`
checkpoints (`m0..m6b` from Cycle 2A or Cycle 3 `U1..U5`) on cached `{TaskName}.pt` RoboMME shards
(or built-in synthetic smoke-test shards) with ZERO intra-episode train/test frame leakage:

  - **Probe 1 (Hop-by-Hop `<x, y>` Target Dereferencing):**
    Decodes canonical `(x_norm, y_norm)` target coordinates (parsed via `analysis.subgoal_grounding`
    with verified RoboMME `<row_y, col_x> -> (x_col, y_row)` convention) from Coda tokens `C_t^(l)`
    and fused `[E_bar_t, C_t^(l)]` across loop depths `l = 1 .. L_max`, plus zero-shot evaluation of
    any Cycle 3 `U4` auxiliary grounding head if present on the checkpoint.

  - **Probe 2 (Count & Post-Swap State Retention from Frozen `S_{past, t}`):**
    Uses exact dual/Woodbury standardized Ridge Regression (and optional 2-layer MLP) to decode
    cumulative pick/stage count (`PickXtimes`, `StopCube`, `SwingXtimes`, `VideoRepick`) and
    post-occlusion 3x3 target spatial bin + subgoal action stage (`ButtonUnmaskSwap`,
    `VideoUnmaskSwap`, `VideoPlaceOrder`) directly from ALL 32,768 entries of frozen `S_{past, t}`
    (`m3/m4/m5/m6/m6b`) vs `FrameSamp-32` buffer (`m2`) vs memoryless context (`m0`).

  - **Probe 3 (Offline Cross-Session Interference Stress Test, `k in {0, 1, 3}` Distractor Episodes
    + Event-Triggered Loop Allocation Profile):**
    Streams `k in {0, 1, 3}` prior distractor episodes through the policy's persistent memory
    WITHOUT resetting recurrent state before streaming each held-out evaluation episode, measuring
    Coda cosine stability `cos(C_t(k), C_t(0))`, `<x, y>` pixel localization degradation, and state
    retention degradation, alongside expected loop depth `E[L_t | event]` on subgoal transitions vs
    steady execution.

CPU-Safe Usage:
  CUDA_VISIBLE_DEVICES="" python analysis/probe_robomme_state.py --smoke-test
  CUDA_VISIBLE_DEVICES="" python analysis/probe_robomme_state.py \\
      --checkpoints checkpoints/robomme/cycle2a/m0_memoryless.pt \\
                    checkpoints/robomme/cycle2a/m2_framesamp_modul.pt \\
                    checkpoints/robomme/cycle2a/m3_gdn_b1_l1.pt \\
                    checkpoints/robomme/cycle2a/m4_gdn_b2_l1.pt \\
                    checkpoints/robomme/cycle2a/m6_deltaloop_b2.pt \\
      --shard-dir data/robomme_cache/val \\
      --tasks ButtonUnmaskSwap VideoUnmaskSwap PickXtimes StopCube \\
      --output-json results/robomme/cycle2a/probe_summary.json
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analysis.subgoal_grounding import (
    ROBOMME_IMAGE_HW,
    STAGE_VOCAB,
    extract_subgoal_metadata,
    norm_xy_to_pixel_xy,
    parse_grounded_subgoals,
)
from src.models.deltaloop_vla import (
    DeltaLoopVLAConfig,
    DeltaLoopVLAPolicy,
    run_dual_backend_gdn,
)
from robomme_eval.interface import ActionStateNormalizer, MME_VLA_NORM_STATS


# =============================================================================
# 1. Exact Dual/Primal Standardized Ridge & MLP Probes (Zero Sklearn Dependency)
# =============================================================================


class StandardizedDualRidgeRegressor:
    """Exact standardized L2 Ridge Regressor supporting `D >> N` (e.g. `D = 33,280` for `S_past`) in ms.

    Uses primal solve `(X^T X + lam I)^{-1} X^T Y` when `D <= N` and exact Woodbury / Gram dual solve
    `X^T (X X^T + lam I)^{-1} Y` when `D > N`, tuned over `lambdas` on an episode-disjoint inner split.
    """

    def __init__(
        self,
        lambdas: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0),
    ) -> None:
        self.lambdas = tuple(float(x) for x in lambdas)
        self.x_mean: Optional[torch.Tensor] = None
        self.x_std: Optional[torch.Tensor] = None
        self.y_mean: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None  # (D, C) float64
        self.best_lambda: float = 10.0

    @staticmethod
    def _precompute_gram(Xn_f32: torch.Tensor) -> torch.Tensor:
        """Compute Gram matrix in fast AVX float32 GEMM, promoted to float64 for exact Cholesky/LU solve."""
        N, D = Xn_f32.shape
        if D <= N:
            return (Xn_f32.T @ Xn_f32).to(torch.float64)
        return (Xn_f32 @ Xn_f32.T).to(torch.float64)

    @staticmethod
    def _solve_from_gram(
        Xn_f32: torch.Tensor,
        gram_f64: torch.Tensor,
        Yc_f64: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        N, D = Xn_f32.shape
        reg = float(lam) * float(max(N, 1)) / 100.0
        mat = gram_f64.clone()
        mat.diagonal().add_(reg)
        if D <= N:
            rhs = Xn_f32.to(torch.float64).T @ Yc_f64
            return torch.linalg.solve(mat, rhs).to(torch.float32)
        alpha = torch.linalg.solve(mat, Yc_f64)
        return (Xn_f32.T @ alpha.to(torch.float32)).to(torch.float32)

    def fit(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        ep_ids: Optional[torch.Tensor] = None,
    ) -> "StandardizedDualRidgeRegressor":
        Xf = X.detach().to(device="cpu", dtype=torch.float32)
        Yd = Y.detach().to(device="cpu", dtype=torch.float64)
        if Yd.ndim == 1:
            Yd = Yd.unsqueeze(-1)
        N, _ = Xf.shape
        if N == 0:
            raise ValueError("Cannot fit RidgeRegressor on empty sample set (N=0)")

        self.x_mean = Xf.mean(dim=0, keepdim=True)
        self.x_std = Xf.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
        self.y_mean = Yd.mean(dim=0, keepdim=True)

        Xn_all = (Xf - self.x_mean) / self.x_std
        Yc_all = Yd - self.y_mean

        # Inner episode-disjoint validation split (75/25) to select regularization `lambda`
        chosen_lam = self.lambdas[len(self.lambdas) // 2]
        if N >= 8 and len(self.lambdas) > 1:
            if ep_ids is not None:
                uniq_eps = torch.unique(ep_ids.cpu())
                if uniq_eps.numel() >= 3:
                    n_tr_ep = max(2, int(round(0.75 * uniq_eps.numel())))
                    tr_set = set(uniq_eps[:n_tr_ep].tolist())
                    tr_mask = torch.tensor([int(e) in tr_set for e in ep_ids.cpu().tolist()], dtype=torch.bool)
                else:
                    cut = max(4, int(0.75 * N))
                    tr_mask = torch.arange(N) < cut
            else:
                cut = max(4, int(0.75 * N))
                tr_mask = torch.arange(N) < cut

            va_mask = ~tr_mask
            if int(tr_mask.sum().item()) >= 2 and int(va_mask.sum().item()) >= 2:
                X_tr, Y_tr = Xn_all[tr_mask], Yc_all[tr_mask]
                X_va, Y_va = Xn_all[va_mask], Yc_all[va_mask]
                gram_tr = self._precompute_gram(X_tr)
                best_mse = float("inf")
                for lam in self.lambdas:
                    try:
                        w_cand = self._solve_from_gram(X_tr, gram_tr, Y_tr, lam)
                        pred_va = (X_va @ w_cand).to(torch.float64)
                        mse = float(((pred_va - Y_va) ** 2).mean().item())
                        if mse < best_mse:
                            best_mse = mse
                            chosen_lam = lam
                    except RuntimeError:
                        continue

        self.best_lambda = chosen_lam
        gram_all = self._precompute_gram(Xn_all)
        self.weight = self._solve_from_gram(Xn_all, gram_all, Yc_all, chosen_lam)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self.x_mean is not None and self.x_std is not None and self.y_mean is not None and self.weight is not None
        Xf = X.detach().to(device="cpu", dtype=torch.float32)
        Xn = (Xf - self.x_mean) / self.x_std
        pred = Xn @ self.weight + self.y_mean.to(torch.float32)
        return pred.float()


class StandardizedRidgeClassifier:
    """Multi-class One-vs-Rest Standardized Ridge Classifier (`D >> N` safe via Woodbury solve)."""

    def __init__(
        self,
        lambdas: Sequence[float] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    ) -> None:
        self.reg = StandardizedDualRidgeRegressor(lambdas=lambdas)
        self.classes: Optional[torch.Tensor] = None

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        ep_ids: Optional[torch.Tensor] = None,
    ) -> "StandardizedRidgeClassifier":
        y_cpu = y.detach().to(device="cpu", dtype=torch.long)
        uniq = torch.unique(y_cpu, sorted=True)
        self.classes = uniq
        if uniq.numel() <= 1:
            return self
        # Map labels to `0 .. K-1` one-hot targets in `{-1, +1}`
        idx = torch.searchsorted(uniq, y_cpu)
        Y_oh = F.one_hot(idx, num_classes=int(uniq.numel())).to(torch.float64) * 2.0 - 1.0
        self.reg.fit(X, Y_oh, ep_ids=ep_ids)
        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self.classes is not None
        if self.classes.numel() == 1:
            return self.classes[0].expand(X.shape[0]).clone()
        scores = self.reg.predict(X)  # (N, K)
        pred_idx = scores.argmax(dim=-1)
        return self.classes[pred_idx]


def fit_and_predict_mlp_xy(
    X_tr: torch.Tensor,
    Y_tr: torch.Tensor,
    X_te: torch.Tensor,
    *,
    hidden_dim: int = 128,
    steps: int = 200,
    lr: float = 3e-3,
    weight_decay: float = 1e-2,
    seed: int = 42,
) -> torch.Tensor:
    """Lightweight 2-layer LayerNorm+SiLU MLP regressor for nonlinear `<x, y>` decodability checks."""
    gen = torch.Generator().manual_seed(seed)
    X_train = X_tr.detach().cpu().float()
    Y_train = Y_tr.detach().cpu().float()
    X_test = X_te.detach().cpu().float()

    x_mean = X_train.mean(dim=0, keepdim=True)
    x_std = X_train.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-5)
    Xn_tr = (X_train - x_mean) / x_std
    Xn_te = (X_test - x_mean) / x_std

    D = Xn_tr.shape[1]
    C = Y_train.shape[1]
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Linear(D, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, C),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    batch_sz = min(128, Xn_tr.shape[0])
    for _ in range(steps):
        if Xn_tr.shape[0] > batch_sz:
            perm = torch.randint(0, Xn_tr.shape[0], (batch_sz,), generator=gen)
            xb, yb = Xn_tr[perm], Y_train[perm]
        else:
            xb, yb = Xn_tr, Y_train
        pred = net(xb)
        loss = F.smooth_l1_loss(pred, yb, beta=0.05)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        return net(Xn_te)


# =============================================================================
# 2. Metric Helpers (`px` Error, `R^2`, Classification Acc / Macro-F1)
# =============================================================================


def compute_xy_metrics(pred_xy_norm: torch.Tensor, target_xy_norm: torch.Tensor) -> Dict[str, float]:
    """Compute pixel localization metrics on native `256x256` RoboMME coordinates and normalized `R^2`."""
    if pred_xy_norm.shape[0] == 0:
        return {
            "n_samples": 0,
            "mean_px_err": float("nan"),
            "median_px_err": float("nan"),
            "acc_at_16px": float("nan"),
            "acc_at_32px": float("nan"),
            "r2": float("nan"),
        }
    pred_clamped = pred_xy_norm.detach().cpu().float().clamp(-1.25, 1.25)
    targ = target_xy_norm.detach().cpu().float()

    pred_px = norm_xy_to_pixel_xy(pred_clamped, image_hw=ROBOMME_IMAGE_HW)
    targ_px = norm_xy_to_pixel_xy(targ, image_hw=ROBOMME_IMAGE_HW)
    dist_px = torch.linalg.vector_norm(pred_px - targ_px, ord=2, dim=-1)

    ss_res = float(((targ - pred_clamped) ** 2).sum().item())
    targ_mean = targ.mean(dim=0, keepdim=True)
    ss_tot = float(((targ - targ_mean) ** 2).sum().item())
    r2 = 1.0 - (ss_res / max(ss_tot, 1e-8)) if targ.shape[0] >= 2 and ss_tot > 1e-7 else 0.0

    return {
        "n_samples": int(targ.shape[0]),
        "mean_px_err": float(dist_px.mean().item()),
        "median_px_err": float(dist_px.median().item()),
        "acc_at_16px": float((dist_px <= 16.0).float().mean().item() * 100.0),
        "acc_at_32px": float((dist_px <= 32.0).float().mean().item() * 100.0),
        "r2": float(r2),
    }


def compute_cls_metrics(pred_y: torch.Tensor, true_y: torch.Tensor) -> Dict[str, float]:
    """Compute classification accuracy (%), macro-F1 (%), and integer MAE."""
    if true_y.numel() == 0:
        return {"n_samples": 0, "acc_pct": float("nan"), "macro_f1_pct": float("nan"), "mae": float("nan")}
    p = pred_y.detach().cpu().long()
    t = true_y.detach().cpu().long()
    acc = float((p == t).float().mean().item() * 100.0)
    mae = float((p - t).abs().float().mean().item())

    classes = torch.unique(t)
    f1s: List[float] = []
    for c in classes.tolist():
        tp = float(((p == c) & (t == c)).sum().item())
        fp = float(((p == c) & (t != c)).sum().item())
        fn = float(((p != c) & (t == c)).sum().item())
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 1.0)
    macro_f1 = float(np.mean(f1s) * 100.0) if f1s else float("nan")
    return {
        "n_samples": int(t.numel()),
        "acc_pct": acc,
        "macro_f1_pct": macro_f1,
        "mae": mae,
    }


def xy_norm_to_3x3_bin(xy_norm: torch.Tensor) -> torch.Tensor:
    """Discretize canonical `(x_norm, y_norm) in [-1, 1]^2` into `3x3` spatial region bin `0..8`."""
    # Thresholds at `-1/3` and `+1/3`
    bx = torch.bucketize(xy_norm[..., 0].contiguous(), torch.tensor([-1.0 / 3.0, 1.0 / 3.0], device=xy_norm.device))
    by = torch.bucketize(xy_norm[..., 1].contiguous(), torch.tensor([-1.0 / 3.0, 1.0 / 3.0], device=xy_norm.device))
    return (by * 3 + bx).long()


# =============================================================================
# 3. Checkpoint Loader & Exact Causal Shard Streamer (`RoboMMEDeltaLoopPolicy` Parity)
# =============================================================================


def load_policy_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> Tuple[DeltaLoopVLAPolicy, ActionStateNormalizer, Dict[str, Any]]:
    """Load any Cycle 2A or Cycle 3 `DeltaLoopVLAPolicy` checkpoint onto `device` (`float32` on CPU)."""
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    raw_cfg = dict(payload["model_config"])
    valid_fields = {f.name for f in dataclasses.fields(DeltaLoopVLAConfig)}
    filtered_cfg = {k: v for k, v in raw_cfg.items() if k in valid_fields}
    cfg = DeltaLoopVLAConfig.from_dict(filtered_cfg)

    model = DeltaLoopVLAPolicy(cfg)
    state_dict = payload["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model.to(device=device, dtype=dtype)
    for name, p in model.named_parameters():
        if name.endswith(("A_log", "dt_bias", "gamma_ctx")):
            p.data = p.data.float()

    action_mode = payload.get("action_mode", "quantile_delta")
    norm_stats = payload.get("norm_stats", MME_VLA_NORM_STATS)
    normalizer = ActionStateNormalizer(mode=action_mode, stats=norm_stats)
    meta = {
        "ckpt_path": str(ckpt_path),
        "arch": cfg.arch,
        "step": payload.get("step", None),
        "state_bytes": model.state_size_bytes(),
        "max_loops": cfg.max_loops if model.is_multi_depth else 1,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    return model, normalizer, meta


def _call_zero_shot_grounding_xy(
    model: nn.Module,
    ctx_tokens: Optional[torch.Tensor],
    coda_tokens: Optional[torch.Tensor],
    *,
    ctx_mask: Optional[torch.Tensor] = None,
    mem_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Invoke Cycle 3 `U4` `model.predict_grounding_xy` regardless of 1-arg or 2-arg signature."""
    fn = getattr(model, "predict_grounding_xy", None)
    if fn is None:
        return None
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())
    kwargs: Dict[str, Any] = {}
    if "ctx_mask" in sig.parameters and ctx_mask is not None:
        kwargs["ctx_mask"] = ctx_mask
    if "mem_mask" in sig.parameters and mem_mask is not None:
        kwargs["mem_mask"] = mem_mask
    pos_params = [
        k
        for k, v in sig.parameters.items()
        if v.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(pos_params) <= 1:
        inp = coda_tokens if coda_tokens is not None else ctx_tokens
        if inp is None:
            return None
        return fn(inp, **kwargs)
    if params[0].startswith("coda"):
        return fn(coda_tokens, ctx_tokens, **kwargs)
    return fn(ctx_tokens, coda_tokens, **kwargs)


class ShardSessionStreamer:
    """Streams cached `{TaskName}.pt` episodes through `DeltaLoopVLAPolicy` with exact A3 & `U2/U5` parity.

    Mirrors `RoboMMEPolicyWrapper` (`src/models/robomme_policy_wrapper.py`) step-for-step across both
    Cycle 2A (`v3_mode=False`) and Cycle 3 (`v3_mode=True`: `U2` `obs_horizon2`, `U5` adaptive demo
    striding `long_demo_stride=8` when `exec_start > adaptive_demo_stride_threshold`, and `U5` visual
    motion RMS $\\beta$-gating), allowing:
      1. Clean single-episode streaming (`reset_session()` before episode) for Probes 1 & 2.
      2. Unreset multi-episode streaming (`k in {0, 1, 3}` prior distractor episodes without reset!)
         for Probe 3's Cross-Session Interference Stress Test.
    """

    def __init__(
        self,
        model: DeltaLoopVLAPolicy,
        normalizer: ActionStateNormalizer,
        device: torch.device,
        *,
        exec_horizon: Optional[int] = None,
        force_pytorch_gdn: bool = True,
    ) -> None:
        self.model = model
        self.cfg = model.cfg
        self.normalizer = normalizer
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.exec_horizon = int(exec_horizon if exec_horizon is not None else self.cfg.exec_horizon)
        self.force_pytorch_gdn = bool(force_pytorch_gdn or device.type == "cpu")

        self._frame_cursor: int = 0
        self._demo_frames_count: int = 0
        self._pending_strided: List[Tuple[int, torch.Tensor, bool, torch.Tensor]] = []
        self._pending_wb: List[Tuple[int, torch.Tensor]] = []
        self._conv_ring: List[torch.Tensor] = []
        self._prev_written_front_tok: Optional[torch.Tensor] = None

        self._S: Optional[torch.Tensor] = None
        self._occ: Optional[torch.Tensor] = None
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
        self._pending_wb.clear()
        self._conv_ring.clear()
        self._prev_written_front_tok = None

    def reset_session(self) -> None:
        self._frame_cursor = 0
        self._demo_frames_count = 0
        self._clear_intra_episode_short_caches()
        self._framesamp_committed.clear()
        self._framesamp_demo_count = 0

        H = self.cfg.gdn_heads
        K = self.cfg.gdn_d_k
        V = self.cfg.gdn_d_v
        if self.model.arch in ("m5_gdn2layer_b2_l1", "m7_deltaloop_2l_b2"):
            Vh = V // 2
            self._S1 = torch.zeros((1, H, Vh, K), device=self.device, dtype=torch.float32)
            self._occ1 = torch.zeros((1, H, K), device=self.device, dtype=torch.float32)
            self._S2 = torch.zeros((1, H, Vh, K), device=self.device, dtype=torch.float32)
            self._occ2 = torch.zeros((1, H, K), device=self.device, dtype=torch.float32)
            self._S = None
            self._occ = None
        elif self.model.uses_gdn:
            self._S = torch.zeros((1, H, V, K), device=self.device, dtype=torch.float32)
            self._occ = torch.zeros((1, H, K), device=self.device, dtype=torch.float32)
            self._S1 = self._occ1 = self._S2 = self._occ2 = None
        else:
            self._S = self._occ = self._S1 = self._occ1 = self._S2 = self._occ2 = None

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
        """Compute causal per-written-frame visual motion RMS `(1,)` matching `RoboMMEPolicyWrapper`."""
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
        front_emb: torch.Tensor,
        wrist_emb: torch.Tensor,
        state_norm: torch.Tensor,
        is_demo: torch.Tensor,
        *,
        is_demo_segment: bool = False,
    ) -> None:
        n = int(front_emb.shape[0])
        if n == 0:
            return
        if not self.model.uses_history:
            self._frame_cursor += n
            if is_demo_segment:
                self._demo_frames_count += n
            return

        assert self.model.slot_resampler is not None
        v3_mode = bool(getattr(self.cfg, "v3_mode", False))
        if v3_mode and is_demo_segment:
            total_demo = self._demo_frames_count + n
            stride = (
                int(getattr(self.cfg, "long_demo_stride", 8))
                if total_demo > int(getattr(self.cfg, "adaptive_demo_stride_threshold", 256))
                else int(self.cfg.write_stride)
            )
        else:
            stride = int(self.cfg.write_stride)

        sel_local: List[int] = []
        sel_global: List[int] = []
        for i in range(n):
            g_idx = self._frame_cursor + i
            if v3_mode and (not is_demo_segment):
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
        f_tok = front_emb[idx_t].to(device=self.device, dtype=self.dtype)
        w_tok = wrist_emb[idx_t].to(device=self.device, dtype=self.dtype)
        s_tok = state_norm[idx_t].to(device=self.device, dtype=self.dtype)
        d_tok = is_demo[idx_t].to(device=self.device, dtype=torch.long)
        step_t = torch.tensor(sel_global, device=self.device, dtype=torch.long)

        raw_slots = self.model.slot_resampler.encode_frames_raw(f_tok, w_tok, s_tok, step_t, d_tok)
        d_list = is_demo[idx_t].tolist()
        for k_i, g_idx in enumerate(sel_global):
            m_val = self._compute_next_motion_rms(f_tok[k_i])
            self._pending_strided.append((g_idx, raw_slots[k_i], bool(d_list[k_i]), m_val))

    def _flush_eligible(self, cutoff_global_excl: int) -> None:
        if not self.model.uses_history:
            self._pending_strided.clear()
            self._pending_wb.clear()
            return

        to_commit: List[Tuple[int, torch.Tensor, bool, torch.Tensor]] = []
        keep_p: List[Tuple[int, torch.Tensor, bool, torch.Tensor]] = []
        for item in self._pending_strided:
            if item[0] < cutoff_global_excl:
                to_commit.append(item)
            else:
                keep_p.append(item)
        self._pending_strided = keep_p

        wb_commit: List[Tuple[int, torch.Tensor]] = []
        wb_keep: List[Tuple[int, torch.Tensor]] = []
        for wb in self._pending_wb:
            if wb[0] < cutoff_global_excl:
                wb_commit.append(wb)
            else:
                wb_keep.append(wb)
        self._pending_wb = wb_keep

        if not to_commit and not wb_commit:
            return

        if self.model.arch == "m2_framesamp_modul":
            for _, rs, is_d, _ in to_commit:
                cs = self._stream_slot_conv_single(rs)
                self._framesamp_committed.append(cs.view(-1))
                if is_d:
                    self._framesamp_demo_count += 1
            return

        if self.model.arch in ("m5_gdn2layer_b2_l1", "m7_deltaloop_2l_b2"):
            if not to_commit:
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
            return

        assert self.model.gdn is not None
        if not wb_commit:
            z_frames = torch.cat([self._stream_slot_conv_single(rs) for _, rs, _, _ in to_commit], dim=1)
            m_frames = torch.stack([mv for _, _, _, mv in to_commit], dim=0).unsqueeze(0)
            k, v, g, b, _ = self.model.gdn.project_frame_slots(z_frames, motion_feat=m_frames)
            _, self._S, self._occ = run_dual_backend_gdn(
                k=k,
                v=v,
                g_fp32=g,
                beta_fp32=b,
                initial_state_fp32=self._S,
                initial_occ_fp32=self._occ,
                track_occ=True,
                force_pytorch=self.force_pytorch_gdn,
            )
            return

        events: List[Tuple[int, int, torch.Tensor, Optional[torch.Tensor]]] = []
        for f_idx, rs, _, mv in to_commit:
            events.append((f_idx, 0, self._stream_slot_conv_single(rs), mv.view(1, 1, 1)))
        for wb_idx, wb_s in wb_commit:
            events.append((wb_idx, 1, wb_s, None))
        events.sort(key=lambda t: (t[0], t[1]))
        ks, vs, gs, bs = [], [], [], []
        for _, kind, item_t, mv_opt in events:
            if kind == 0:
                ki, vi, gi, bi, _ = self.model.gdn.project_frame_slots(item_t, motion_feat=mv_opt)
            else:
                ki, vi, gi, bi = self.model.gdn.project_writeback_slot(item_t)
            ks.append(ki)
            vs.append(vi)
            gs.append(gi)
            bs.append(bi)
        _, self._S, self._occ = run_dual_backend_gdn(
            k=torch.cat(ks, dim=1),
            v=torch.cat(vs, dim=1),
            g_fp32=torch.cat(gs, dim=1),
            beta_fp32=torch.cat(bs, dim=1),
            initial_state_fp32=self._S,
            initial_occ_fp32=self._occ,
            track_occ=True,
            force_pytorch=self.force_pytorch_gdn,
        )

    def _current_mem_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self.model.uses_history:
            return None
        if self.model.arch == "m2_framesamp_modul":
            B_sz = self.cfg.framesamp_buffer_size
            D = self.cfg.d_model
            buf_tok = torch.zeros((1, B_sz, D), device=self.device, dtype=self.dtype)
            buf_mask = torch.zeros((1, B_sz), device=self.device, dtype=torch.bool)
            n_avail = len(self._framesamp_committed)
            if n_avail > 0:
                sel_idx, n_sel = self.model.select_framesamp_slot_indices(
                    n_avail, self._framesamp_demo_count, budget=B_sz, device=self.device
                )
                stacked = torch.stack(self._framesamp_committed, dim=0).to(self.dtype)
                buf_tok[0, :n_sel] = stacked[sel_idx]
                buf_mask[0, :n_sel] = True
            return {"buf_tokens": buf_tok, "buf_mask": buf_mask}

        if self.model.arch in ("m5_gdn2layer_b2_l1", "m7_deltaloop_2l_b2"):
            assert self._S1 is not None and self._S2 is not None
            return {"S1": self._S1, "occ1": self._occ1, "S2": self._S2, "occ2": self._occ2}  # type: ignore[dict-item]

        assert self._S is not None
        return {"S": self._S, "occ": self._occ}  # type: ignore[dict-item]

    def _flatten_state_vector(self, mem_state: Optional[Dict[str, torch.Tensor]], ctx_pool: torch.Tensor) -> torch.Tensor:
        """Extract exact flattened `S_{past, t}` (or `buf_tokens` for `m2`, `ctx_pool` fallback for `m0`)."""
        if mem_state is None:
            return ctx_pool.detach().cpu().float().reshape(-1)
        if "S" in mem_state:
            parts = [mem_state["S"].detach().cpu().float().reshape(-1)]
            if mem_state.get("occ") is not None:
                parts.append(mem_state["occ"].detach().cpu().float().reshape(-1))
            return torch.cat(parts, dim=0)
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
            # Flush any trailing pending execution frames from the previous distractor episode
            # and clear ONLY short-term intra-episode caches (`eval_cycle3.py` exact parity!)
            self._flush_eligible(cutoff_global_excl=self._frame_cursor)
            self._clear_intra_episode_short_caches()

        ep_base_global = self._frame_cursor
        self._demo_frames_count = 0
        s = int(shard["ep_offsets"][ep_local].item())
        e = int(shard["ep_offsets"][ep_local + 1].item())
        ep_len = e - s
        exec_start = int(shard["exec_start_idx"][ep_local].item())

        front_ep = shard["front_emb"][s:e]
        wrist_ep = shard["wrist_emb"][s:e]
        raw_state_np = shard["state"][s:e].float().numpy()
        state_norm_ep = torch.from_numpy(self.normalizer.normalize_state(raw_state_np)).float()
        is_demo_ep = shard["is_demo"][s:e].bool()

        # If we are ONLY streaming a distractor episode (`collect_features=False`), and the model is
        # NOT write-back `m6b`, we can ingest demo + execution segments cleanly!
        if not collect_features and self.model.arch != "m6b_deltaloop_wb":
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

        text_tok = shard["text_token_emb"][ep_local].unsqueeze(0).to(device=self.device, dtype=self.dtype)
        text_mask = shard["text_attn_mask"][ep_local].unsqueeze(0).to(device=self.device, dtype=torch.bool)

        chunk_steps = np.arange(exec_start, ep_len, self.exec_horizon, dtype=np.int64)
        if chunk_steps.size == 0:
            chunk_steps = np.array([min(exec_start, max(0, ep_len - 1))], dtype=np.int64)
        M_ch = int(chunk_steps.size)
        c_idx_cpu = torch.from_numpy(chunk_steps).long()

        # Precompute `obs_encoder` for ALL `M_ch` chunks in a single batched forward pass (with `U2` `obs_horizon2`!)
        f_batch = front_ep[c_idx_cpu].to(device=self.device, dtype=self.dtype)
        w_batch = wrist_ep[c_idx_cpu].to(device=self.device, dtype=self.dtype)
        s_batch = state_norm_ep[c_idx_cpu].to(device=self.device, dtype=self.dtype)
        if getattr(self.cfg, "obs_horizon2", False):
            prev_steps = np.maximum(exec_start, chunk_steps - 1)
            p_idx_cpu = torch.from_numpy(prev_steps).long()
            pf_batch = front_ep[p_idx_cpu].to(device=self.device, dtype=self.dtype)
            pw_batch = wrist_ep[p_idx_cpu].to(device=self.device, dtype=self.dtype)
            ps_batch = state_norm_ep[p_idx_cpu].to(device=self.device, dtype=self.dtype)
            ctx_tokens_all, ctx_mask_all = self.model.obs_encoder(
                f_batch,
                w_batch,
                s_batch,
                text_tok.expand(M_ch, -1, -1),
                text_mask.expand(M_ch, -1),
                prev_front_emb=pf_batch,
                prev_wrist_emb=pw_batch,
                prev_state_norm=ps_batch,
            )
        else:
            ctx_tokens_all, ctx_mask_all = self.model.obs_encoder(
                f_batch,
                w_batch,
                s_batch,
                text_tok.expand(M_ch, -1, -1),
                text_mask.expand(M_ch, -1),
            )
        m_f_all = ctx_mask_all.float().unsqueeze(-1)
        ctx_pool_all = (ctx_tokens_all.float() * m_f_all).sum(dim=1) / m_f_all.sum(dim=1).clamp(min=1.0)

        # Ingest conditioning video demo `[0 : exec_start]` first (adaptive `long_demo_stride=8` when `exec_start > threshold`)
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
        coda_by_depth: List[List[torch.Tensor]] = [[] for _ in range(L_eval)]
        zero_shot_xy_by_depth: List[List[torch.Tensor]] = [[] for _ in range(L_eval)]
        has_zero_shot_u4 = False

        halt_expected_L: List[float] = []
        halt_gate_L: List[float] = []
        coda_conv_L: List[float] = []
        coda_rel_drift: List[float] = []

        # Fast batched Coda readout path for non-writeback models (`m0..m7`), exact sequential path for `m6b`
        can_batch_coda = self.model.arch != "m6b_deltaloop_wb"
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
            if collect_features:
                state_vecs.append(self._flatten_state_vector(mem_state, ctx_pools[m_idx]))

            if can_batch_coda:
                if mem_state is None:
                    collected_mem_states.append(None)
                else:
                    collected_mem_states.append({k: v.clone() for k, v in mem_state.items() if v is not None})
                ro_step = None
            else:
                ctx_tokens = ctx_tokens_all[m_idx : m_idx + 1]
                ctx_mask = ctx_mask_all[m_idx : m_idx + 1]
                ro_step = self.model.run_coda_readout(
                    ctx_tokens,
                    ctx_mask,
                    mem_state,
                    num_loops=L_eval,
                    halt_mode="fixed",
                )

            # Ingest current frame `t_step` + optional `m6b` write-back slot before advancing!
            self._queue_frame_range(
                front_ep[t_step : t_step + 1],
                wrist_ep[t_step : t_step + 1],
                state_norm_ep[t_step : t_step + 1],
                is_demo_ep[t_step : t_step + 1],
                is_demo_segment=False,
            )
            local_ingested = t_step + 1
            if (
                ro_step is not None
                and self.model.arch == "m6b_deltaloop_wb"
                and self.model.wb_proj is not None
                and ro_step["coda"] is not None
            ):
                wb_vec = self.model.wb_proj(ro_step["coda"].mean(dim=1)).unsqueeze(1)
                self._pending_wb.append((ep_base_global + t_step, wb_vec))

            if (not can_batch_coda) and collect_features and ro_step is not None:
                coda_list = ro_step["coda_per_depth"]
                ro_mem_mask = ro_step.get("mem_mask")
                for l_i in range(L_eval):
                    c_tok = coda_list[l_i]
                    c_flat = ctx_pools[m_idx] if c_tok is None else c_tok[0].detach().cpu().float().reshape(-1)
                    coda_by_depth[l_i].append(c_flat)
                    pxy_step = _call_zero_shot_grounding_xy(
                        self.model,
                        ctx_tokens_all[m_idx : m_idx + 1],
                        c_tok,
                        ctx_mask=ctx_mask_all[m_idx : m_idx + 1],
                        mem_mask=ro_mem_mask,
                    )
                    if pxy_step is not None:
                        has_zero_shot_u4 = True
                        zero_shot_xy_by_depth[l_i].append(pxy_step[0].detach().cpu().float())

                halt_dict = ro_step.get("halt")
                if halt_dict is not None and "p_tilde" in halt_dict and L_eval > 1:
                    probs = halt_dict["p_tilde"][0].detach().cpu().float()
                    depths = torch.arange(1, L_eval + 1, dtype=torch.float32)
                    exp_l = float((probs * depths).sum().item())
                    cdf = torch.cumsum(probs, dim=0)
                    hit = cdf >= 0.5
                    hit[-1] = True
                    gate_l = float(int(torch.argmax(hit.int()).item()) + 1)
                else:
                    exp_l = float(L_eval)
                    gate_l = float(L_eval)

                if L_eval > 1 and coda_list[0] is not None and coda_list[-1] is not None:
                    c1 = coda_list[0][0].detach().cpu().float()
                    cL = coda_list[-1][0].detach().cpu().float()
                    drift = float(
                        (
                            torch.linalg.vector_norm(cL - c1, ord=2, dim=-1).mean()
                            / (torch.linalg.vector_norm(c1, ord=2, dim=-1).mean() + 1e-5)
                        ).item()
                    )
                    conv_depth = L_eval
                    for l_i in range(1, L_eval):
                        ci = coda_list[l_i][0].detach().cpu().float()
                        cp = coda_list[l_i - 1][0].detach().cpu().float()
                        rd = float(
                            (
                                torch.linalg.vector_norm(ci - cp, ord=2, dim=-1).mean()
                                / (torch.linalg.vector_norm(cp, ord=2, dim=-1).mean() + 1e-5)
                            ).item()
                        )
                        if rd < 0.08:
                            conv_depth = l_i + 1
                            break
                else:
                    drift = 0.0
                    conv_depth = 1

                halt_expected_L.append(exp_l)
                halt_gate_L.append(gate_l)
                coda_conv_L.append(float(conv_depth))
                coda_rel_drift.append(drift)

        if can_batch_coda and collect_features:
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
                batched_mem,
                num_loops=L_eval,
                halt_mode="fixed",
            )
            coda_list_all = ro_all["coda_per_depth"]
            ro_mem_mask_all = ro_all.get("mem_mask")
            for l_i in range(L_eval):
                c_tok_m = coda_list_all[l_i]
                if c_tok_m is None:
                    coda_by_depth[l_i] = list(ctx_pools)
                else:
                    c_cpu = c_tok_m.detach().cpu().float().reshape(M_ch, -1)
                    coda_by_depth[l_i] = [c_cpu[i] for i in range(M_ch)]
                pxy_all = _call_zero_shot_grounding_xy(
                    self.model,
                    ctx_tokens_all,
                    c_tok_m,
                    ctx_mask=ctx_mask_all,
                    mem_mask=ro_mem_mask_all,
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

            if L_eval > 1 and coda_list_all[0] is not None and coda_list_all[-1] is not None:
                c_stack = torch.stack([c.detach().cpu().float() for c in coda_list_all], dim=1)  # (M_ch, L, N_c, d)
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

        # Ingest any remaining tail frames `[local_ingested : ep_len]` so cross-session state is complete
        if local_ingested < ep_len:
            self._queue_frame_range(
                front_ep[local_ingested:ep_len],
                wrist_ep[local_ingested:ep_len],
                state_norm_ep[local_ingested:ep_len],
                is_demo_ep[local_ingested:ep_len],
                is_demo_segment=False,
            )
        self._flush_eligible(cutoff_global_excl=self._frame_cursor)

        if not collect_features:
            return None

        # Extract per-chunk ground-truth subgoal labels at `chunk_steps`
        grounded_ep = shard["grounded_subgoals"][ep_local]
        simple_ep = shard["simple_subgoals"][ep_local]
        g_parsed = parse_grounded_subgoals(grounded_ep, resolve_complete=False)
        s_meta = extract_subgoal_metadata([simple_ep], exec_start_idx=[exec_start])

        c_idx_t = torch.from_numpy(chunk_steps).long()
        # Check if a subgoal transition occurred inside `[max(exec_start, t - exec_horizon // 2) .. min(ep_len, t + exec_horizon)]`
        trans_frame_mask = s_meta["is_transition"]
        chunk_has_trans = torch.zeros(( len(chunk_steps),), dtype=torch.bool)
        for m_i, t_s in enumerate(chunk_steps.tolist()):
            lo = max(exec_start, int(t_s) - 4)
            hi = min(ep_len, int(t_s) + self.exec_horizon)
            chunk_has_trans[m_i] = bool(trans_frame_mask[lo:hi].any().item())

        out: Dict[str, Any] = {
            "ep_local": ep_local,
            "exec_start": exec_start,
            "chunk_steps": c_idx_t,
            "elapsed_exec_steps": c_idx_t - exec_start,
            "ctx_pool": torch.stack(ctx_pools, dim=0),            # (M, d)
            "state_vec": torch.stack(state_vecs, dim=0),          # (M, D_state)
            "coda_by_depth": [torch.stack(coda_by_depth[l], dim=0) for l in range(L_eval)],
            "xy_norm": g_parsed["xy_norm"][c_idx_t],              # (M, 2)
            "has_xy": g_parsed["has_xy"][c_idx_t],                # (M,)
            "spatial_bin_3x3": xy_norm_to_3x3_bin(g_parsed["xy_norm"][c_idx_t]),
            "ordinal_index": s_meta["ordinal_index"][c_idx_t],    # (M,)
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
# 4. Probe 1, Probe 2, & Probe 3 Suite Runner per `(Checkpoint, Task Shard)`
# =============================================================================


def _concat_ep_records(records: Sequence[Dict[str, Any]], L_eval: int) -> Dict[str, Any]:
    ep_ids = torch.cat(
        [torch.full((r["ctx_pool"].shape[0],), int(r["ep_local"]), dtype=torch.long) for r in records],
        dim=0,
    )
    out: Dict[str, Any] = {
        "ep_ids": ep_ids,
        "elapsed_exec_steps": torch.cat([r["elapsed_exec_steps"] for r in records], dim=0),
        "ctx_pool": torch.cat([r["ctx_pool"] for r in records], dim=0),
        "state_vec": torch.cat([r["state_vec"] for r in records], dim=0),
        "coda_by_depth": [torch.cat([r["coda_by_depth"][l] for r in records], dim=0) for l in range(L_eval)],
        "xy_norm": torch.cat([r["xy_norm"] for r in records], dim=0),
        "has_xy": torch.cat([r["has_xy"] for r in records], dim=0),
        "spatial_bin_3x3": torch.cat([r["spatial_bin_3x3"] for r in records], dim=0),
        "ordinal_index": torch.cat([r["ordinal_index"] for r in records], dim=0),
        "completed_picks": torch.cat([r["completed_picks"] for r in records], dim=0),
        "completed_transitions": torch.cat([r["completed_transitions"] for r in records], dim=0),
        "stage_id": torch.cat([r["stage_id"] for r in records], dim=0),
        "color_id": torch.cat([r["color_id"] for r in records], dim=0),
        "chunk_has_transition": torch.cat([r["chunk_has_transition"] for r in records], dim=0),
        "halt_expected_L": torch.cat([r["halt_expected_L"] for r in records], dim=0),
        "halt_gate_L": torch.cat([r["halt_gate_L"] for r in records], dim=0),
        "coda_conv_L": torch.cat([r["coda_conv_L"] for r in records], dim=0),
        "coda_rel_drift": torch.cat([r["coda_rel_drift"] for r in records], dim=0),
    }
    if all("zero_shot_xy_by_depth" in r for r in records):
        out["zero_shot_xy_by_depth"] = [
            torch.cat([r["zero_shot_xy_by_depth"][l] for r in records], dim=0) for l in range(L_eval)
        ]
    return out


def evaluate_probes_on_shard(
    model: DeltaLoopVLAPolicy,
    normalizer: ActionStateNormalizer,
    shard: Dict[str, Any],
    device: torch.device,
    *,
    train_frac: float = 0.8,
    max_episodes: Optional[int] = None,
    distractor_ks: Sequence[int] = (0, 1, 3),
    run_mlp_probe1: bool = True,
) -> Dict[str, Any]:
    """Execute Probes 1, 2, and 3 on a single `{TaskName}.pt` shard with strict episode-disjoint splits."""
    total_eps = int(shard["ep_offsets"].numel() - 1)
    n_use = min(total_eps, int(max_episodes)) if max_episodes is not None else total_eps
    if n_use < 2:
        raise ValueError(f"Need at least 2 episodes for episode-disjoint probing, got {n_use}")

    n_train = max(1, min(n_use - 1, int(round(n_use * float(train_frac)))))
    train_ep_indices = list(range(0, n_train))
    eval_ep_indices = list(range(n_train, n_use))

    streamer = ShardSessionStreamer(model, normalizer, device=device)
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

        # Memoryless context-only baseline (`E_bar_t`)
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
        # Hop gain summary (`l=1 -> l=L_eval` reduction in pixel error)
        if L_eval > 1:
            px_l1 = probe1_out["coda_only_by_depth"]["l=1"]["mean_px_err"]
            px_lmax = probe1_out["coda_only_by_depth"][f"l={L_eval}"]["mean_px_err"]
            probe1_out["coda_hop_px_improvement_l1_to_lmax"] = float(px_l1 - px_lmax)

    # -------------------------------------------------------------------------
    # PROBE 2: Count & Post-Swap State Retention from Frozen `S_{past, t}`
    # -------------------------------------------------------------------------
    probe2_out: Dict[str, Any] = {
        "state_feature_dim": int(tr["state_vec"].shape[1]),
    }
    fitted_state_count_clf: Optional[StandardizedRidgeClassifier] = None
    fitted_state_bin_clf: Optional[StandardizedRidgeClassifier] = None
    count_label_key: str = "completed_picks"

    # Choose the richest count label present in the shard (`completed_picks` if >= 2 classes, else `completed_transitions`)
    if torch.unique(tr["completed_picks"]).numel() >= 2:
        count_label_key = "completed_picks"
    else:
        count_label_key = "completed_transitions"
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

        # Horizon-stratified count retention (`early < 32`, `mid 32..95`, `late >= 96` steps)
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

    # 2B: Post-occlusion 3x3 target spatial bin & subgoal stage directly from `S_{past, t}` (excluding last 16 frames!)
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
        # Continuous `<x, y>` decoded purely from `S_{past, t}` (without current frame `E_t`!)
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
    # PROBE 3: Cross-Session Interference (`k in {0, 1, 3}`) + Event Loop Profile
    # -------------------------------------------------------------------------
    # 3A: Event-triggered loop allocation profile on clean `k=0` eval episodes
    trans_m = te["chunk_has_transition"]
    onset_m = te["elapsed_exec_steps"] < (2 * streamer.exec_horizon)
    steady_m = (~trans_m) & (~onset_m)

    def _profile_slice(mask: torch.Tensor) -> Dict[str, float]:
        if int(mask.sum().item()) == 0:
            return {"n_chunks": 0, "mean_expected_L": float("nan"), "mean_gate_L": float("nan"), "mean_coda_conv_L": float("nan"), "mean_coda_drift": float("nan")}
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

    # 3B: Cross-session distractor streaming (`k in distractor_ks`)
    cross_session_by_k: Dict[str, Any] = {}
    coda_k0_ref = te["coda_by_depth"][-1]  # (M_eval, D_coda)

    for k_val in distractor_ks:
        k_int = int(k_val)
        if k_int == 0 or (not model.uses_history):
            te_k = te
        else:
            eval_recs_k: List[Dict[str, Any]] = []
            for eval_pos, ep_target in enumerate(eval_ep_indices):
                streamer.reset_session()
                # Deterministically pick `k_int` distinct-if-possible distractor episodes from `train_ep_indices`
                for d_step in range(k_int):
                    dist_ep = train_ep_indices[(eval_pos * 7 + d_step * 3) % len(train_ep_indices)]
                    streamer.stream_episode(
                        shard, dist_ep, reset_before=False, collect_features=False
                    )
                # Now stream target episode WITHOUT resetting persistent memory!
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

    # Compute k=0 -> k=max degradation deltas cleanly
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
# 5. Built-In Synthetic RoboMME Shard Builder for Local CPU `--smoke-test`
# =============================================================================


def build_synthetic_robomme_shard(
    task_name: str,
    *,
    num_episodes: int = 12,
    seed: int = 123,
) -> Dict[str, Any]:
    """Build an in-memory `{TaskName}.pt`-compatible shard dict with realistic `<row_y, col_x>` & count labels."""
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    # Fixed orthogonal basis vectors in 768-D SigLIP space so target `<x, y>` and count signals
    # are genuinely recoverable from demo/past frames!
    basis = torch.randn((8, 768), generator=gen, dtype=torch.float32) * 0.25

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
        # Include >256-frame demo episodes to exercise `U5` adaptive `long_demo_stride=8`!
        demo_len = 264 if (ep % 5 == 0) else 36
        exec_len = 80
        T = demo_len + exec_len
        exec_starts.append(demo_len)
        epis_idx.append(ep)

        # Sample post-swap target pixel `(col_x, row_y)` in `[32, 224)^2`
        col_x = int(rng.integers(32, 224))
        row_y = int(rng.integers(32, 224))
        x_norm = ((col_x + 0.5) / 256.0) * 2.0 - 1.0
        y_norm = ((row_y + 0.5) / 256.0) * 2.0 - 1.0
        color = colors[ep % len(colors)]

        f_ep = torch.randn((T, 64, 768), generator=gen, dtype=torch.float32) * 0.08
        w_ep = torch.randn((T, 16, 768), generator=gen, dtype=torch.float32) * 0.08
        # Embed target `(x_norm, y_norm)` strongly in demo frames `[0 : demo_len]` and weakly in execution!
        xy_sig = x_norm * basis[0] + y_norm * basis[1]  # (768,)
        f_ep[:demo_len] += xy_sig.view(1, 1, 768) * 1.5
        f_ep[demo_len:] += xy_sig.view(1, 1, 768) * 0.35

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
                stage_idx = min(2, rel // 27)  # 0, 1, 2 across execution
                # Inject cumulative stage signal into history tokens + gripper state
                f_ep[t] += float(stage_idx + 1) * basis[2 + stage_idx].view(1, 768)
                st_ep[t, 0] = x_norm * 0.2
                st_ep[t, 1] = y_norm * 0.2
                st_ep[t, 7] = 1.0 if (rel % 27) < 14 else -1.0

                if task_name == "PickXtimes":
                    ord_word = ordinals[stage_idx]
                    if (rel % 27) < 18:
                        ep_simple.append(f"pick up the {color} cube for the {ord_word} time")
                        # Remember RoboMME string format is `<row_y, col_x>`!
                        ep_grounded.append(
                            f"pick up the {color} cube at <{row_y}, {col_x}> for the {ord_word} time"
                        )
                    else:
                        ep_simple.append("put down")
                        ep_grounded.append("put down")
                else:
                    # ButtonUnmaskSwap style
                    if rel < 20:
                        btn_y, btn_x = 60, 128
                        ep_simple.append("press the button to unmask")
                        ep_grounded.append(f"press the first button at <{btn_y}, {btn_x}>")
                    elif rel < 68:
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
        prompts.append(f"solve {task_name} for the {color} cube")
        simple_sgs.append(ep_simple)
        grounded_sgs.append(ep_grounded)

    text_tok = torch.randn((num_episodes, 64, 768), generator=gen, dtype=torch.bfloat16) * 0.1
    text_pool = text_tok.float().mean(dim=1).to(torch.bfloat16)
    text_mask = torch.ones((num_episodes, 64), dtype=torch.bool)
    text_mask[:, 16:] = False

    return {
        "task_name": task_name,
        "suite": "SyntheticSmoke",
        "front_emb": torch.cat(front_list, dim=0),
        "wrist_emb": torch.cat(wrist_list, dim=0),
        "state": torch.cat(state_list, dim=0),
        "actions": torch.cat(actions_list, dim=0),
        "is_demo": torch.cat(is_demo_list, dim=0),
        "ep_offsets": torch.tensor(ep_offsets, dtype=torch.long),
        "exec_start_idx": torch.tensor(exec_starts, dtype=torch.long),
        "epis_idx": torch.tensor(epis_idx, dtype=torch.long),
        "prompts": prompts,
        "text_token_emb": text_tok,
        "text_pool_emb": text_pool,
        "text_attn_mask": text_mask,
        "simple_subgoals": simple_sgs,
        "grounded_subgoals": grounded_sgs,
    }


# =============================================================================
# 6. Summary Table Printer & CLI Entrypoint
# =============================================================================


def print_probe_summary_table(results: Dict[str, Any]) -> None:
    """Print a concise, publication-ready summary of Probes 1, 2, and 3 across models & tasks."""
    print("\n" + "=" * 112)
    print(
        f"{'Model / Arch':<24} | {'Task':<17} | {'P1 Coda l=1->L (px)':<21} | "
        f"{'P2 S_past Cnt / Bin%':<21} | {'P3 cos(k=3) / +px':<20}"
    )
    print("-" * 112)

    for ckpt_key, ckpt_data in results["models"].items():
        arch = ckpt_data["meta"]["arch"]
        L_max = ckpt_data["meta"]["max_loops"]
        for task_name, t_res in ckpt_data["tasks"].items():
            p1 = t_res["probe1_xy_localization"]
            coda_d = p1.get("coda_only_by_depth", {})
            if "l=1" in coda_d:
                px1 = coda_d["l=1"]["mean_px_err"]
                pxL = coda_d[f"l={L_max}"]["mean_px_err"]
                p1_str = f"{px1:5.1f} -> {pxL:5.1f} px" if L_max > 1 else f"{px1:5.1f} px (L=1)"
            else:
                p1_str = "n/a"

            p2 = t_res["probe2_state_retention"]
            cnt_acc = p2.get("count_from_S_past", {}).get("acc_pct", float("nan"))
            bin_acc = p2.get("target_3x3_bin_from_S_past", {}).get("acc_pct", float("nan"))
            p2_str = f"{cnt_acc:5.1f}% / {bin_acc:5.1f}%"

            p3_k = t_res["probe3_cross_session_and_halting"]["by_distractor_k"]
            p3_s = t_res["probe3_cross_session_and_halting"]["summary"]
            cos3 = p3_k.get("k=3", p3_k.get("k=1", {})).get("coda_cos_sim_vs_k0", float("nan"))
            dpx = next((v for k, v in p3_s.items() if k.startswith("px_err_increase")), float("nan"))
            p3_str = f"{cos3:5.3f} / {dpx:+5.1f}px"

            label = ckpt_key[:24]
            print(f"{label:<24} | {task_name:<17} | {p1_str:<21} | {p2_str:<21} | {p3_str:<20}")
    print("=" * 112 + "\n")


def run_smoke_test(ckpt_dir: Path, output_json: Path) -> None:
    """Smoke-test Probes 1-3 on CPU using synthetic shards + all synced Cycle 2A checkpoints."""
    device = torch.device("cpu")
    t0 = time.time()

    candidate_names = [
        "m0_memoryless.pt",
        "m1_rdvla_loop4.pt",
        "m2_framesamp_modul.pt",
        "m3_gdn_b1_l1.pt",
        "m4_gdn_b2_l1.pt",
        "m5_gdn2layer_b2_l1.pt",
        "m6_deltaloop_b2.pt",
        "m6b_deltaloop_wb.pt",
        "m7_deltaloop_2l_b2.pt",
    ]
    ckpt_paths = [ckpt_dir / name for name in candidate_names if (ckpt_dir / name).is_file()]
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    shards = {
        "ButtonUnmaskSwap": build_synthetic_robomme_shard("ButtonUnmaskSwap", num_episodes=10, seed=101),
        "PickXtimes": build_synthetic_robomme_shard("PickXtimes", num_episodes=10, seed=202),
    }

    all_results: Dict[str, Any] = {
        "mode": "cpu_smoke_test",
        "device": "cpu",
        "models": {},
    }

    for cp in ckpt_paths:
        model, normalizer, meta = load_policy_checkpoint(cp, device=device)
        key = cp.stem
        task_out: Dict[str, Any] = {}
        for task_name, shard in shards.items():
            res = evaluate_probes_on_shard(
                model,
                normalizer,
                shard,
                device=device,
                train_frac=0.7,
                distractor_ks=(0, 1, 3),
                run_mlp_probe1=True,
            )
            # Sanity assertions on finite outputs & expected loop keys
            p1_coda = res["probe1_xy_localization"]["coda_only_by_depth"]
            assert len(p1_coda) == meta["max_loops"], f"Expected {meta['max_loops']} loop depths for {key}"
            for d_metrics in p1_coda.values():
                assert math.isfinite(d_metrics["mean_px_err"]), f"Non-finite px err for {key}"
            p3_k3 = res["probe3_cross_session_and_halting"]["by_distractor_k"]["k=3"]
            assert math.isfinite(p3_k3["coda_cos_sim_vs_k0"]), f"Non-finite k=3 cos sim for {key}"
            task_out[task_name] = res

        all_results["models"][key] = {"meta": meta, "tasks": task_out}

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(all_results, indent=2))
    elapsed = time.time() - t0
    print_probe_summary_table(all_results)
    print(
        f"[PASS] Smoke-tested {len(ckpt_paths)} real checkpoints ({[p.stem for p in ckpt_paths]}) "
        f"across {list(shards.keys())} on CPU in {elapsed:.2f}s -> {output_json}"
    )


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeltaLoop-VLA Mechanistic State & Multi-Hop Loop Probe Suite")
    p.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        default=[],
        help="One or more `.pt` policy checkpoints (`m0..m6b`)",
    )
    p.add_argument(
        "--shard-dir",
        type=str,
        default="",
        help="Directory containing `{TaskName}.pt` cached RoboMME shards",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "ButtonUnmaskSwap",
            "VideoUnmaskSwap",
            "PickXtimes",
            "StopCube",
            "VideoPlaceOrder",
            "SwingXtimes",
            "VideoRepick",
        ],
        help="Task names corresponding to `{TaskName}.pt` inside `--shard-dir`",
    )
    p.add_argument("--train-frac", type=float, default=0.8, help="Episode fraction used to fit probes (default 0.8 -> 0..79 train, 80..99 eval)")
    p.add_argument("--max-episodes", type=int, default=None, help="Optional cap on episodes per shard")
    p.add_argument("--distractor-ks", nargs="+", type=int, default=[0, 1, 3], help="Prior distractor episode counts `k` for Probe 3")
    p.add_argument("--no-mlp", action="store_true", help="Skip 2-layer MLP check in Probe 1 (Ridge only)")
    p.add_argument("--device", type=str, default="cpu", help="Compute device (`cpu` or `cuda`)")
    p.add_argument("--cpu-threads", type=int, default=4, help="Max PyTorch CPU threads when `--device cpu`")
    p.add_argument("--output-json", type=str, default="", help="Path to write structured JSON results")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run self-contained CPU smoke test using synthetic shards + synced `checkpoints/robomme/cycle2a/*.pt`",
    )
    p.add_argument(
        "--smoke-ckpt-dir",
        type=str,
        default=str(REPO_ROOT / "checkpoints" / "robomme" / "cycle2a"),
        help="Checkpoint directory for `--smoke-test`",
    )
    return p


def main() -> None:
    args = _build_cli().parse_args()
    if args.device == "cpu" or args.smoke_test:
        torch.set_num_threads(max(1, min(int(args.cpu_threads), os.cpu_count() or 4)))
    if args.smoke_test:
        out_path = Path(args.output_json) if args.output_json else Path("/tmp/probe_robomme_smoke.json")
        run_smoke_test(Path(args.smoke_ckpt_dir), out_path)
        return

    if not args.checkpoints or not args.shard_dir:
        raise SystemExit("Provide `--checkpoints ... --shard-dir ...` or run with `--smoke-test`.")

    device = torch.device(args.device)
    shard_dir = Path(args.shard_dir)
    loaded_shards: Dict[str, Dict[str, Any]] = {}
    for task_name in args.tasks:
        sp = shard_dir / f"{task_name}.pt"
        if sp.is_file():
            loaded_shards[task_name] = torch.load(str(sp), map_location="cpu", weights_only=False)
        else:
            print(f"[WARN] Shard not found, skipping: {sp}")

    if not loaded_shards:
        raise FileNotFoundError(f"None of requested tasks {args.tasks} found in {shard_dir}")

    results: Dict[str, Any] = {
        "mode": "robomme_cached_shards",
        "shard_dir": str(shard_dir),
        "device": str(device),
        "train_frac": args.train_frac,
        "distractor_ks": list(args.distractor_ks),
        "models": {},
    }

    for cp_str in args.checkpoints:
        cp = Path(cp_str)
        t_m = time.time()
        model, normalizer, meta = load_policy_checkpoint(cp, device=device)
        key = cp.stem
        task_map: Dict[str, Any] = {}
        for task_name, shard in loaded_shards.items():
            task_map[task_name] = evaluate_probes_on_shard(
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
        print(f"[DONE] Saved mechanistic probe report to {out_p}")


if __name__ == "__main__":
    main()
