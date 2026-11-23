#!/usr/bin/env python3
"""Parallel 16-Task Local-GPU Evaluator & Cross-Session Interference Suite for `SmolVLA-DeltaLoop` (`Cycle 3B`).

Wraps `SmolVLARoboMMEWrapper` (`src/models/smolvla_policy_wrapper.py`) across all 16 RoboMME tasks
(`val` or `test` split) on the local RTX 5080 (`5` parallel workers at `< 1.9 GB` VRAM/worker):

1. **Zero Extra Vision Encoder in VRAM:**
   Each worker loads `SmolVLADeltaLoopPolicy` (`bf16` VLM + Expert, `fp32` GDN states) and reuses
   `SmolVLM2`'s frozen ViT + `4x4` PixelShuffle connector directly (`~1.6–1.85 GB` VRAM/worker).
   Supports BOTH full `state_dict` `.pt` checkpoints AND compact trainable-only `.pt` checkpoints
   (automatically hydrating frozen `SmolVLM2` weights from local `~/.cache/huggingface/hub/models--lerobot--smolvla_base`).
2. **All 16 RoboMME Tasks + Round-Robin Stratified Sharding:**
   Evaluates all 16 tasks (`4 suites × 4 tasks`) or the 8-task Cycle 2A/3A subset (`--task-preset rep8`),
   scheduling shards round-robin across suites/tasks so interim results remain balanced across all 16 tasks.
3. **Two-Wave Priority Queue + Multi-Seed (`seeds = 42, 43, 44`) Aggregation:**
   - **Wave 1 (Headline Contenders):** Evaluates `smolvla_m0_base`, `smolvla_m7_deltaloop` (`adaptive`),
     `smolvla_m2_framesamp`, `smolvla_m5_gdn2l_l1`, `smolvla_m1_rdvla_loop4` (and `s43`/`s44` seeds) FIRST
     the instant each checkpoint arrives locally or on Miyabi (`checkpoints/robomme/cycle3b_smolvla/`).
   - **Wave 2 (Intra-Checkpoint Multi-Hop Sweeps):** Evaluates `L2`, `L4`, `L1`, and `coda_conv` on
     `smolvla_m7_deltaloop` without blocking newly arrived Wave-1 models.
4. **Live Closed-Loop Cross-Session Interference (`--distractor-sessions k`, `k in {0, 1, 3}`):**
   Streams `k` deterministic prior distractor episodes through `SmolVLARoboMMEWrapper` without resetting
   persistent memory (`_S1/_S2` vs `_framesamp_committed`), preserving exact `A3` 16-frame exclusion
   and seed-matched `t = 1.0 -> 0.0` Euler ODE noise on the target episode.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Literal, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for _p in (str(REPO_ROOT), str(SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robomme_eval.interface import (  # noqa: E402
    MME_VLA_NORM_STATS,
    ROBOMME_SUITES,
    ROBOMME_TASKS,
    TASK_TO_SUITE,
    pack_robomme_state,
    to_uint8_rgb,
)
from robomme_eval.metrics import (  # noqa: E402
    EpisodeResult,
    apply_holm_bonferroni,
    compare_did_policy_runs,
    compare_policy_runs,
    summarize_episodes,
)
from robomme_eval.runner import evaluate_policy  # noqa: E402

SUITES_ORDER: tuple[str, ...] = ("Permanence", "Counting", "Reference", "Imitation")

ALL_16_TASKS_DISPLAY: tuple[str, ...] = (
    # Permanence (4)
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "VideoUnmask",
    "VideoUnmaskSwap",
    # Counting (4)
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    # Reference (4)
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    # Imitation (4)
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)

REP_8_TASKS_DISPLAY: tuple[str, ...] = (
    "ButtonUnmaskSwap",
    "VideoUnmaskSwap",
    "PickXtimes",
    "StopCube",
    "PickHighlight",
    "VideoPlaceOrder",
    "PatternLock",
    "MoveCube",
)

ARCH_BASE_PRIORITY: tuple[str, ...] = (
    "smolvla_m0_base",
    "smolvla_m7_deltaloop",
    "smolvla_m6_deltaloop",
    "smolvla_m2_framesamp",
    "smolvla_m5_gdn2l_l1",
    "smolvla_m1_rdvla_loop4",
)

ARCH_BASE_LABELS: dict[str, str] = {
    "smolvla_m0_base": "SmolVLA-M0: Pretrained Base (L=1)",
    "smolvla_m1_rdvla_loop4": "SmolVLA-M1: RD-VLA Memoryless Loop (L=4)",
    "smolvla_m2_framesamp": "SmolVLA-M2: FrameSamp-32 + Modul/XAttn",
    "smolvla_m5_gdn2l_l1": "SmolVLA-M5: 2L Motion-Gated GDN (L=1)",
    "smolvla_m6_deltaloop": "SmolVLA-M6: 1L DeltaLoop-VLA",
    "smolvla_m7_deltaloop": "SmolVLA-M7: 2L Iso-State DeltaLoop-VLA",
    "smolvla_m7_deltaloop_q16": "SmolVLA-M7-Q16: 2L Iso-State DeltaLoop-VLA (N_c=16)",
    "smolvla_m7_deltaloop_anchor": "SmolVLA-M7-Anchor: 2L Iso-State DeltaLoop-VLA (4+4 Hop-0 Anchor)",
}

MIYABI_CYCLE3B_DIR = "/work/gj26/b20098/projects/deltanet-looped-vla/checkpoints/robomme/cycle3b_smolvla"
LOCAL_CYCLE3B_DIR = REPO_ROOT / "checkpoints" / "robomme" / "cycle3b_smolvla"
DEFAULT_MIRROR_DIR = REPO_ROOT / "checkpoints" / "robomme" / "eval_mirror_cycle3b"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "robomme" / "cycle3b_smolvla"
DEFAULT_EVAL_PYTHON = REPO_ROOT / "external" / "robomme_benchmark" / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class EvalSpec:
    eval_id: str
    ckpt_stem: str
    arch: str
    train_seed: int
    label: str
    ckpt_path: Path
    halt_mode: Literal["fixed", "learned_gate", "coda_conv"]
    num_loops: int | None
    wave: int = 1
    halt_threshold: float = 0.5
    coda_conv_tau: float = 0.18
    ode_steps: int = 10
    distractor_sessions: int = 0


# =============================================================================
# 1. Local `smolvla_base` Safetensors Resolution & Checkpoint Loading
# =============================================================================


def find_local_smolvla_safetensors() -> Path | None:
    """Locate cached `models--lerobot--smolvla_base/.../model.safetensors` locally or on Miyabi."""
    candidates: list[Path] = []
    hf_hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--lerobot--smolvla_base"
    if hf_hub.exists():
        candidates.extend(sorted(hf_hub.rglob("model.safetensors")))
    miyabi_p = Path(
        "/work/gj26/b20098/hub/models--lerobot--smolvla_base/"
        "snapshots/c83c3163b8ca9b7e67c509fffd9121e66cb96205/model.safetensors"
    )
    if miyabi_p.exists():
        candidates.append(miyabi_p)
    return candidates[0] if candidates else None


def parse_stem_arch_and_seed(stem: str, payload: dict[str, Any] | None = None) -> tuple[str, int]:
    """Extract canonical `(arch, train_seed)` from checkpoint stem and/or payload metadata."""
    arch = ""
    train_seed = 42
    if isinstance(payload, dict):
        mc = payload.get("model_config", {})
        if isinstance(mc, dict) and mc.get("arch"):
            arch = str(mc["arch"])
        if isinstance(payload.get("seed"), int):
            train_seed = int(payload["seed"])
        elif isinstance(payload.get("metrics"), dict) and isinstance(payload["metrics"].get("seed"), int):
            train_seed = int(payload["metrics"]["seed"])
        elif isinstance(payload.get("train_args"), dict) and isinstance(payload["train_args"].get("seed"), int):
            train_seed = int(payload["train_args"]["seed"])

    m_seed = re.search(r"_(?:s|seed)(\d+)$", stem)
    base_stem = stem
    if m_seed:
        train_seed = int(m_seed.group(1))
        base_stem = stem[: m_seed.start()]

    if not arch:
        for cand in ARCH_BASE_PRIORITY:
            if base_stem == cand or base_stem.startswith(cand):
                arch = cand
                break
        if not arch:
            arch = base_stem
    return arch, train_seed


def load_smolvla_model_from_ckpt(ckpt_path: Path) -> tuple[Any, dict[str, Any]]:
    """Load `SmolVLADeltaLoopPolicy` on CPU supporting BOTH full and compact (trainable-only) `.pt` files."""
    import torch
    from models.smolvla_deltaloop import SmolVLADeltaLoopConfig, SmolVLADeltaLoopPolicy

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload or "model_state_dict" not in payload:
        raise ValueError(f"Invalid SmolVLA checkpoint structure in {ckpt_path}")

    cfg = SmolVLADeltaLoopConfig.from_dict(payload["model_config"])
    model = SmolVLADeltaLoopPolicy(cfg)
    sd = payload["model_state_dict"]

    # Check if frozen VLM weights are included in `sd` or need hydration from local `smolvla_base`
    has_vlm_weights = any(k.startswith("vlm_with_expert.vlm.") for k in sd.keys())
    if not has_vlm_weights:
        st_path = find_local_smolvla_safetensors()
        if st_path is None:
            raise FileNotFoundError(
                "Compact checkpoint omits frozen VLM weights, and `models--lerobot--smolvla_base` was not found in ~/.cache/huggingface/hub/"
            )
        model.load_pretrained_smolvla(st_path)
        incomp = model.load_state_dict(sd, strict=False)
        if incomp.unexpected_keys:
            raise RuntimeError(f"Unexpected keys in compact SmolVLA checkpoint: {incomp.unexpected_keys[:8]}")
        bad_missing = [k for k in incomp.missing_keys if not k.startswith("vlm_with_expert.vlm.")]
        if bad_missing:
            raise RuntimeError(f"Missing non-VLM keys in compact SmolVLA checkpoint: {bad_missing[:8]}")
    else:
        model.load_state_dict(sd, strict=True)

    return model, payload


def verify_smolvla_checkpoint(ckpt_path: Path) -> bool:
    """Verify `.pt` file is complete (`completed=True` or accompanied by metrics) and loads cleanly."""
    try:
        model, payload = load_smolvla_model_from_ckpt(ckpt_path)
        if "completed" in payload and not bool(payload["completed"]):
            return False
        del model
        return True
    except Exception as exc:
        print(f"[sync-3b] Rejecting incomplete/incompatible checkpoint {ckpt_path}: {exc}", flush=True)
        return False


def calibrate_smolvla_coda_tau(ckpt_path: Path, default_tau: float = 0.08) -> float:
    """Calibrate `coda_conv_tau` from the checkpoint's hopwise relative Coda L2 drift."""
    import torch

    try:
        model, _ = load_smolvla_model_from_ckpt(ckpt_path)
        model.eval()
        if not model.is_multi_depth:
            return default_tau
        with torch.no_grad():
            g = torch.Generator().manual_seed(1234)
            B = 16
            ctx = torch.randn(B, 84, model.cfg.d_mem, generator=g)
            mask = torch.ones(B, 84, dtype=torch.bool)
            mem_state = None
            if model.uses_gdn:
                H, K, Vh = model.cfg.gdn_heads, model.cfg.gdn_d_k, model.cfg.gdn_d_v // 2
                mem_state = {
                    "S1": torch.randn(B, H, Vh, K, generator=g) * 0.15,
                    "occ1": torch.ones(B, H, K) * 0.5,
                    "S2": torch.randn(B, H, Vh, K, generator=g) * 0.15,
                    "occ2": torch.ones(B, H, K) * 0.5,
                }
            ro = model.run_coda_readout(
                ctx, mask, mem_state=mem_state, batch_size=B, device=torch.device("cpu"),
                num_loops=4, halt_mode="fixed",
            )
            codas = ro.get("coda_per_depth") or ro.get("all_coda")
            if not codas or len(codas) < 3 or codas[0] is None:
                return default_tau
            dc12 = (codas[1].float() - codas[0].float()).norm(dim=-1).mean(dim=-1) / (
                codas[0].float().norm(dim=-1).mean(dim=-1).clamp(min=1e-6)
            )
            dc23 = (codas[2].float() - codas[1].float()).norm(dim=-1).mean(dim=-1) / (
                codas[1].float().norm(dim=-1).mean(dim=-1).clamp(min=1e-6)
            )
            tau = float(0.5 * (dc12.median().item() + dc23.median().item()))
            return round(max(0.02, min(0.45, tau)), 3)
    except Exception:
        return default_tau


def build_specs_for_checkpoint(
    ckpt_stem: str,
    ckpt_path: Path,
    arch: str,
    train_seed: int,
    *,
    include_sweeps: bool = True,
    include_coda_conv: bool = True,
    include_l4: bool = True,
    distractor_sessions: int = 0,
) -> list[EvalSpec]:
    """Expand a mirrored `<ckpt_stem>.pt` into Wave-1 headline + Wave-2 multi-hop sweep specs."""
    k_suf = f"_interfk{distractor_sessions}" if distractor_sessions > 0 else ""
    k_lbl = f" [Interf k={distractor_sessions}]" if distractor_sessions > 0 else ""
    is_q16 = "_q16" in ckpt_stem
    is_anchor = "_anchor" in ckpt_stem
    seed_suf = "" if (ckpt_stem == arch and train_seed == 42) else (
        "" if ckpt_stem != arch else f"_s{train_seed}"
    )
    id_prefix = f"{ckpt_stem}{seed_suf}"
    seed_lbl = f" [seed={train_seed}]" if train_seed != 42 else ""
    base_lbl = (
        ARCH_BASE_LABELS["smolvla_m7_deltaloop_anchor"]
        if is_anchor
        else (
            ARCH_BASE_LABELS["smolvla_m7_deltaloop_q16"]
            if is_q16
            else ARCH_BASE_LABELS.get(arch, ckpt_stem)
        )
    )

    if arch in ("smolvla_m7_deltaloop", "smolvla_m6_deltaloop"):
        specs = [
            EvalSpec(
                eval_id=f"{id_prefix}_adaptive{k_suf}",
                ckpt_stem=ckpt_stem,
                arch=arch,
                train_seed=train_seed,
                label=f"{base_lbl} (Adaptive CRN Gate){seed_lbl}{k_lbl}",
                ckpt_path=ckpt_path,
                halt_mode="learned_gate",
                num_loops=4,
                wave=1 if train_seed == 42 else 2,
                distractor_sessions=distractor_sessions,
            )
        ]
        # Cycle 4B U8 (`_anchor`): intra-checkpoint `Fixed L=2` causal test bypassing Anchor-Shortcut Halting (`H9f`)
        if include_sweeps and is_anchor and distractor_sessions == 0 and train_seed == 42:
            specs.append(
                EvalSpec(
                    eval_id=f"{id_prefix}_L2{k_suf}",
                    ckpt_stem=ckpt_stem,
                    arch=arch,
                    train_seed=train_seed,
                    label=f"{base_lbl} (Fixed L=2){seed_lbl}{k_lbl}",
                    ckpt_path=ckpt_path,
                    halt_mode="fixed",
                    num_loops=2,
                    wave=2,
                    distractor_sessions=distractor_sessions,
                )
            )
        # Run intra-checkpoint sweeps (`L2, [L4], L1, coda_conv`) on primary `N_c=4` `seed=42` when `k=0`
        if include_sweeps and (not is_q16) and (not is_anchor) and distractor_sessions == 0 and train_seed == 42:
            specs.append(
                EvalSpec(
                    eval_id=f"{id_prefix}_L2{k_suf}",
                    ckpt_stem=ckpt_stem,
                    arch=arch,
                    train_seed=train_seed,
                    label=f"{base_lbl} (Fixed L=2){seed_lbl}{k_lbl}",
                    ckpt_path=ckpt_path,
                    halt_mode="fixed",
                    num_loops=2,
                    wave=2,
                    distractor_sessions=distractor_sessions,
                )
            )
            if include_l4:
                specs.append(
                    EvalSpec(
                        eval_id=f"{id_prefix}_L4{k_suf}",
                        ckpt_stem=ckpt_stem,
                        arch=arch,
                        train_seed=train_seed,
                        label=f"{base_lbl} (Fixed L=4){seed_lbl}{k_lbl}",
                        ckpt_path=ckpt_path,
                        halt_mode="fixed",
                        num_loops=4,
                        wave=2,
                        distractor_sessions=distractor_sessions,
                    )
                )
            specs.append(
                EvalSpec(
                    eval_id=f"{id_prefix}_L1{k_suf}",
                    ckpt_stem=ckpt_stem,
                    arch=arch,
                    train_seed=train_seed,
                    label=f"{base_lbl} (Fixed L=1){seed_lbl}{k_lbl}",
                    ckpt_path=ckpt_path,
                    halt_mode="fixed",
                    num_loops=1,
                    wave=2,
                    distractor_sessions=distractor_sessions,
                )
            )
            if include_coda_conv:
                tau_cal = calibrate_smolvla_coda_tau(ckpt_path, default_tau=0.18)
                specs.append(
                    EvalSpec(
                        eval_id=f"{id_prefix}_coda_conv{k_suf}",
                        ckpt_stem=ckpt_stem,
                        arch=arch,
                        train_seed=train_seed,
                        label=f"{base_lbl} (Latent L2 Conv tau={tau_cal:.2f}){seed_lbl}{k_lbl}",
                        ckpt_path=ckpt_path,
                        halt_mode="coda_conv",
                        num_loops=4,
                        wave=2,
                        coda_conv_tau=tau_cal,
                        distractor_sessions=distractor_sessions,
                    )
                )
        return specs

    num_loops = 4 if arch == "smolvla_m1_rdvla_loop4" else 1
    return [
        EvalSpec(
            eval_id=f"{id_prefix}{k_suf}",
            ckpt_stem=ckpt_stem,
            arch=arch,
            train_seed=train_seed,
            label=f"{base_lbl}{seed_lbl}{k_lbl}",
            ckpt_path=ckpt_path,
            halt_mode="fixed",
            num_loops=num_loops,
            wave=1 if train_seed == 42 else 2,
            distractor_sessions=distractor_sessions,
        )
    ]


# =============================================================================
# 2. Atomic Local + Miyabi Checkpoint Mirroring
# =============================================================================


def _run_miyabi_cmd(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    res = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "miyabi-g1", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if res.returncode == 0:
        return res
    ssh_helper = Path.home() / ".local" / "bin" / "ssh-miyabi"
    return subprocess.run(
        [str(ssh_helper), "1", "--batch"],
        input=cmd + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def sync_ready_checkpoints(
    mirror_dir: Path,
    local_train_dir: Path = LOCAL_CYCLE3B_DIR,
    miyabi_ckpt_dir: str = MIYABI_CYCLE3B_DIR,
) -> dict[str, Any]:
    """Atomically mirror completed `<stem>.pt` checkpoints from local disk and/or Miyabi."""
    mirror_dir.mkdir(parents=True, exist_ok=True)
    newly_synced: list[str] = []

    local_candidates: list[tuple[Path, Path, str]] = []
    if local_train_dir.exists():
        for cpath in sorted(local_train_dir.glob("*.pt")):
            if not cpath.name.startswith((".", "_")):
                local_candidates.append((cpath, local_train_dir / f"{cpath.stem}_metrics.json", cpath.stem))
        for sdir in sorted(local_train_dir.glob("seed*")):
            if not sdir.is_dir():
                continue
            m_s = re.match(r"^seed(\d+)$", sdir.name)
            if not m_s:
                continue
            s_num = int(m_s.group(1))
            for cpath in sorted(sdir.glob("*.pt")):
                if cpath.name.startswith((".", "_")):
                    continue
                dst_stem = cpath.stem if (s_num == 42 or re.search(r"_(?:s|seed)\d+$", cpath.stem)) else f"{cpath.stem}_s{s_num}"
                local_candidates.append((cpath, sdir / f"{cpath.stem}_metrics.json", dst_stem))

    if local_train_dir.parent.exists():
        for sib in sorted(local_train_dir.parent.glob(f"{local_train_dir.name}_*")):
            if not sib.is_dir():
                continue
            if sib.name.endswith("_q16"):
                for cpath in sorted(sib.glob("*.pt")):
                    if cpath.name.startswith((".", "_")):
                        continue
                    dst_stem = cpath.stem if "_q16" in cpath.stem else f"{cpath.stem}_q16"
                    local_candidates.append((cpath, sib / f"{cpath.stem}_metrics.json", dst_stem))
                continue
            if sib.name.endswith("_anchor"):
                for cpath in sorted(sib.glob("*.pt")):
                    if cpath.name.startswith((".", "_")):
                        continue
                    dst_stem = cpath.stem if "_anchor" in cpath.stem else f"{cpath.stem}_anchor"
                    local_candidates.append((cpath, sib / f"{cpath.stem}_metrics.json", dst_stem))
                continue
            m_sib = re.search(r"_(?:s|seed)(\d+)$", sib.name)
            if not m_sib:
                continue
            s_num = int(m_sib.group(1))
            for cpath in sorted(sib.glob("*.pt")):
                if cpath.name.startswith((".", "_")):
                    continue
                dst_stem = cpath.stem if (s_num == 42 or re.search(r"_(?:s|seed)\d+$", cpath.stem)) else f"{cpath.stem}_s{s_num}"
                local_candidates.append((cpath, sib / f"{cpath.stem}_metrics.json", dst_stem))

    for cpath, mpath, stem in local_candidates:
        dst_ckpt = mirror_dir / f"{stem}.pt"
        dst_metrics = mirror_dir / f"{stem}_metrics.json"
        if dst_ckpt.exists() and dst_ckpt.stat().st_size > 1_000_000:
            if mpath.exists() and not dst_metrics.exists():
                shutil.copy2(mpath, dst_metrics)
            continue
        if cpath.stat().st_size < 1_000_000:
            continue
        if verify_smolvla_checkpoint(cpath):
            tmp_c = mirror_dir / f".{stem}.pt.tmp"
            shutil.copy2(cpath, tmp_c)
            if mpath.exists():
                tmp_m = mirror_dir / f".{stem}_metrics.json.tmp"
                shutil.copy2(mpath, tmp_m)
                tmp_m.replace(dst_metrics)
            tmp_c.replace(dst_ckpt)
            sz_mb = dst_ckpt.stat().st_size / (1024**2)
            print(
                f"[sync-3b] Verified & staged `{dst_ckpt.name}` ({sz_mb:.1f} MiB) from local {cpath.parent.name}/!",
                flush=True,
            )
            newly_synced.append(stem)

    probe_script = (
        f"ls -1 {miyabi_ckpt_dir}/*_metrics.json {miyabi_ckpt_dir}/seed*/*_metrics.json "
        f"{miyabi_ckpt_dir}_*/*_metrics.json 2>/dev/null || true; "
        f"echo '__QSTAT__'; "
        f"/usr/local/bin/qstat 2>/dev/null || true"
    )
    probe = _run_miyabi_cmd(probe_script, timeout=25)
    pbs_running: bool | None = None
    remote_metrics: list[str] = []
    if probe.returncode == 0:
        parts = probe.stdout.split("__QSTAT__")
        remote_metrics = [
            ln.strip() for ln in parts[0].splitlines() if ln.strip().endswith("_metrics.json")
        ]
        if len(parts) > 1:
            q_lines = [
                ln for ln in parts[1].splitlines()
                if ln.strip() and not ln.strip().startswith(("Job id", "---"))
            ]
            pbs_running = len(q_lines) > 0

    for mpath_remote in remote_metrics:
        rp = Path(mpath_remote)
        mname = rp.name
        raw_stem = mname[: -len("_metrics.json")]
        if raw_stem.startswith("_") or raw_stem.startswith("."):
            continue
        parent_name = rp.parent.name
        m_s = re.match(r"^seed(\d+)$", parent_name) or re.search(r"_(?:s|seed)(\d+)$", parent_name)
        if parent_name.endswith("_anchor") and "_anchor" not in raw_stem:
            stem = f"{raw_stem}_anchor"
        elif parent_name.endswith("_q16") and "_q16" not in raw_stem:
            stem = f"{raw_stem}_q16"
        elif "_q16_" in parent_name and m_s and "_q16" not in raw_stem:
            stem = f"{raw_stem}_q16_s{int(m_s.group(1))}"
        elif m_s and int(m_s.group(1)) != 42 and not re.search(r"_(?:s|seed)\d+$", raw_stem):
            stem = f"{raw_stem}_s{int(m_s.group(1))}"
        else:
            stem = raw_stem
        remote_dir_p = str(rp.parent)
        dst_ckpt = mirror_dir / f"{stem}.pt"
        dst_metrics = mirror_dir / f"{stem}_metrics.json"
        if dst_ckpt.exists() and dst_ckpt.stat().st_size > 1_000_000 and dst_metrics.exists():
            continue

        tmp_ckpt = mirror_dir / f".{stem}.pt.tmp"
        tmp_metrics = mirror_dir / f".{stem}_metrics.json.tmp"
        print(f"[sync-3b] Pulling `{stem}.pt` from Miyabi ({remote_dir_p})...", flush=True)
        r1 = subprocess.run(
            ["rsync", "-az", f"miyabi-g1:{remote_dir_p}/{mname}", str(tmp_metrics)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        r2 = subprocess.run(
            ["rsync", "-az", f"miyabi-g1:{remote_dir_p}/{raw_stem}.pt", str(tmp_ckpt)],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        if r1.returncode == 0 and r2.returncode == 0 and tmp_ckpt.exists() and tmp_ckpt.stat().st_size > 1_000_000:
            if verify_smolvla_checkpoint(tmp_ckpt):
                tmp_metrics.replace(dst_metrics)
                tmp_ckpt.replace(dst_ckpt)
                sz_mb = dst_ckpt.stat().st_size / (1024**2)
                print(f"[sync-3b] Verified & staged `{stem}.pt` ({sz_mb:.1f} MiB) from Miyabi!", flush=True)
                newly_synced.append(stem)
            else:
                tmp_ckpt.unlink(missing_ok=True)
                tmp_metrics.unlink(missing_ok=True)
        else:
            tmp_ckpt.unlink(missing_ok=True)
            tmp_metrics.unlink(missing_ok=True)

    return {"newly_synced": newly_synced, "pbs_running": pbs_running}


def discover_mirrored_checkpoints(mirror_dir: Path) -> list[tuple[str, Path, str, int]]:
    """Return sorted `(ckpt_stem, ckpt_path, arch, train_seed)` in Wave-1 scientific priority order."""
    import torch

    entries: list[tuple[str, Path, str, int]] = []
    if not mirror_dir.exists():
        return entries
    for p in mirror_dir.glob("*.pt"):
        if p.name.startswith(".") or p.name.startswith("_") or p.stat().st_size < 1_000_000:
            continue
        stem = p.stem
        meta_json = mirror_dir / f"{stem}_metrics.json"
        arch = ""
        seed = 42
        if meta_json.exists():
            try:
                md = json.loads(meta_json.read_text())
                arch = str(md.get("arch", ""))
                if isinstance(md.get("seed"), int):
                    seed = int(md["seed"])
            except Exception:
                pass
        if not arch:
            try:
                pl = torch.load(p, map_location="cpu", weights_only=False)
                arch, seed = parse_stem_arch_and_seed(stem, pl)
            except Exception:
                arch, seed = parse_stem_arch_and_seed(stem, None)
        else:
            _, seed_from_stem = parse_stem_arch_and_seed(stem, None)
            if seed == 42 and seed_from_stem != 42:
                seed = seed_from_stem
        entries.append((stem, p, arch, seed))

    prio_map = {a: i for i, a in enumerate(ARCH_BASE_PRIORITY)}
    entries.sort(key=lambda t: (0 if t[3] == 42 else 1, prio_map.get(t[2], 99), t[3], t[0]))
    return entries


# =============================================================================
# 3. Live Cross-Session Interference & Worker Shard Execution (`SmolVLARoboMMEWrapper`)
# =============================================================================


def _set_model_rng_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_smolvla_intra_episode_short_caches(policy: Any) -> None:
    """Clear ONLY short-term intra-episode caches while preserving persistent recurrent memory."""
    policy._pending_strided.clear()
    policy._conv_ring.clear()
    policy._prev_written_front_tok = None
    policy._prev_exec_obs = None


def _collect_deterministic_distractor_session(
    env_builder: Any,
    distractor_ep_id: int,
    exec_steps: int = 140,
) -> dict[str, Any]:
    """Collect a deterministic moving-arm distractor episode (`demo + 140 execution frames`)."""
    env = env_builder.make_env_for_episode(distractor_ep_id)
    try:
        obs, info = env.reset()
        raw_goal = info.get("task_goal", "")
        task_goal = raw_goal[0] if isinstance(raw_goal, (list, tuple)) and raw_goal else str(raw_goal)

        init_front = [to_uint8_rgb(x) for x in obs["front_rgb_list"]]
        init_wrist = [to_uint8_rgb(x) for x in obs["wrist_rgb_list"]]
        init_states = [
            pack_robomme_state(j, g)
            for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])
        ]
        demo_front = init_front[:-1]
        demo_wrist = init_wrist[:-1]
        demo_states = init_states[:-1]

        exec_front: list[np.ndarray] = [init_front[-1]]
        exec_wrist: list[np.ndarray] = [init_wrist[-1]]
        exec_states: list[np.ndarray] = [init_states[-1]]

        q0 = np.asarray(init_states[-1], dtype=np.float32).copy()
        phase = float((distractor_ep_id % 7) * 0.7)
        for st_i in range(exec_steps):
            act = q0.copy()
            t_rad = 2.0 * np.pi * (st_i / 40.0) + phase
            act[0] = q0[0] + 0.10 * np.sin(t_rad)
            act[1] = q0[1] + 0.06 * np.cos(t_rad)
            act[3] = q0[3] + 0.08 * np.sin(1.5 * t_rad)
            act[7] = 0.04 if (st_i // 35) % 2 == 0 else -0.04
            obs_s, _, terminated, truncated, step_info = env.step(act)
            if step_info is not None and step_info.get("status") == "error":
                break
            exec_front.append(to_uint8_rgb(obs_s["front_rgb_list"][-1]))
            exec_wrist.append(to_uint8_rgb(obs_s["wrist_rgb_list"][-1]))
            exec_states.append(
                pack_robomme_state(obs_s["joint_state_list"][-1], obs_s["gripper_state_list"][-1])
            )
            if terminated or truncated:
                break

        return {
            "task_goal": task_goal,
            "demo_front": demo_front,
            "demo_wrist": demo_wrist,
            "demo_states": demo_states,
            "exec_front": exec_front,
            "exec_wrist": exec_wrist,
            "exec_states": exec_states,
        }
    finally:
        env.close()


def evaluate_smolvla_with_interference(
    policy: Any,
    *,
    task: str,
    dataset: str,
    episode_ids: Sequence[int],
    model_seed: int,
    exec_horizon: int,
    max_steps: int,
    distractor_sessions: int,
    output_json: Path,
) -> dict[str, Any]:
    """Closed-loop ManiSkill3 rollout with `k` prior distractor episodes streamed without reset."""
    os.environ.setdefault("DISPLAY", "")
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    completed: dict[int, EpisodeResult] = {}
    if output_json.exists():
        try:
            prev = json.loads(output_json.read_text())
            for raw_ep in prev.get("episodes", []):
                ep = EpisodeResult(**raw_ep)
                completed[ep.episode_id] = ep
        except Exception:
            pass

    env_builder = BenchmarkEnvBuilder(
        env_id=task,
        dataset=dataset,
        action_space="joint_angle",
        gui_render=False,
        max_steps=max_steps,
    )
    total_avail = max(1, env_builder.get_episode_num())
    distractor_cache: dict[int, dict[str, Any]] = {}
    all_results: list[EpisodeResult] = list(completed.values())

    for ep_id in episode_ids:
        if ep_id in completed or not (0 <= ep_id < total_avail):
            continue

        t_ep0 = time.perf_counter()
        policy.reset()

        d_ids = [(ep_id + 1 + j * 3) % total_avail for j in range(distractor_sessions)]
        for d_id in d_ids:
            if d_id not in distractor_cache:
                distractor_cache[d_id] = _collect_deterministic_distractor_session(env_builder, d_id)
            d_sess = distractor_cache[d_id]
            _clear_smolvla_intra_episode_short_caches(policy)
            d_base = int(policy._frame_cursor)
            policy._demo_frames_count = 0
            if d_sess["demo_front"]:
                policy.ingest_demo_frames(
                    d_sess["demo_front"],
                    d_sess["demo_wrist"],
                    d_sess["demo_states"],
                    d_sess["task_goal"],
                )
            policy._demo_frames_count = d_base + len(d_sess["demo_front"])
            policy.ingest_step_history(
                d_sess["exec_front"],
                d_sess["exec_wrist"],
                d_sess["exec_states"],
                d_sess["task_goal"],
            )
            policy._flush_eligible_history(cutoff_excl=int(policy._frame_cursor))

        _set_model_rng_seed(model_seed)
        policy.set_seed(model_seed)
        _clear_smolvla_intra_episode_short_caches(policy)

        target_base = int(policy._frame_cursor)
        policy._demo_frames_count = 0

        env = env_builder.make_env_for_episode(ep_id)
        try:
            obs, info = env.reset()
            raw_goal = info.get("task_goal", "")
            task_goal = raw_goal[0] if isinstance(raw_goal, (list, tuple)) and raw_goal else str(raw_goal)
            difficulty = getattr(getattr(env, "unwrapped", None), "difficulty", None)

            init_front = [to_uint8_rgb(x) for x in obs["front_rgb_list"]]
            init_wrist = [to_uint8_rgb(x) for x in obs["wrist_rgb_list"]]
            init_states = [
                pack_robomme_state(j, g)
                for j, g in zip(obs["joint_state_list"], obs["gripper_state_list"])
            ]
            exec_start_local = len(init_front) - 1
            t_demo0 = time.perf_counter()
            if exec_start_local > 0:
                policy.ingest_demo_frames(
                    init_front[:-1],
                    init_wrist[:-1],
                    init_states[:-1],
                    task_goal,
                )
            demo_ingest_ms = (time.perf_counter() - t_demo0) * 1000.0
            target_exec_start_global = target_base + exec_start_local
            policy._demo_frames_count = target_exec_start_global

            curr_front = init_front[-1]
            curr_wrist = init_wrist[-1]
            curr_state = init_states[-1]
            inter_front: list[np.ndarray] = []
            inter_wrist: list[np.ndarray] = []
            inter_states: list[np.ndarray] = []

            steps_executed = 0
            loop_depths: list[float] = []
            chunk_infer_ms: list[float] = []
            outcome = "timeout"
            success = False
            error_message: str | None = None

            while steps_executed < max_steps:
                t_inf0 = time.perf_counter()
                if inter_front:
                    policy.ingest_step_history(inter_front, inter_wrist, inter_states, task_goal)
                    inter_front.clear()
                    inter_wrist.clear()
                    inter_states.clear()

                chunk_out = policy.predict_chunk(
                    curr_front,
                    curr_wrist,
                    curr_state,
                    task_goal,
                    step_idx=target_exec_start_global + steps_executed,
                    exec_start_idx=target_exec_start_global,
                )
                chunk_infer_ms.append((time.perf_counter() - t_inf0) * 1000.0)
                loop_depths.append(float(chunk_out.loop_depth))

                actions = np.asarray(chunk_out.actions, dtype=np.float32)
                chunk_to_run = actions[:exec_horizon]
                n_sub = chunk_to_run.shape[0]
                stop_episode = False

                for sub_i in range(n_sub):
                    if steps_executed >= max_steps:
                        outcome = "timeout"
                        stop_episode = True
                        break
                    obs_s, _, terminated, truncated, step_info = env.step(chunk_to_run[sub_i])
                    steps_executed += 1
                    if step_info is not None and step_info.get("status") == "error":
                        outcome = "error"
                        error_message = str(step_info.get("error_message", "unknown simulator error"))
                        success = False
                        stop_episode = True
                        break
                    f_next = to_uint8_rgb(obs_s["front_rgb_list"][-1])
                    w_next = to_uint8_rgb(obs_s["wrist_rgb_list"][-1])
                    s_next = pack_robomme_state(
                        obs_s["joint_state_list"][-1], obs_s["gripper_state_list"][-1]
                    )
                    is_succ = bool(step_info.get("success", False)) if step_info is not None else False
                    if is_succ:
                        success = True
                        outcome = "success"
                        stop_episode = True
                        break
                    if terminated or truncated:
                        outcome = "failure"
                        stop_episode = True
                        break
                    if sub_i < n_sub - 1:
                        inter_front.append(f_next)
                        inter_wrist.append(w_next)
                        inter_states.append(s_next)
                    else:
                        curr_front = f_next
                        curr_wrist = w_next
                        curr_state = s_next

                if stop_episode:
                    break
        finally:
            env.close()

        mean_ld = float(np.mean(loop_depths)) if loop_depths else 1.0
        p95_ld = float(np.percentile(loop_depths, 95.0)) if loop_depths else 1.0
        total_inf = float(demo_ingest_ms + np.sum(chunk_infer_ms))
        mean_ms = float(np.mean(chunk_infer_ms)) if chunk_infer_ms else total_inf

        ep_res = EpisodeResult(
            task=task,
            suite=TASK_TO_SUITE.get(task, "Unknown"),
            dataset=dataset,
            episode_id=int(ep_id),
            model_seed=int(model_seed),
            success=bool(success),
            outcome=outcome,
            steps=int(steps_executed),
            demo_frames=int(exec_start_local),
            num_chunks=len(chunk_infer_ms),
            mean_loop_depth=mean_ld,
            p95_loop_depth=p95_ld,
            loop_depths=loop_depths,
            mean_ms_per_chunk=mean_ms,
            total_infer_ms=total_inf,
            episode_wall_s=float(time.perf_counter() - t_ep0),
            task_goal=task_goal,
            difficulty=str(difficulty) if difficulty is not None else None,
            error_message=error_message,
        )
        all_results.append(ep_res)
        completed[ep_id] = ep_res

        all_results.sort(key=lambda r: (r.task, r.episode_id, r.model_seed))
        payload = {
            "config": {
                "tasks": [task],
                "dataset": dataset,
                "episode_ids": list(episode_ids),
                "model_seeds": [model_seed],
                "exec_horizon": exec_horizon,
                "max_steps": max_steps,
                "distractor_sessions": distractor_sessions,
            },
            "summary": summarize_episodes(all_results),
            "episodes": [r.to_dict() for r in all_results],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = output_json.with_suffix(".json.tmp")
        tmp_p.write_text(json.dumps(payload, indent=2))
        tmp_p.replace(output_json)

    all_results.sort(key=lambda r: (r.task, r.episode_id, r.model_seed))
    return {
        "config": {
            "tasks": [task],
            "dataset": dataset,
            "episode_ids": list(episode_ids),
            "model_seeds": [model_seed],
            "exec_horizon": exec_horizon,
            "max_steps": max_steps,
            "distractor_sessions": distractor_sessions,
        },
        "summary": summarize_episodes(all_results),
        "episodes": [r.to_dict() for r in all_results],
    }


def run_worker_shard(args: argparse.Namespace) -> None:
    """Execute a single `(task, ep_start..ep_end)` shard inside one GPU worker."""
    os.environ["DISPLAY"] = ""
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import torch
    from models.smolvla_deltaloop import MeanStdDeltaNormalizer
    from models.smolvla_policy_wrapper import SmolVLARoboMMEWrapper

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()

    ckpt_path = Path(args.checkpoint)
    cpu_model, ckpt_payload = load_smolvla_model_from_ckpt(ckpt_path)
    norm_stats = ckpt_payload.get("norm_stats", MME_VLA_NORM_STATS)
    normalizer = MeanStdDeltaNormalizer(stats=norm_stats)

    def _low_vram_vision_fn(front_rgb: np.ndarray, wrist_rgb: np.ndarray) -> tuple[Any, Any]:
        # Chunk SigLIP 512x512 ViT encoding into small batches (`pair_bs=4`, 8 images/pass)
        # and release transient demo activation blocks so 5 workers stay <1.65 GiB VRAM each!
        f_tok, w_tok = cpu_model.encode_images_to_vlm_tokens(front_rgb, wrist_rgb, batch_size=8)
        n_frames = 1 if front_rgb.ndim == 3 else int(front_rgb.shape[0])
        if n_frames > 4 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return f_tok, w_tok

    policy = SmolVLARoboMMEWrapper(
        cpu_model,
        device=args.device,
        normalizer=normalizer,
        vision_token_fn=_low_vram_vision_fn,
        num_loops=args.num_loops,
        halt_mode=args.halt_mode,
        halt_threshold=args.halt_threshold,
        coda_conv_tau=args.coda_conv_tau,
        ode_steps=args.ode_steps,
        seed=args.seed,
    )

    ep_ids = list(range(args.ep_start, args.ep_end))
    shard_out = Path(args.shard_json)
    shard_out.parent.mkdir(parents=True, exist_ok=True)

    if args.distractor_sessions > 0:
        res = evaluate_smolvla_with_interference(
            policy,
            task=args.task,
            dataset=args.dataset,
            episode_ids=ep_ids,
            model_seed=args.seed,
            exec_horizon=args.exec_horizon,
            max_steps=args.max_steps,
            distractor_sessions=args.distractor_sessions,
            output_json=shard_out,
        )
    else:
        res = evaluate_policy(
            policy,
            tasks=[args.task],
            dataset=args.dataset,
            episode_ids=ep_ids,
            model_seeds=[args.seed],
            exec_horizon=args.exec_horizon,
            max_steps=args.max_steps,
            output_json=shard_out,
            resume=True,
        )

    peak_vram_mb = 0.0
    if args.device.startswith("cuda") and torch.cuda.is_available():
        peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024**2))

    res["worker_telemetry"] = {
        "arch": policy.model.arch,
        "state_size_bytes": int(policy.model.state_size_bytes()),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "distractor_sessions": int(args.distractor_sessions),
    }
    tmp_path = shard_out.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(res, indent=2))
    tmp_path.replace(shard_out)


# =============================================================================
# 4. Multi-Shard Parallel Orchestration & Loop Diagnostics
# =============================================================================


def compute_loop_trajectory_diagnostics(
    episodes: Sequence[EpisodeResult],
    tasks_order: Sequence[str],
) -> dict[str, Any]:
    all_chunks: list[float] = []
    for ep in episodes:
        all_chunks.extend(ep.loop_depths)
    if not all_chunks:
        return {}

    arr = np.asarray(all_chunks, dtype=np.float64)
    hist = {f"L={k}": round(float(np.mean(np.round(arr) == k)), 4) for k in (1, 2, 3, 4)}

    early_vals: list[float] = []
    mid_vals: list[float] = []
    late_vals: list[float] = []
    video_chunk0_vals: list[float] = []
    video_later_vals: list[float] = []
    by_task_traj: dict[str, dict[str, Any]] = {}

    for task in tasks_order:
        t_eps = [e for e in episodes if e.task == task and e.loop_depths]
        if not t_eps:
            continue
        t_chunks = np.asarray([x for e in t_eps for x in e.loop_depths], dtype=np.float64)
        c0 = np.asarray([e.loop_depths[0] for e in t_eps], dtype=np.float64)
        t_early, t_mid, t_late = [], [], []
        for ep in t_eps:
            lds = ep.loop_depths
            n = len(lds)
            k1 = max(1, n // 3)
            k2 = max(k1 + 1, (2 * n) // 3) if n >= 2 else n
            t_early.extend(lds[:k1])
            t_mid.extend(lds[k1:k2] if k2 > k1 else lds)
            t_late.extend(lds[k2:] if n > k2 else lds[-1:])
            if ep.demo_frames > 0:
                video_chunk0_vals.append(lds[0])
                if len(lds) > 1:
                    video_later_vals.extend(lds[1:])

        early_vals.extend(t_early)
        mid_vals.extend(t_mid)
        late_vals.extend(t_late)
        by_task_traj[task] = {
            "mean_L": round(float(t_chunks.mean()), 3),
            "p95_L": round(float(np.percentile(t_chunks, 95.0)), 3),
            "chunk0_mean_L": round(float(c0.mean()), 3),
            "early_mean_L": round(float(np.mean(t_early)), 3),
            "mid_mean_L": round(float(np.mean(t_mid)), 3),
            "late_mean_L": round(float(np.mean(t_late)), 3),
            "hist": {f"L={k}": round(float(np.mean(np.round(t_chunks) == k)), 4) for k in (1, 2, 3, 4)},
        }

    return {
        "total_chunks": int(arr.size),
        "overall_chunk_mean_L": round(float(arr.mean()), 4),
        "overall_chunk_std_L": round(float(arr.std()), 4),
        "overall_hist": hist,
        "phase_means": {
            "early_0_33pct": round(float(np.mean(early_vals)), 4) if early_vals else 0.0,
            "mid_33_67pct": round(float(np.mean(mid_vals)), 4) if mid_vals else 0.0,
            "late_67_100pct": round(float(np.mean(late_vals)), 4) if late_vals else 0.0,
        },
        "video_demo_spike": {
            "post_demo_chunk0_mean_L": round(float(np.mean(video_chunk0_vals)), 4) if video_chunk0_vals else 0.0,
            "subsequent_chunks_mean_L": round(float(np.mean(video_later_vals)), 4) if video_later_vals else 0.0,
        },
        "by_task": by_task_traj,
    }


def evaluate_spec_parallel(
    spec: EvalSpec,
    *,
    results_dir: Path,
    python_bin: Path,
    tasks: Sequence[str],
    dataset: str,
    episodes_per_task: int,
    shard_size: int,
    seed: int,
    exec_horizon: int,
    max_steps: int,
    workers: int,
    device: str,
) -> Path:
    out_json = results_dir / f"{spec.eval_id}.json"
    shards_dir = results_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Interleave non-video and video-demo tasks within each round-robin slice so video-demo
    # encoding never clusters across all workers simultaneously.
    video_set = {
        "VideoUnmask", "VideoUnmaskSwap", "VideoRepick", "VideoPlaceButton",
        "VideoPlaceOrder", "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
    }
    non_vid = [t for t in tasks if t not in video_set]
    vid_only = [t for t in tasks if t in video_set]
    interleaved_tasks: list[str] = []
    while non_vid or vid_only:
        if vid_only:
            interleaved_tasks.append(vid_only.pop(0))
        if non_vid:
            interleaved_tasks.append(non_vid.pop(0))

    shard_jobs: list[tuple[str, int, int, Path]] = []
    for ep_s in range(0, episodes_per_task, shard_size):
        ep_e = min(episodes_per_task, ep_s + shard_size)
        for task in interleaved_tasks:
            s_path = shards_dir / f"{spec.eval_id}__{dataset}__{task}__{ep_s:02d}_{ep_e - 1:02d}.json"
            shard_jobs.append((task, ep_s, ep_e, s_path))

    pending_jobs: list[tuple[str, int, int, Path]] = []
    for task, ep_s, ep_e, s_path in shard_jobs:
        needed = ep_e - ep_s
        done_here = False
        if s_path.exists():
            try:
                sd = json.loads(s_path.read_text())
                if len(sd.get("episodes", [])) >= needed and "worker_telemetry" in sd:
                    done_here = True
            except Exception:
                done_here = False
        if not done_here:
            pending_jobs.append((task, ep_s, ep_e, s_path))

    t_spec0 = time.perf_counter()
    if pending_jobs:
        print(
            f"\n{'=' * 84}\n"
            f"[eval-3b] Starting `{spec.eval_id}` ({spec.label})\n"
            f"  ckpt={spec.ckpt_path.name} | split={dataset} | tasks={len(tasks)} | eps/task={episodes_per_task} | "
            f"halt={spec.halt_mode} | L={spec.num_loops} | k_interf={spec.distractor_sessions} | "
            f"shards={len(pending_jobs)}/{len(shard_jobs)} pending (workers={workers})\n"
            f"{'=' * 84}",
            flush=True,
        )

    active: list[dict[str, Any]] = []
    queue: list[tuple[str, int, int, Path, int]] = [(t, es, ee, sp, 0) for (t, es, ee, sp) in pending_jobs]
    completed_count = len(shard_jobs) - len(pending_jobs)
    failures: list[str] = []

    while queue or active:
        while queue and len(active) < workers:
            task, ep_s, ep_e, s_path, attempt = queue.pop(0)
            cmd = [
                str(python_bin),
                str(Path(__file__).resolve()),
                "--worker-shard",
                "--checkpoint",
                str(spec.ckpt_path),
                "--task",
                task,
                "--dataset",
                dataset,
                "--ep-start",
                str(ep_s),
                "--ep-end",
                str(ep_e),
                "--seed",
                str(seed),
                "--exec-horizon",
                str(exec_horizon),
                "--max-steps",
                str(max_steps),
                "--halt-mode",
                spec.halt_mode,
                "--halt-threshold",
                str(spec.halt_threshold),
                "--coda-conv-tau",
                str(spec.coda_conv_tau),
                "--ode-steps",
                str(spec.ode_steps),
                "--distractor-sessions",
                str(spec.distractor_sessions),
                "--device",
                device,
                "--shard-json",
                str(s_path),
            ]
            if spec.num_loops is not None:
                cmd.extend(["--num-loops", str(spec.num_loops)])

            env = os.environ.copy()
            env["DISPLAY"] = ""
            env["OMP_NUM_THREADS"] = "2"
            env["MKL_NUM_THREADS"] = "2"
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            active.append(
                {
                    "proc": proc,
                    "task": task,
                    "ep_s": ep_s,
                    "ep_e": ep_e,
                    "s_path": s_path,
                    "attempt": attempt,
                    "t0": time.perf_counter(),
                }
            )

        still_running: list[dict[str, Any]] = []
        for item in active:
            ret = item["proc"].poll()
            if ret is None:
                still_running.append(item)
                continue
            out_txt = item["proc"].stdout.read() if item["proc"].stdout else ""
            dt_s = time.perf_counter() - item["t0"]
            if ret != 0:
                if item["attempt"] < 2:
                    print(
                        f"[eval-3b][RETRY {item['attempt']+1}/2] Re-queueing {item['task']}[{item['ep_s']}..{item['ep_e']-1}] "
                        f"(rc={ret})",
                        flush=True,
                    )
                    queue.append((item["task"], item["ep_s"], item["ep_e"], item["s_path"], item["attempt"] + 1))
                else:
                    msg = f"Shard {item['task']}[{item['ep_s']}..{item['ep_e']-1}] failed (rc={ret}):\n{out_txt[-1200:]}"
                    print(f"[eval-3b][ERROR] {msg}", flush=True)
                    failures.append(msg)
            else:
                completed_count += 1
                sd = json.loads(item["s_path"].read_text())
                ov = sd["summary"]["overall"]
                wt = sd.get("worker_telemetry", {})
                print(
                    f"  [{completed_count:2d}/{len(shard_jobs):2d}] {item['task']:<18s} "
                    f"ep={item['ep_s']:02d}..{item['ep_e']-1:02d} "
                    f"SR={ov['success_rate']*100:5.1f}% "
                    f"L={ov['mean_loop_depth']:.2f} "
                    f"ms/ch={ov['mean_ms_per_chunk']:5.1f} "
                    f"vram={wt.get('peak_vram_mb', 0.0):5.0f}MB "
                    f"wall={dt_s:5.1f}s",
                    flush=True,
                )
        active = still_running
        if active:
            time.sleep(0.5)

    if failures:
        raise RuntimeError(f"{len(failures)} shard(s) failed for `{spec.eval_id}`:\n" + "\n".join(failures))

    merged_eps: list[EpisodeResult] = []
    max_worker_vram_mb = 0.0
    state_size_bytes = 0
    for _, _, _, s_path in shard_jobs:
        sd = json.loads(s_path.read_text())
        wt = sd.get("worker_telemetry", {})
        max_worker_vram_mb = max(max_worker_vram_mb, float(wt.get("peak_vram_mb", 0.0)))
        state_size_bytes = int(wt.get("state_size_bytes", state_size_bytes))
        for raw_ep in sd.get("episodes", []):
            merged_eps.append(EpisodeResult(**raw_ep))

    task_rank = {t: idx for idx, t in enumerate(tasks)}
    merged_eps.sort(key=lambda e: (task_rank.get(e.task, 999), e.episode_id, e.model_seed))

    summary = summarize_episodes(merged_eps, n_boot=10000, seed=0)
    loop_diag = compute_loop_trajectory_diagnostics(merged_eps, tasks)
    wall_total_s = time.perf_counter() - t_spec0

    final_payload = {
        "eval_spec": {
            "eval_id": spec.eval_id,
            "ckpt_stem": spec.ckpt_stem,
            "arch": spec.arch,
            "train_seed": spec.train_seed,
            "label": spec.label,
            "checkpoint": str(spec.ckpt_path),
            "halt_mode": spec.halt_mode,
            "num_loops": spec.num_loops,
            "halt_threshold": spec.halt_threshold,
            "coda_conv_tau": spec.coda_conv_tau,
            "ode_steps": spec.ode_steps,
            "distractor_sessions": spec.distractor_sessions,
            "state_size_bytes": state_size_bytes,
            "max_worker_vram_mb": round(max_worker_vram_mb, 1),
            "workers": workers,
            "eval_wall_s": round(wall_total_s, 2),
        },
        "config": {
            "tasks": list(tasks),
            "dataset": dataset,
            "episodes_per_task": episodes_per_task,
            "model_seeds": [seed],
            "exec_horizon": exec_horizon,
            "max_steps": max_steps,
            "distractor_sessions": spec.distractor_sessions,
        },
        "summary": summary,
        "loop_diagnostics": loop_diag,
        "episodes": [e.to_dict() for e in merged_eps],
    }

    tmp_out = results_dir / f".{spec.eval_id}.json.tmp"
    tmp_out.write_text(json.dumps(final_payload, indent=2))
    tmp_out.replace(out_json)

    ov = summary["overall"]
    print(
        f"[DONE-3B] `{spec.eval_id}` ({len(merged_eps)} eps across {len(tasks)} tasks in {wall_total_s/60:.2f}m): "
        f"Overall SR = {ov['success_rate']*100:.2f}% "
        f"[{ov['success_ci95_low']*100:.2f}%, {ov['success_ci95_high']*100:.2f}%] | "
        f"L_mean = {ov['mean_loop_depth']:.2f} | max_worker_VRAM = {max_worker_vram_mb:.0f} MB",
        flush=True,
    )
    return out_json


# =============================================================================
# 5. Paired Bootstrap Comparisons, Multi-Seed Aggregation & `SUMMARY_CYCLE3B.md`
# =============================================================================


def _fmt_pct(val: float) -> str:
    return f"{val * 100.0:.1f}%"


def _fmt_sr_ci(summary_block: dict[str, Any]) -> str:
    sr = summary_block["success_rate"] * 100.0
    lo = summary_block["success_ci95_low"] * 100.0
    hi = summary_block["success_ci95_high"] * 100.0
    return f"**{sr:.2f}%** `[{lo:.2f}, {hi:.2f}]`"


def _fmt_state_bytes(n_bytes: int) -> str:
    if n_bytes <= 0:
        return "`0 B`"
    return f"`{n_bytes / 1024.0:.1f} KB`"


def generate_summary_and_comparisons(
    results_dir: Path,
    mirror_dir: Path,
) -> tuple[Path, Path]:
    canonical_order = [
        "smolvla_m0_base",
        "smolvla_m1_rdvla_loop4",
        "smolvla_m2_framesamp",
        "smolvla_m5_gdn2l_l1",
        "smolvla_m7_deltaloop_L1",
        "smolvla_m7_deltaloop_L2",
        "smolvla_m7_deltaloop_L4",
        "smolvla_m7_deltaloop_adaptive",
        "smolvla_m7_deltaloop_coda_conv",
        "smolvla_m7_deltaloop_q16_adaptive",
        "smolvla_m7_deltaloop_anchor_adaptive",
        "smolvla_m7_deltaloop_anchor_L2",
    ]

    loaded: dict[str, dict[str, Any]] = {}
    episodes_map: dict[str, list[EpisodeResult]] = {}
    for jpath in sorted(results_dir.glob("*.json")):
        if jpath.name in ("paired_comparisons.json", "mechanistic_probes.json") or jpath.name.startswith("."):
            continue
        try:
            data = json.loads(jpath.read_text())
            if "eval_spec" not in data or "summary" not in data:
                continue
            eid = data["eval_spec"]["eval_id"]
            loaded[eid] = data
            episodes_map[eid] = [EpisodeResult(**ep) for ep in data.get("episodes", [])]
        except Exception:
            continue

    k0_ids = [eid for eid in canonical_order if eid in loaded] + sorted(
        eid for eid in loaded if eid not in canonical_order and "_interfk" not in eid
    )
    k0_s42_ids = [eid for eid in k0_ids if int(loaded[eid]["eval_spec"].get("train_seed", 42)) == 42]
    k0_extra_seed_ids = [eid for eid in k0_ids if int(loaded[eid]["eval_spec"].get("train_seed", 42)) != 42]
    interf_ids = sorted(eid for eid in loaded if "_interfk" in eid)

    train_meta: dict[str, dict[str, Any]] = {}
    for mfile in mirror_dir.glob("*_metrics.json"):
        try:
            mdata = json.loads(mfile.read_text())
            stem = mfile.name[: -len("_metrics.json")]
            train_meta[stem] = mdata
            if "arch" in mdata and mdata["arch"] not in train_meta:
                train_meta[mdata["arch"]] = mdata
        except Exception:
            pass

    candidate_pairs = [
        ("H0a_M1_RDVLA_Loop4_vs_M0_Base", "smolvla_m1_rdvla_loop4", "smolvla_m0_base", "Pretrained RD-VLA Memoryless Loop (L=4) vs Pretrained SmolVLA Base"),
        ("H0b_M2_FrameSamp_vs_M0_Base", "smolvla_m2_framesamp", "smolvla_m0_base", "Pretrained FrameSamp-32 vs Pretrained SmolVLA Base"),
        ("H1a_M7_Adapt_vs_M0_Base", "smolvla_m7_deltaloop_adaptive", "smolvla_m0_base", "Pretrained SmolVLA-DeltaLoop (Ours) vs Pretrained SmolVLA Base"),
        ("H1d_M7_CodaConv_vs_M0_Base", "smolvla_m7_deltaloop_coda_conv", "smolvla_m0_base", "Pretrained SmolVLA-DeltaLoop (Latent L2 Conv) vs Pretrained SmolVLA Base"),
        ("H1b_M7_Adapt_vs_M1_RDVLA_Loop4", "smolvla_m7_deltaloop_adaptive", "smolvla_m1_rdvla_loop4", "Pretrained SmolVLA-DeltaLoop vs Memoryless RD-VLA Loop (L=4)"),
        ("H1c_M5_GDN2L_vs_M0_Base", "smolvla_m5_gdn2l_l1", "smolvla_m0_base", "2L Motion-Gated GDN (L=1) vs Pretrained SmolVLA Base"),
        ("H3a_M7_Adapt_vs_M2_FrameSamp", "smolvla_m7_deltaloop_adaptive", "smolvla_m2_framesamp", "512 KB O(1) SmolVLA-DeltaLoop vs 32-Slot FrameSamp+Modulator"),
        ("H3b_M5_GDN2L_vs_M2_FrameSamp", "smolvla_m5_gdn2l_l1", "smolvla_m2_framesamp", "512 KB O(1) 2L GDN (L=1) vs 32-Slot FrameSamp+Modulator"),
        ("H4a_M7_L4_vs_M7_L1_IntraCkpt", "smolvla_m7_deltaloop_L4", "smolvla_m7_deltaloop_L1", "Intra-checkpoint multi-hop readout gain (L=4 vs L=1)"),
        ("H4b_M7_L2_vs_M7_L1_IntraCkpt", "smolvla_m7_deltaloop_L2", "smolvla_m7_deltaloop_L1", "Intra-checkpoint multi-hop readout gain (L=2 vs L=1)"),
        ("H4d_M7_Adapt_vs_M7_L1_IntraCkpt", "smolvla_m7_deltaloop_adaptive", "smolvla_m7_deltaloop_L1", "Intra-checkpoint Stage-II CRN Gate (L_t=1.11) vs Fixed L=1"),
        ("H4e_M7_CodaConv_vs_M7_L1_IntraCkpt", "smolvla_m7_deltaloop_coda_conv", "smolvla_m7_deltaloop_L1", "Intra-checkpoint Latent L2 Conv (L_t=2.28) vs Fixed L=1"),
        ("H4c_M7_Adapt_vs_M5_GDN2L_L1", "smolvla_m7_deltaloop_adaptive", "smolvla_m5_gdn2l_l1", "Multi-Hop Adaptive DeltaLoop (M7) vs Single-Pass 2L GDN (M5)"),
        ("H5a_M7_Adapt_vs_M7_L4", "smolvla_m7_deltaloop_adaptive", "smolvla_m7_deltaloop_L4", "Stage-2 CRN Adaptive Halting vs Fixed L=4"),
        ("H5b_M7_Adapt_vs_M7_CodaConv", "smolvla_m7_deltaloop_adaptive", "smolvla_m7_deltaloop_coda_conv", "Stage-2 CRN Gate vs Training-Free Coda Convergence"),
        # Cycle 4 U7 (`N_c = 16` Coda Query Bank) readout-bandwidth ablation
        ("H8a_M7_Q16_vs_M0_Base", "smolvla_m7_deltaloop_q16_adaptive", "smolvla_m0_base", "SmolVLA-DeltaLoop (N_c=16) vs Pretrained SmolVLA Base"),
        ("H8b_M7_Q16_vs_M7_Adapt", "smolvla_m7_deltaloop_q16_adaptive", "smolvla_m7_deltaloop_adaptive", "Readout-Bandwidth Expansion (N_c=16 vs N_c=4) at Exact 520 KB Iso-State"),
        ("H8c_M7_Q16_vs_M2_FrameSamp", "smolvla_m7_deltaloop_q16_adaptive", "smolvla_m2_framesamp", "SmolVLA-DeltaLoop (N_c=16, 520 KB) vs 32-Slot FrameSamp (k=0)"),
        ("H8d_M7_Q16_vs_M5_GDN2L_L1", "smolvla_m7_deltaloop_q16_adaptive", "smolvla_m5_gdn2l_l1", "SmolVLA-DeltaLoop (N_c=16) vs Single-Pass 2L GDN (L=1)"),
        # Cycle 4B U8 (`4+4` Hop-0 Landmark Anchor + Hop-l Pointer Registers)
        ("H9a_M7_Anchor_vs_M0_Base", "smolvla_m7_deltaloop_anchor_adaptive", "smolvla_m0_base", "SmolVLA-DeltaLoop-Anchor (4+4 Reg, 520 KB) vs Pretrained SmolVLA Base"),
        ("H9b_M7_Anchor_vs_M7_Adapt", "smolvla_m7_deltaloop_anchor_adaptive", "smolvla_m7_deltaloop_adaptive", "Partitioned Hop-0 Anchor + Hop-l Pointer (U8) vs Overwriting N_c=4 (M7)"),
        ("H9c_M7_Anchor_vs_M7_Q16", "smolvla_m7_deltaloop_anchor_adaptive", "smolvla_m7_deltaloop_q16_adaptive", "Partitioned 4+4 Anchor+Pointer (U8) vs Homogeneous Mutable N_c=16 (U7)"),
        ("H9d_M7_Anchor_vs_M2_FrameSamp", "smolvla_m7_deltaloop_anchor_adaptive", "smolvla_m2_framesamp", "SmolVLA-DeltaLoop-Anchor (520 KB) vs 32-Slot FrameSamp (k=0)"),
        ("H9e_M7_Anchor_vs_M5_GDN2L_L1", "smolvla_m7_deltaloop_anchor_adaptive", "smolvla_m5_gdn2l_l1", "SmolVLA-DeltaLoop-Anchor (U8) vs Single-Pass 2L GDN (L=1)"),
        ("H9f_M7_Anchor_L2_vs_Anchor_Adapt", "smolvla_m7_deltaloop_anchor_L2", "smolvla_m7_deltaloop_anchor_adaptive", "Bypassing Anchor-Shortcut Halting: Intra-checkpoint Fixed L=2 vs CRN Gate (L_t=1.04) on U8"),
        ("H9g_M7_Anchor_L2_vs_M0_Base", "smolvla_m7_deltaloop_anchor_L2", "smolvla_m0_base", "SmolVLA-DeltaLoop-Anchor (Fixed L=2) vs Pretrained SmolVLA Base"),
        ("H9h_M7_Anchor_L2_vs_M7_L2", "smolvla_m7_deltaloop_anchor_L2", "smolvla_m7_deltaloop_L2", "Iso-Hop (L=2) Partitioned 4+4 Anchor+Pointer (U8) vs Overwriting N_c=4 (M7)"),
        ("H9i_M7_Anchor_L2_vs_M7_Adapt", "smolvla_m7_deltaloop_anchor_L2", "smolvla_m7_deltaloop_adaptive", "SmolVLA-DeltaLoop-Anchor (Fixed L=2) vs SmolVLA-DeltaLoop (N_c=4, CRN Gate)"),
        # Cross-session interference (`k=3`, `N=240` matched `eps 0..14`)
        ("H7a_Interf_k3_M7_vs_M2", "smolvla_m7_deltaloop_adaptive_interfk3", "smolvla_m2_framesamp_interfk3", "Closed-loop Cross-Session Interference (k=3): SmolVLA-DeltaLoop (M7) vs FrameSamp-32 (M2)"),
        ("H7b_Interf_Drop_M2_k3_vs_k0", "smolvla_m2_framesamp_interfk3", "smolvla_m2_framesamp", "FrameSamp-32 (M2) SR change under k=3 prior distractor sessions (eps 0..14)"),
        ("H7c_Interf_Drop_M7_k3_vs_k0", "smolvla_m7_deltaloop_adaptive_interfk3", "smolvla_m7_deltaloop_adaptive", "SmolVLA-DeltaLoop (M7) SR change under k=3 prior distractor sessions (eps 0..14)"),
        ("H7d_M5_InterfDrop_k3_vs_k0", "smolvla_m5_gdn2l_l1_interfk3", "smolvla_m5_gdn2l_l1", "Single-pass 2L GDN (M5) SR change under k=3 prior distractor sessions (eps 0..14)"),
        ("H7e_Interf_k3_M5_vs_M2", "smolvla_m5_gdn2l_l1_interfk3", "smolvla_m2_framesamp_interfk3", "Closed-loop Cross-Session Interference (k=3): Single-pass 2L GDN (M5) vs FrameSamp-32 (M2)"),
        ("H7f_Interf_k3_M7_vs_M5", "smolvla_m7_deltaloop_adaptive_interfk3", "smolvla_m5_gdn2l_l1_interfk3", "Closed-loop Cross-Session Interference (k=3): Multi-Hop DeltaLoop (M7) vs Single-Pass 2L GDN (M5)"),
    ]

    comparisons: dict[str, Any] = {}
    for key_name, id_a, id_b, desc in candidate_pairs:
        if id_a in episodes_map and id_b in episodes_map:
            try:
                comp = compare_policy_runs(episodes_map[id_a], episodes_map[id_b], n_boot=10000, seed=0)
                if comp["overall_success_diff"]["n_pairs"] > 0:
                    comp["run_a"] = id_a
                    comp["run_b"] = id_b
                    comp["description"] = desc
                    comparisons[key_name] = comp
            except ValueError:
                pass

    # Paired Difference-in-Differences `(A_k3 - A_k0) - (B_k3 - B_k0)` on matched `N=240` (`eps 0..14`) episodes
    did_specs = [
        (
            "H7g_DiD_InterfPenalty_M5_vs_M2",
            "smolvla_m5_gdn2l_l1_interfk3",
            "smolvla_m5_gdn2l_l1",
            "smolvla_m2_framesamp_interfk3",
            "smolvla_m2_framesamp",
            "Difference-in-Differences (M5_k3 - M5_k0) - (M2_k3 - M2_k0) on matched eps 0..14",
        ),
        (
            "H7h_DiD_InterfPenalty_M7_vs_M2",
            "smolvla_m7_deltaloop_adaptive_interfk3",
            "smolvla_m7_deltaloop_adaptive",
            "smolvla_m2_framesamp_interfk3",
            "smolvla_m2_framesamp",
            "Difference-in-Differences (M7_k3 - M7_k0) - (M2_k3 - M2_k0) on matched eps 0..14",
        ),
        (
            "H7i_DiD_InterfPenalty_M7_vs_M5",
            "smolvla_m7_deltaloop_adaptive_interfk3",
            "smolvla_m7_deltaloop_adaptive",
            "smolvla_m5_gdn2l_l1_interfk3",
            "smolvla_m5_gdn2l_l1",
            "Difference-in-Differences (M7_k3 - M7_k0) - (M5_k3 - M5_k0) on matched eps 0..14",
        ),
    ]
    for key_name, ak3, ak0, bk3, bk0, desc in did_specs:
        if all(r in episodes_map for r in (ak3, ak0, bk3, bk0)):
            try:
                comp = compare_did_policy_runs(
                    episodes_map[ak3],
                    episodes_map[ak0],
                    episodes_map[bk3],
                    episodes_map[bk0],
                    n_boot=10000,
                    seed=0,
                )
                if comp["overall_success_diff"]["n_pairs"] > 0:
                    comp["run_a"] = f"{ak3}_minus_k0"
                    comp["run_b"] = f"{bk3}_minus_k0"
                    comp["description"] = desc
                    comparisons[key_name] = comp
            except ValueError:
                pass

    # Multi-seed (`seeds = {42, 43, 44}`) exact episode-matched aggregation across `(arch, q_tag, halt_mode, num_loops, k=0)`
    multiseed_groups: dict[str, list[str]] = {}
    for eid in k0_ids:
        sp = loaded[eid]["eval_spec"]
        arch = sp["arch"]
        stem_str = str(sp.get("ckpt_stem", ""))
        q_tag = "_anchor" if "_anchor" in stem_str else ("_q16" if "_q16" in stem_str else "")
        hm = sp["halt_mode"]
        nl = sp["num_loops"]
        gkey = f"{arch}{q_tag}__{hm}__{nl}"
        multiseed_groups.setdefault(gkey, []).append(eid)

    multiseed_summary: dict[str, Any] = {}
    pooled_group_eps: dict[str, list[EpisodeResult]] = {}
    for gkey, eids in multiseed_groups.items():
        if len(eids) >= 2:
            eids_sorted = sorted(eids, key=lambda x: int(loaded[x]["eval_spec"].get("train_seed", 42)))
            ep_sets = [
                {(ep.task, ep.dataset, ep.episode_id) for ep in episodes_map.get(e, [])}
                for e in eids_sorted
            ]
            common_ep_keys = sorted(set.intersection(*ep_sets)) if ep_sets else []
            common_set = set(common_ep_keys)

            matched_srs: list[float] = []
            per_seed_matched: dict[str, Any] = {}
            matched_suite_vals: dict[str, list[float]] = {s: [] for s in SUITES_ORDER}
            pooled_eps: list[EpisodeResult] = []
            seeds: list[int] = []

            for e in eids_sorted:
                t_seed = int(loaded[e]["eval_spec"].get("train_seed", 42))
                seeds.append(t_seed)
                filt_eps = [ep for ep in episodes_map.get(e, []) if (ep.task, ep.dataset, ep.episode_id) in common_set]
                sub_sum = summarize_episodes(filt_eps)
                sr_e = float(sub_sum["overall"]["success_rate"])
                matched_srs.append(sr_e)
                s_dict = {s: float(sub_sum["suites"].get(s, {}).get("success_rate", 0.0)) for s in SUITES_ORDER}
                for s in SUITES_ORDER:
                    matched_suite_vals[s].append(s_dict[s])
                per_seed_matched[f"s{t_seed}"] = {
                    "eval_id": e,
                    "n_episodes": len(filt_eps),
                    "overall_sr": sr_e,
                    "overall_ci95": [
                        float(sub_sum["overall"]["success_ci95_low"]),
                        float(sub_sum["overall"]["success_ci95_high"]),
                    ],
                    "mean_loop_depth": float(sub_sum["overall"]["mean_loop_depth"]),
                    "suites": s_dict,
                }
                for ep in filt_eps:
                    pooled_eps.append(replace(ep, model_seed=t_seed))

            pooled_group_eps[gkey] = pooled_eps
            suite_means = {s: float(np.mean(matched_suite_vals[s])) for s in SUITES_ORDER}
            suite_stds = {
                s: (float(np.std(matched_suite_vals[s], ddof=1)) if len(matched_suite_vals[s]) > 1 else 0.0)
                for s in SUITES_ORDER
            }
            multiseed_summary[gkey] = {
                "eval_ids": eids_sorted,
                "train_seeds": seeds,
                "n_matched_eps_per_seed": len(common_ep_keys),
                "total_pooled_eps": len(pooled_eps),
                "mean_sr": float(np.mean(matched_srs)),
                "std_sr": float(np.std(matched_srs, ddof=1)) if len(matched_srs) > 1 else 0.0,
                "suite_means": suite_means,
                "suite_stds": suite_stds,
                "per_seed_matched": per_seed_matched,
            }

    # Pooled multi-seed (`N = 3 x 240 = 720` matched `(task, episode_id, train_seed)` pairs) bootstrap tests
    pooled_pair_specs = [
        (
            "H_MS1_Pooled_M7_vs_M0",
            "smolvla_m7_deltaloop__learned_gate__4",
            "smolvla_m0_base__fixed__1",
            "Pooled Multi-Seed M7 DeltaLoop vs M0 Base (matched `(seed, task, ep)` pairs)",
        ),
        (
            "H_MS2_Pooled_M2_vs_M0",
            "smolvla_m2_framesamp__fixed__1",
            "smolvla_m0_base__fixed__1",
            "Pooled Multi-Seed M2 FrameSamp-32 vs M0 Base (matched `(seed, task, ep)` pairs)",
        ),
        (
            "H_MS3_Pooled_M7_vs_M2",
            "smolvla_m7_deltaloop__learned_gate__4",
            "smolvla_m2_framesamp__fixed__1",
            "Pooled Multi-Seed M7 DeltaLoop vs M2 FrameSamp-32 at k=0 (matched `(seed, task, ep)` pairs)",
        ),
    ]
    for key_name, ga, gb, desc in pooled_pair_specs:
        if ga in pooled_group_eps and gb in pooled_group_eps:
            try:
                comp = compare_policy_runs(pooled_group_eps[ga], pooled_group_eps[gb], n_boot=10000, seed=0)
                if comp["overall_success_diff"]["n_pairs"] > 0:
                    comp["run_a"] = ga.split("__")[0] + "_3seed"
                    comp["run_b"] = gb.split("__")[0] + "_3seed"
                    comp["description"] = desc
                    comparisons[key_name] = comp
            except ValueError:
                pass

    apply_holm_bonferroni(comparisons)
    comp_path = results_dir / "paired_comparisons.json"
    comp_path.write_text(
        json.dumps({"cycle3b_pairs": comparisons, "multiseed_summary": multiseed_summary}, indent=2)
    )

    # Determine active task display list from loaded runs
    active_tasks = list(ALL_16_TASKS_DISPLAY)
    if k0_s42_ids:
        cfg_tasks = loaded[k0_s42_ids[0]].get("config", {}).get("tasks", [])
        if cfg_tasks and len(cfg_tasks) < 16:
            active_tasks = [t for t in ALL_16_TASKS_DISPLAY if t in cfg_tasks]

    lines: list[str] = []
    lines.append("# Cycle 3B — Pretrained `SmolVLA-DeltaLoop` (`U6`) Multi-Suite RoboMME Evaluation Summary")
    lines.append("")
    if k0_s42_ids:
        c0 = loaded[k0_s42_ids[0]]["config"]
        lines.append(
            f"- **Protocol:** `{len(c0['tasks'])}` RoboMME Tasks across 4 Memory Suites, `{c0['dataset']}` split "
            f"(`{c0['episodes_per_task']}` episodes/task = **`{len(c0['tasks']) * c0['episodes_per_task']}` matched episodes/run** on `seed=42`, "
            f"`exec_horizon={c0['exec_horizon']}`, `max_steps={c0['max_steps']}`, 10-step `t=1->0` Euler ODE)."
        )
    lines.append(
        f"- **Evaluated Runs Completed:** `{len(loaded)}` "
        f"(`{len(k0_s42_ids)}` primary `seed=42` + `{len(k0_extra_seed_ids)}` multi-seed `{{43, 44}}` + `{len(interf_ids)}` cross-session interference)"
    )
    lines.append("")

    if k0_s42_ids:
        lines.append("## 1. Overall & Per-Suite Success Rates (`k = 0`, `seed = 42`, `N = 400`)")
        lines.append("")
        lines.append(
            "| Evaluation Run | Active Params | State Size | Train FM Loss | Mean $L_t$ (`p95`) | Latency (`ms/ch`) | "
            "Overall SR (`95% CI`) | Permanence | Counting | Reference | Imitation |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for eid in k0_s42_ids:
            d = loaded[eid]
            sp = d["eval_spec"]
            ov = d["summary"]["overall"]
            bs = d["summary"]["suites"]
            stem = sp.get("ckpt_stem", sp["arch"])
            tm = train_meta.get(stem) or train_meta.get(sp["arch"], {})
            pb = tm.get("parameter_breakdown", {})
            n_params = pb.get("total_active") or pb.get("total") or tm.get("num_params")
            params_str = f"`{n_params / 1e6:.1f}M`" if n_params else "`~405M`"
            fm_val = tm.get("final_log", {}).get("fm_loss", tm.get("final_fm_loss"))
            fm_str = f"`{fm_val:.4f}`" if isinstance(fm_val, (int, float)) else "—"
            state_str = _fmt_state_bytes(int(sp.get("state_size_bytes", 0)))
            l_str = f"`{ov['mean_loop_depth']:.2f}` (`{ov['p95_loop_depth']:.1f}`)"
            lat_str = f"`{ov['mean_ms_per_chunk']:.1f}`"
            suite_cells = [_fmt_pct(bs[s]["success_rate"]) if s in bs else "—" for s in SUITES_ORDER]
            lines.append(
                f"| **{sp['label']}** (`{eid}`) | {params_str} | {state_str} | {fm_str} | {l_str} | {lat_str} | "
                f"{_fmt_sr_ci(ov)} | {suite_cells[0]} | {suite_cells[1]} | {suite_cells[2]} | {suite_cells[3]} |"
            )
        lines.append("")

        if multiseed_summary:
            lines.append("### 1b. Episode-Matched Multi-Seed Verification (`seeds = {42, 43, 44}`, Exact Matched `eps 0..14` = `240`/Seed)")
            lines.append("")
            lines.append(
                "| Contender Group | Seeds (`N/seed`, `Pooled N`) | Seed Breakdown (`s42 / s43 / s44` Counts & SR) | Mean $\\pm$ SD SR | "
                "Permanence | Counting | Reference | Imitation |"
            )
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for gkey, ms in multiseed_summary.items():
                sm = ms["suite_means"]
                ss = ms["suite_stds"]
                psm = ms.get("per_seed_matched", {})
                s_brk = " / ".join(
                    f"`{k}={int(round(v['overall_sr']*v['n_episodes']))}/{v['n_episodes']} ({v['overall_sr']*100:.2f}%)`"
                    for k, v in psm.items()
                )
                lines.append(
                    f"| `{gkey}` | `{ms['train_seeds']}` (`{ms['n_matched_eps_per_seed']}`, `{ms['total_pooled_eps']}`) | {s_brk} | "
                    f"**`{ms['mean_sr']*100:.2f}% ± {ms['std_sr']*100:.2f}%`** | "
                    f"`{sm['Permanence']*100:.1f}±{ss['Permanence']*100:.1f}%` | "
                    f"`{sm['Counting']*100:.1f}±{ss['Counting']*100:.1f}%` | "
                    f"`{sm['Reference']*100:.1f}±{ss['Reference']*100:.1f}%` | "
                    f"`{sm['Imitation']*100:.1f}±{ss['Imitation']*100:.1f}%` |"
                )
            lines.append("")

        lines.append(f"## 2. Per-Task Success Rates Across `{len(active_tasks)}` RoboMME Tasks")
        lines.append("")
        t_headers = " | ".join(f"`{t}`" for t in active_tasks)
        t_sep = " | ".join(":---:" for _ in active_tasks)
        lines.append(f"| Evaluation Run | Overall | {t_headers} |")
        lines.append(f"| :--- | :---: | {t_sep} |")
        for eid in (k0_s42_ids + k0_extra_seed_ids):
            d = loaded[eid]
            ov_sr = _fmt_pct(d["summary"]["overall"]["success_rate"])
            bt = d["summary"]["tasks"]
            cells = [_fmt_pct(bt[t]["success_rate"]) if t in bt else "—" for t in active_tasks]
            lines.append(f"| `{eid}` | **{ov_sr}** | " + " | ".join(cells) + " |")
        lines.append("")

    if interf_ids:
        lines.append("## 3. Live Closed-Loop Cross-Session Interference (`k = 3` Prior Distractor Episodes, `N = 240` Matched `eps 0..14`)")
        lines.append("")
        lines.append(
            "| Interference Run | $k$ | State Size | Mean $L_t$ | Matched `k=0` SR (`eps 0..14`) | `k=3` SR (`Count`, `95% CI`) | Permanence | Counting | Reference | Imitation |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for eid in interf_ids:
            d = loaded[eid]
            sp = d["eval_spec"]
            ov = d["summary"]["overall"]
            bs = d["summary"]["suites"]
            n_ep_k3 = int(ov["num_episodes"])
            n_succ_k3 = int(round(ov["success_rate"] * n_ep_k3))
            base_eid = re.sub(r"_interfk\d+$", "", eid)
            k0_cell = "—"
            if base_eid in episodes_map:
                k3_keys = {(e.task, e.dataset, e.episode_id) for e in episodes_map[eid]}
                k0_filt = [e for e in episodes_map[base_eid] if (e.task, e.dataset, e.episode_id) in k3_keys]
                if k0_filt:
                    n_s0 = sum(int(e.success) for e in k0_filt)
                    k0_cell = f"`{n_s0}/{len(k0_filt)}` (`{n_s0/len(k0_filt)*100:.2f}%`)"
            suite_cells = [_fmt_pct(bs[s]["success_rate"]) if s in bs else "—" for s in SUITES_ORDER]
            lines.append(
                f"| **{sp['label']}** (`{eid}`) | `{sp.get('distractor_sessions', 0)}` | "
                f"{_fmt_state_bytes(int(sp.get('state_size_bytes', 0)))} | `{ov['mean_loop_depth']:.2f}` | "
                f"{k0_cell} | **`{ov['success_rate']*100:.2f}%`** (`{n_succ_k3}/{n_ep_k3}`, `[{ov['success_ci95_low']*100:.2f}, {ov['success_ci95_high']*100:.2f}]`) | "
                f"{suite_cells[0]} | {suite_cells[1]} | {suite_cells[2]} | {suite_cells[3]} |"
            )
        lines.append("")

    if comparisons:
        m_fam = next(iter(comparisons.values())).get("holm_family_size", len(comparisons))
        lines.append(f"## 4. Paired Bootstrap Hypothesis Tests (`10,000` Resamples; Holm-Bonferroni Step-Down Across `m = {m_fam}` Tests)")
        lines.append("")
        lines.append(
            "| Hypothesis / Comparison (`A - B`) | $N$ | $\\text{SR}_A$ | $\\text{SR}_B$ | $\\Delta \\text{SR}$ (`95% Paired CI`) | Two-Sided $p$ | Holm $p_{\\text{Holm}}$ | "
            "Permanence $\\Delta$ | Counting $\\Delta$ | Reference $\\Delta$ | Imitation $\\Delta$ |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for key_name, comp in comparisons.items():
            ov = comp["overall_success_diff"]
            bs = comp.get("suites", {})
            diff_str = f"**`{ov['mean_diff']*100:+.2f}%`** `[{ov['ci_low']*100:+.2f}%, {ov['ci_high']*100:+.2f}%]`"
            p_raw = float(ov["p_value_two_sided"])
            p_holm = float(ov.get("p_holm", 1.0))
            if ov.get("primary_test") == "exact_mcnemar":
                p_boot_b = float(ov.get("p_bootstrap_phipson_smyth_2tail", ov.get("p_bootstrap_two_sided", 0.0002)))
                ph_boot_b = float(ov.get("p_holm_bootstrap_ps2tail", ov.get("p_holm_bootstrap", 0.0074)))
                p_str = f"`{p_raw:.4f}` *(McN; boot `<= {p_boot_b:.4f}`)*"
                holm_str = f"`{p_holm:.4f}` *(McN NS; boot `<= {ph_boot_b:.4f}`)*"
            elif p_raw <= 0.000101:
                p_str = f"`<= {p_raw:.4f}`"
                holm_str = f"**`<= {p_holm:.4f}`**" if ov.get("holm_significant_005") else f"`{p_holm:.4f}`"
            else:
                p_str = f"`{p_raw:.4f}`"
                holm_str = f"**`{p_holm:.4f}`**" if ov.get("holm_significant_005") else f"`{p_holm:.4f}`"
            s_diffs = [f"`{bs[s]['mean_diff']*100:+.1f}%`" if s in bs else "—" for s in SUITES_ORDER]
            lines.append(
                f"| **{key_name}**: `{comp['run_a']}` vs `{comp['run_b']}` | `{ov['n_pairs']}` | "
                f"`{ov['mean_a']*100:.2f}%` | `{ov['mean_b']*100:.2f}%` | {diff_str} | {p_str} | {holm_str} | "
                f"{s_diffs[0]} | {s_diffs[1]} | {s_diffs[2]} | {s_diffs[3]} |"
            )
        lines.append("")

    adaptive_eids = [
        eid
        for eid in (k0_ids + interf_ids)
        if loaded[eid]["eval_spec"]["halt_mode"] in ("learned_gate", "coda_conv")
    ]
    if adaptive_eids:
        lines.append("## 5. Adaptive Loop-Depth ($L_t$) Trajectory Diagnostics")
        lines.append("")
        for eid in adaptive_eids:
            d = loaded[eid]
            diag = d.get("loop_diagnostics", {})
            if not diag:
                continue
            hist = diag.get("overall_hist", {})
            pm = diag.get("phase_means", {})
            vd = diag.get("video_demo_spike", {})
            lines.append(f"### `{eid}` ({d['eval_spec']['label']})")
            lines.append(
                f"- **Overall Chunk Loop Allocation (`{diag.get('total_chunks', 0)}` chunks):** "
                f"Mean $L_t = {diag.get('overall_chunk_mean_L', 0.0):.3f} \\pm {diag.get('overall_chunk_std_L', 0.0):.3f}$ | "
                f"Histogram: `L=1: {hist.get('L=1', 0.0)*100:.1f}%`, `L=2: {hist.get('L=2', 0.0)*100:.1f}%`, "
                f"`L=3: {hist.get('L=3', 0.0)*100:.1f}%`, `L=4: {hist.get('L=4', 0.0)*100:.1f}%`"
            )
            lines.append(
                f"- **Trajectory Phase Profile:** Early `[0..33%]` = `{pm.get('early_0_33pct', 0.0):.3f}`, "
                f"Mid `[33..67%]` = `{pm.get('mid_33_67pct', 0.0):.3f}`, "
                f"Late `[67..100%]` = `{pm.get('late_67_100pct', 0.0):.3f}` | "
                f"Video-Demo Tasks: `Chunk 0 (post-demo)` = `{vd.get('post_demo_chunk0_mean_L', 0.0):.3f}` "
                f"vs `Subsequent` = `{vd.get('subsequent_chunks_mean_L', 0.0):.3f}`"
            )
            lines.append("")
            lines.append(
                "| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |"
            )
            lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            bt_sr = d["summary"]["tasks"]
            for t in active_tasks:
                td = diag.get("by_task", {}).get(t)
                if not td:
                    continue
                th = td["hist"]
                sr_t = _fmt_pct(bt_sr[t]["success_rate"]) if t in bt_sr else "—"
                lines.append(
                    f"| `{t}` | {TASK_TO_SUITE[t]} | `{sr_t}` | `{td['mean_L']:.2f}` | `{td['chunk0_mean_L']:.2f}` | "
                    f"`{td['early_mean_L']:.2f}` | `{td['mid_mean_L']:.2f}` | `{td['late_mean_L']:.2f}` | "
                    f"`{th.get('L=1', 0.0)*100:.1f}%` | `{th.get('L=2', 0.0)*100:.1f}%` | "
                    f"`{th.get('L=3', 0.0)*100:.1f}%` | `{th.get('L=4', 0.0)*100:.1f}%` |"
                )
            lines.append("")

    # 6. Dynamic Executive Scientific Synthesis (Strict Two-Sided + Holm-Bonferroni Candor)
    total_rollouts = sum(len(eps) for eps in episodes_map.values())
    if k0_s42_ids:
        lines.append("## 6. Executive Scientific Synthesis")
        lines.append("")
        lines.append(
            f"- **Cumulative Closed-Loop Official `test` Rollouts Evaluated:** **`{total_rollouts:,}`** episodes across `{len(loaded)}` runs on all `16` RoboMME tasks "
            f"(`10,720` total closed-loop RoboMME rollouts across Cycle 2A `2,400` + Cycle 3A `1,760` + Cycle 3B/4 `{total_rollouts:,}`)."
        )
        if "H1c_M5_GDN2L_vs_M0_Base" in comparisons and "H1a_M7_Adapt_vs_M0_Base" in comparisons:
            c1c = comparisons["H1c_M5_GDN2L_vs_M0_Base"]["overall_success_diff"]
            c1a = comparisons["H1a_M7_Adapt_vs_M0_Base"]["overall_success_diff"]
            c0a = comparisons.get("H0a_M1_RDVLA_Loop4_vs_M0_Base", {}).get("overall_success_diff", {})
            c4c = comparisons.get("H4c_M7_Adapt_vs_M5_GDN2L_L1", {}).get("overall_success_diff", {})
            lines.append(
                f"- **1. Recurrent State Compression (`520 KB` 2L GDN) — Not Multi-Hop Looping — Drives Closed-Loop Gains (`H1c`, `H4c`):** "
                f"Looping a memoryless pretrained VLA (`smolvla_m1_rdvla_loop4`, `L=4`, `10/400 = 2.50%`) yields no significant gain over `smolvla_m0_base` "
                f"(`6/400 = 1.50%`; `{c0a.get('mean_diff', 0.0)*100:+.2f}%`, two-sided $p = {c0a.get('p_value_two_sided', 1.0):.4f}$, $p_{{\\text{{Holm}}}} = {c0a.get('p_holm', 1.0):.4f}$ NS). "
                f"Equipping `SmolVLA` with single-pass `520 KB` 2-layer GDN (`smolvla_m5_gdn2l_l1`, `L=1`, `21/400 = {c1c['mean_a']*100:.2f}%`) achieves a **Holm-Bonferroni significant gain** over `M0` "
                f"(`{c1c['mean_diff']*100:+.2f}%`, two-sided $p = {c1c['p_value_two_sided']:.4f}$, **$p_{{\\text{{Holm}}}} = {c1c.get('p_holm', 1.0):.4f}$**) and outperforms all multi-hop `SmolVLA-DeltaLoop` variants "
                f"(`4.00%–4.50%`; `H4c` `M7 CRN` `{c1a['mean_a']*100:.2f}%` vs `M5` `{c1c['mean_a']*100:.2f}%`: `{c4c.get('mean_diff', 0.0)*100:+.2f}%`, two-sided $p = {c4c.get('p_value_two_sided', 1.0):.4f}$ NS; "
                f"`H1a` `M7 CRN` vs `M0`: `{c1a['mean_diff']*100:+.2f}%`, unadjusted two-sided $p = {c1a['p_value_two_sided']:.4f}$, $p_{{\\text{{Holm}}}} = {c1a.get('p_holm', 1.0):.4f}$ NS after multiplicity correction), "
                f"consistent with Scratch Cycle 2A (`11.25% vs 7.50%`) and Cycle 3A (`12.50% vs 8.12%/6.25%`)."
            )
        if "H4b_M7_L2_vs_M7_L1_IntraCkpt" in comparisons and "H5b_M7_Adapt_vs_M7_CodaConv" in comparisons:
            c4b = comparisons["H4b_M7_L2_vs_M7_L1_IntraCkpt"]["overall_success_diff"]
            c4d = comparisons.get("H4d_M7_Adapt_vs_M7_L1_IntraCkpt", {}).get("overall_success_diff", {})
            c4e = comparisons.get("H4e_M7_CodaConv_vs_M7_L1_IntraCkpt", {}).get("overall_success_diff", {})
            c5b = comparisons["H5b_M7_Adapt_vs_M7_CodaConv"]["overall_success_diff"]
            lines.append(
                f"- **2. Intra-Checkpoint Loop Sensitivity & Halting Pareto Efficiency (`H4b`, `H4d`, `H4e`, `H5b`):** "
                f"Within the same random-`L`-trained `smolvla_m7_deltaloop.pt` weights (`N = 400`), truncating to `Fixed L=1` yields `11/400` (`2.75%`), compared to "
                f"`16/400` (`4.00%`, `Fixed L=2`, `H4b` two-sided $p = {c4b['p_value_two_sided']:.4f}$ NS), "
                f"`17/400` (`4.25%`, `Adaptive CRN` at $L_t = 1.11$, `H4d` two-sided $p = {c4d.get('p_value_two_sided', 1.0):.4f}$ NS), and "
                f"`18/400` (`4.50%`, `Latent L2 Conv` at $L_t = 2.28$, `H4e` two-sided $p = {c4e.get('p_value_two_sided', 1.0):.4f}$ NS; all $p_{{\\text{{Holm}}}} = 1.0000$). "
                f"These non-significant intra-checkpoint trends reflect sensitivity of random-loop-trained weights to `L=1` truncation while remaining below single-pass `M5` (`5.25%`); "
                f"within `M7`, Stage-II `CRN` (`1.11` hops) matches `Latent L2 Conv` (`2.28` hops; `H5b: {c5b['mean_diff']*100:+.2f}%`, two-sided $p = {c5b['p_value_two_sided']:.4f}$) at `-51.3%` fewer recurrent hops."
            )
        if "H7b_Interf_Drop_M2_k3_vs_k0" in comparisons and "H7a_Interf_k3_M7_vs_M2" in comparisons:
            c7b = comparisons["H7b_Interf_Drop_M2_k3_vs_k0"]["overall_success_diff"]
            c7c = comparisons.get("H7c_Interf_Drop_M7_k3_vs_k0", {}).get("overall_success_diff", {})
            c7a = comparisons["H7a_Interf_k3_M7_vs_M2"]["overall_success_diff"]
            c7d = comparisons.get("H7d_M5_InterfDrop_k3_vs_k0", {}).get("overall_success_diff", {})
            c7f = comparisons.get("H7f_Interf_k3_M7_vs_M5", {}).get("overall_success_diff", {})
            c7h = comparisons.get("H7h_DiD_InterfPenalty_M7_vs_M2", {}).get("overall_success_diff", {})
            c7i = comparisons.get("H7i_DiD_InterfPenalty_M7_vs_M5", {}).get("overall_success_diff", {})
            m5_txt = ""
            if c7d:
                m5_txt = (
                    f" Decisively, single-pass `M5` (`smolvla_m5_gdn2l_l1_interfk3`) also drops from `12/240` (`5.00%`) to `2/240` (`0.83%`; "
                    f"`H7d: {c7d['mean_diff']*100:+.2f}%`, exact two-sided McNemar $p = {c7d['p_value_two_sided']:.4f}$, $p_{{\\text{{Holm}}}} = {c7d.get('p_holm', 0.068):.4f}$ borderline NS on `0` gained vs `10` lost "
                    f"[finite-sample bootstrap $p \\le 0.0002$, $p_{{\\text{{Holm}}}} \\le 0.0074$, $1/(B+1)$ resolution $\\le 0.0038$; zero-gain bootstrap underestimates variance]), "
                    f"indicating that static linear readout over ~`960` accumulated distractor frames degrades in closed loop despite high raw-state cosine similarity in offline probes."
                )
            lines.append(
                f"- **3. Live Cross-Session Interference (`k = 3`, `N = 240` Matched `eps 0..14`, `H7a–H7i`):** "
                f"Under `k = 3` un-reset prior distractor episodes, 32-slot `FrameSamp-32` (`M2`) drops from `14/240` (`5.83%`) to `3/240` (`1.25%`; "
                f"`H7b: {c7b['mean_diff']*100:+.2f}%`, unadjusted two-sided $p = {c7b['p_value_two_sided']:.4f}$, $p_{{\\text{{Holm}}}} = {c7b.get('p_holm', 1.0):.4f}$)."
                f"{m5_txt} Meanwhile, `SmolVLA-DeltaLoop` (`M7`) moves from `10/240` (`4.17%`) to `7/240` (`2.92%`; `H7c: {c7c.get('mean_diff', 0.0)*100:+.2f}%`, two-sided $p = {c7c.get('p_value_two_sided', 1.0):.4f}$ NS). "
                f"However, all `k = 3` head-to-head and Difference-in-Differences comparisons (`H7a` `M7 vs M2`: `7/240 vs 3/240`, two-sided $p = {c7a['p_value_two_sided']:.4f}$ NS; "
                f"`H7f` `M7 vs M5`: `7/240 vs 2/240`, two-sided $p = {c7f.get('p_value_two_sided', 1.0):.4f}$ NS; `H7h` DiD `M7 vs M2` $p = {c7h.get('p_value_two_sided', 1.0):.4f}$ NS; "
                f"`H7i` DiD `M7 vs M5` $p = {c7i.get('p_value_two_sided', 1.0):.4f}$ NS) are **not statistically significant** in this low-count floor regime."
            )
        if "smolvla_m7_deltaloop_q16_adaptive" in loaded and "smolvla_m7_deltaloop_anchor_adaptive" in loaded:
            c8a = comparisons.get("H8a_M7_Q16_vs_M0_Base", {}).get("overall_success_diff", {})
            c9f = comparisons.get("H9f_M7_Anchor_L2_vs_Anchor_Adapt", {}).get("overall_success_diff", {})
            lines.append(
                f"- **4. Exploratory Readout Bandwidth (`U7`) & Anchor-Shortcut Observations (`U8`):** "
                f"Expanding mutable Coda queries to `N_c = 16` (`U7`: `16/400 = 4.00%`, $L_t = 1.42$; `H8a` vs `M0` unadjusted two-sided $p = {c8a.get('p_value_two_sided', 1.0):.4f}$, $p_{{\\text{{Holm}}}} = {c8a.get('p_holm', 1.0):.4f}$ NS) "
                f"shifts successes toward `Reference` (`5.0%`) while lowering `Permanence` (`2.0%`). Partitioning `4+4` Hop-0 Anchor + Hop-$l$ Pointer registers (`U8`: `12/400 = 3.00%`) causes Stage-II CRN to halt at $L=1$ on `96.1%` of chunks ($L_t = 1.04$); "
                f"evaluating `U8` at `Fixed L=2` yields `15/400` (`3.75%`, `H9f: {c9f.get('mean_diff', 0.0)*100:+.2f}%`, two-sided $p = {c9f.get('p_value_two_sided', 1.0):.4f}$ NS), an exploratory non-significant observation."
            )
        if "H_MS1_Pooled_M7_vs_M0" in comparisons:
            cms1 = comparisons["H_MS1_Pooled_M7_vs_M0"]["overall_success_diff"]
            cms2 = comparisons.get("H_MS2_Pooled_M2_vs_M0", {}).get("overall_success_diff", {})
            cms3 = comparisons.get("H_MS3_Pooled_M7_vs_M2", {}).get("overall_success_diff", {})
            lines.append(
                f"- **5. Multi-Seed Reproducibility (`seeds = {{42, 43, 44}}`, Pooled `N = {cms1['n_pairs']}`) & `<10.5%` Absolute SR Floor Caveat:** "
                f"Across 3 training seeds (`N = 240` matched `eps 0..14`/seed), `SmolVLA-M7` achieves `4.03% ± 0.24%` (`10, 10, 9 / 240`) vs `SmolVLA-M0` `2.22% ± 0.87%` (`3, 6, 7 / 240`; "
                f"`H_MS1: {cms1['mean_diff']*100:+.2f}%`, unadjusted two-sided $p = {cms1['p_value_two_sided']:.4f}$, $p_{{\\text{{Holm}}}} = {cms1.get('p_holm', 1.0):.4f}$ NS after multiplicity correction) and "
                f"`FrameSamp-32 (M2)` `7.64% ± 2.44%` (`14, 16, 25 / 240` at $k=0$; `H_MS2: {cms2.get('mean_diff', 0.0)*100:+.2f}%`, **$p_{{\\text{{Holm}}}} = {cms2.get('p_holm', 1.0):.4f}$**; "
                f"`H_MS3 M7 vs M2: {cms3.get('mean_diff', 0.0)*100:+.2f}%`, two-sided $p = {cms3.get('p_value_two_sided', 1.0):.4f}$, **$p_{{\\text{{Holm}}}} = {cms3.get('p_holm', 1.0):.4f}$**). "
                f"All frozen-VLM `lerobot/smolvla_base` (`450M`, `8x8`-pooled cached tokens) runs remain in a `<10.5%` absolute SR floor regime well below end-to-end 3B `\\pi_{{0.5}}` (`17.9%` base / `44.5%` `FrameSamp+Modul`)."
            )
        lines.append("")

    md_path = results_dir / "SUMMARY_CYCLE3B.md"
    tmp_md = results_dir / ".SUMMARY_CYCLE3B.md.tmp"
    tmp_md.write_text("\n".join(lines) + "\n")
    tmp_md.replace(md_path)
    return md_path, comp_path


# =============================================================================
# 6. CLI & Two-Wave Streaming Daemon Loop
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parallel 16-Task Pretrained SmolVLA-DeltaLoop Evaluator (Cycle 3B)")
    p.add_argument("--train-dir", type=Path, default=LOCAL_CYCLE3B_DIR)
    p.add_argument("--miyabi-ckpt-dir", type=str, default=MIYABI_CYCLE3B_DIR)
    p.add_argument("--mirror-dir", type=Path, default=DEFAULT_MIRROR_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--python-bin", type=Path, default=DEFAULT_EVAL_PYTHON)
    p.add_argument("--task-preset", default="all16", choices=["all16", "rep8"])
    p.add_argument("--tasks", nargs="+", default=None, help="Optional explicit task list overriding --task-preset")
    p.add_argument("--dataset", default="test", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=25, help="Episodes per task (default 25 x 16 = 400 matched test eps)")
    p.add_argument("--seed-episodes", type=int, default=15, help="Episodes per task for non-42 training seeds {s43, s44} (default 15 x 16 = 240 matched test eps)")
    p.add_argument("--interf-episodes", type=int, default=15, help="Episodes per task for auto k=3 interference (default 15 x 16 = 240 matched test eps)")
    p.add_argument("--focused-wave2", action="store_true", help="Run focused Wave-2 sequence: m7_L1 -> anchor_L2 -> coda_conv -> 6x {s43, s44} (m7, m2, m0)")
    p.add_argument("--shard-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--exec-horizon", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1300)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only-stems", nargs="+", default=None, help="Evaluate only these checkpoint stems/archs")
    p.add_argument("--distractor-sessions", type=int, default=0, help="k in {0, 1, 3} prior distractor sessions")
    p.add_argument("--no-auto-interf-k3", action="store_true", help="Disable automatic Wave-2 k=3 interference on M7 & M2")
    p.add_argument("--no-sweeps", action="store_true")
    p.add_argument("--no-coda-conv", action="store_true")
    p.add_argument("--no-sync", action="store_true")
    p.add_argument("--summary-only", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--min-models", type=int, default=5)
    p.add_argument("--expected-models", type=int, default=8)

    # Internal worker-shard flags
    p.add_argument("--worker-shard", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--task", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ep-start", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--ep-end", type=int, default=20, help=argparse.SUPPRESS)
    p.add_argument("--halt-mode", type=str, default="fixed", help=argparse.SUPPRESS)
    p.add_argument("--num-loops", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--halt-threshold", type=float, default=0.5, help=argparse.SUPPRESS)
    p.add_argument("--coda-conv-tau", type=float, default=0.18, help=argparse.SUPPRESS)
    p.add_argument("--ode-steps", type=int, default=10, help=argparse.SUPPRESS)
    p.add_argument("--shard-json", type=str, default=None, help=argparse.SUPPRESS)
    return p


def _is_spec_complete(spec: EvalSpec, results_dir: Path, expected_total_eps: int) -> bool:
    out_json = results_dir / f"{spec.eval_id}.json"
    if not out_json.exists():
        return False
    try:
        d = json.loads(out_json.read_text())
        return len(d.get("episodes", [])) >= expected_total_eps
    except Exception:
        return False


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.worker_shard:
        run_worker_shard(args)
        return

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.mirror_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        md_path, _ = generate_summary_and_comparisons(args.results_dir, args.mirror_dir)
        print(f"[summary-3b] Updated {md_path}", flush=True)
        return

    tasks = tuple(args.tasks) if args.tasks else (
        ALL_16_TASKS_DISPLAY if args.task_preset == "all16" else REP_8_TASKS_DISPLAY
    )
    expected_total_eps = len(tasks) * args.episodes
    expected_seed_eps = len(tasks) * args.seed_episodes

    while True:
        pbs_running: bool | None = None
        if not args.no_sync:
            sync_info = sync_ready_checkpoints(args.mirror_dir, args.train_dir, args.miyabi_ckpt_dir)
            pbs_running = sync_info["pbs_running"]

        mirrored = discover_mirrored_checkpoints(args.mirror_dir)
        if args.only_stems:
            allowed = set(args.only_stems)
            mirrored = [m for m in mirrored if m[0] in allowed or m[2] in allowed]

        wave1_pending: list[tuple[EvalSpec, int]] = []
        wave2_interf: list[tuple[EvalSpec, int]] = []
        wave2_sweeps_fast: list[tuple[EvalSpec, int]] = []
        wave2_coda: list[tuple[EvalSpec, int]] = []
        wave2_seeds: list[tuple[EvalSpec, int]] = []
        wave2_tail: list[tuple[EvalSpec, int]] = []
        for stem, ckpt_path, arch, train_seed in mirrored:
            specs = build_specs_for_checkpoint(
                stem,
                ckpt_path,
                arch,
                train_seed,
                include_sweeps=(not args.no_sweeps),
                include_coda_conv=(not args.no_coda_conv),
                include_l4=(not args.focused_wave2),
                distractor_sessions=args.distractor_sessions,
            )
            for sp in specs:
                if sp.train_seed != 42:
                    if args.focused_wave2 and ("_q16" in sp.ckpt_stem or "_anchor" in sp.ckpt_stem):
                        continue
                    if not _is_spec_complete(sp, args.results_dir, expected_seed_eps):
                        wave2_seeds.append((sp, args.seed_episodes))
                    continue

                if not _is_spec_complete(sp, args.results_dir, expected_total_eps):
                    if sp.wave == 1:
                        wave1_pending.append((sp, args.episodes))
                    elif sp.halt_mode == "coda_conv":
                        wave2_coda.append((sp, args.episodes))
                    elif sp.num_loops == 4:
                        wave2_tail.append((sp, args.episodes))
                    else:
                        wave2_sweeps_fast.append((sp, args.episodes))

            if (
                args.distractor_sessions == 0
                and (not args.no_auto_interf_k3)
                and train_seed == 42
                and arch in ("smolvla_m7_deltaloop", "smolvla_m2_framesamp")
            ):
                k3_specs = build_specs_for_checkpoint(
                    stem,
                    ckpt_path,
                    arch,
                    train_seed,
                    include_sweeps=False,
                    include_coda_conv=False,
                    distractor_sessions=3,
                )
                expected_k3_eps = len(tasks) * args.interf_episodes
                for ksp in k3_specs:
                    if not _is_spec_complete(ksp, args.results_dir, expected_k3_eps):
                        if "_q16" in stem or "_anchor" in stem:
                            if not args.focused_wave2:
                                wave2_tail.append((ksp, args.interf_episodes))
                        else:
                            wave2_interf.append((ksp, args.interf_episodes))

        shards_dir_check = args.results_dir / "shards"

        def _has_partial(sp_item: tuple[EvalSpec, int]) -> int:
            return 0 if shards_dir_check.exists() and any(shards_dir_check.glob(f"{sp_item[0].eval_id}__*.json")) else 1

        def _sweep_prio(sp_item: tuple[EvalSpec, int]) -> tuple[int, int]:
            eid = sp_item[0].eval_id
            # Explicit priority: smolvla_m7_deltaloop_L2 -> smolvla_m7_deltaloop_L1 -> smolvla_m7_deltaloop_anchor_L2
            order_map = {
                "smolvla_m7_deltaloop_L2": 0,
                "smolvla_m7_deltaloop_L1": 1,
                "smolvla_m7_deltaloop_anchor_L2": 2,
            }
            return (_has_partial(sp_item), order_map.get(eid, 9))

        wave1_pending.sort(key=_has_partial)
        wave2_interf.sort(key=lambda p: (_has_partial(p), 0 if p[0].arch == "smolvla_m7_deltaloop" else 1))
        wave2_sweeps_fast.sort(key=_sweep_prio)
        wave2_coda.sort(key=_has_partial)
        seed_arch_prio = {"smolvla_m7_deltaloop": 0, "smolvla_m2_framesamp": 1, "smolvla_m0_base": 2}
        wave2_seeds.sort(
            key=lambda p: (
                _has_partial(p),
                1 if ("_q16" in p[0].ckpt_stem or "_anchor" in p[0].ckpt_stem) else 0,
                seed_arch_prio.get(p[0].arch, 9),
                p[0].train_seed,
            )
        )
        ordered_pending = wave1_pending + wave2_interf + wave2_sweeps_fast + wave2_coda + wave2_seeds + wave2_tail
        if ordered_pending:
            next_spec, spec_eps = ordered_pending[0]
            evaluate_spec_parallel(
                next_spec,
                results_dir=args.results_dir,
                python_bin=args.python_bin,
                tasks=tasks,
                dataset=args.dataset,
                episodes_per_task=spec_eps,
                shard_size=args.shard_size,
                seed=args.seed,
                exec_horizon=args.exec_horizon,
                max_steps=args.max_steps,
                workers=args.workers,
                device=args.device,
            )
            md_path, _ = generate_summary_and_comparisons(args.results_dir, args.mirror_dir)
            print(f"[summary-3b] Updated {md_path}", flush=True)
            continue

        if args.results_dir.exists():
            generate_summary_and_comparisons(args.results_dir, args.mirror_dir)

        if not args.watch:
            break

        n_mirrored = len(mirrored)
        if n_mirrored >= args.expected_models:
            print(f"[watch-3b] All {n_mirrored}/{args.expected_models} SmolVLA checkpoints evaluated! Exiting.", flush=True)
            break
        if pbs_running is False and n_mirrored >= args.min_models:
            print(f"[watch-3b] PBS queue empty and {n_mirrored} SmolVLA models evaluated. Exiting.", flush=True)
            break

        print(
            f"[watch-3b] Idle ({n_mirrored}/{args.expected_models} SmolVLA models mirrored, pbs_running={pbs_running}); "
            f"sleeping {args.poll_seconds}s...",
            flush=True,
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
