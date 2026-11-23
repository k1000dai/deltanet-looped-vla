#!/usr/bin/env python3
"""Parallel Local-GPU RoboMME Evaluator & Atomic Miyabi Streamer for Cycle 2A.

Executes `EXP_TIPS.md`'s Parallel Miyabi-Train + Local-GPU-Eval pipeline:
1. Atomically pulls completed `<arch>.pt` checkpoints from Miyabi (`miyabi-g1`) into
   `checkpoints/robomme/eval_mirror/` ONLY AFTER `<arch>_metrics.json` exists remotely.
2. Evaluates each checkpoint on our local RTX 5080 (`DISPLAY=""`) across all 8 Cycle 2A
   representative RoboMME tasks (`20` `val` episodes/task = `160` matched episodes/model,
   `seed=7`, `exec_horizon=16`, `max_steps=1300`) using bounded parallel worker subprocesses
   (`16` Longest-Processing-Time-First shards of 10 episodes each, reclaiming 100% of
   SAPIEN/Vulkan VRAM after every shard and staying strictly `<= 8 GB` total VRAM).
3. For multi-depth `DeltaLoop` checkpoints (`m6_deltaloop_b2`, `m6b_deltaloop_wb`), evaluates
   BOTH Stage-2 CRN Adaptive Halting (`learned_gate`) AND fixed loop depths (`L=1, 2, 4`)
   (plus training-free `coda_conv` on `m6_deltaloop_b2`), recording full per-chunk loop-depth
   trajectories `L_t` and regenerating `results/robomme/cycle2a/SUMMARY_CYCLE2A.md` +
   `results/robomme/cycle2a/paired_comparisons.json` with 10,000-resample paired bootstrap CIs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _p in (str(SRC_ROOT), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robomme_eval.interface import ROBOMME_SUITES, TASK_TO_SUITE
from robomme_eval.metrics import EpisodeResult, apply_holm_bonferroni, compare_policy_runs, summarize_episodes

DEFAULT_PYTHON = REPO_ROOT / "external" / "robomme_benchmark" / ".venv" / "bin" / "python"
DEFAULT_MIRROR_DIR = REPO_ROOT / "checkpoints" / "robomme" / "eval_mirror"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "robomme" / "cycle2a"
MIYABI_CKPT_DIR = "/work/gj26/b20098/projects/deltanet-looped-vla/checkpoints/robomme/cycle2a"
MIYABI_TRAIN_LOG = "/work/gj26/b20098/projects/deltanet-looped-vla/outputs/cycle2a_train.log"

# Ordered Longest-Processing-Time-First (LPT) for optimal parallel worker scheduling
CYCLE2A_TASKS_LPT: tuple[str, ...] = (
    "VideoPlaceOrder",
    "PickHighlight",
    "PickXtimes",
    "ButtonUnmaskSwap",
    "VideoUnmaskSwap",
    "MoveCube",
    "PatternLock",
    "StopCube",
)

# Canonical table display order grouped by the 4 RoboMME memory suites
CYCLE2A_TASKS_DISPLAY: tuple[str, ...] = (
    "ButtonUnmaskSwap",
    "VideoUnmaskSwap",
    "PickXtimes",
    "StopCube",
    "PickHighlight",
    "VideoPlaceOrder",
    "PatternLock",
    "MoveCube",
)

CYCLE2A_SUITES_ORDER: tuple[str, ...] = (
    "Permanence",
    "Counting",
    "Reference",
    "Imitation",
)

PRIORITY_MODEL_ORDER: tuple[str, ...] = (
    "m0_memoryless",
    "m2_framesamp_modul",
    "m4_gdn_b2_l1",
    "m6_deltaloop_b2",
    "m1_rdvla_loop4",
    "m3_gdn_b1_l1",
    "m5_gdn2layer_b2_l1",
    "m6b_deltaloop_wb",
)


@dataclass(frozen=True)
class EvalSpec:
    """Single evaluation configuration over the 160 matched Cycle 2A episodes."""

    eval_id: str
    arch: str
    label: str
    ckpt_path: Path
    halt_mode: str
    num_loops: int | None
    halt_threshold: float = 0.5
    coda_conv_tau: float = 0.08
    ode_steps: int = 10


def build_eval_specs_for_arch(
    arch: str,
    ckpt_path: Path,
    *,
    include_coda_conv: bool = True,
) -> list[EvalSpec]:
    """Expand a synced `<arch>.pt` checkpoint into its required evaluation run(s)."""
    labels = {
        "m0_memoryless": "M0: Memoryless Flow (L=1)",
        "m1_rdvla_loop4": "M1: RD-VLA Memoryless Loop (L=4)",
        "m2_framesamp_modul": "M2: FrameSamp-32 + Modul/XAttn",
        "m3_gdn_b1_l1": "M3: GDN beta in (0,1) Single-Pass (L=1)",
        "m4_gdn_b2_l1": "M4: GDN beta in (0,2) Householder (L=1)",
        "m5_gdn2layer_b2_l1": "M5: 2-Layer Iso-State GDN beta in (0,2) (L=1)",
    }
    if arch in ("m6_deltaloop_b2", "m6b_deltaloop_wb"):
        prefix = "M6: DeltaLoop-VLA" if arch == "m6_deltaloop_b2" else "M6b: DeltaLoop-VLA + WriteBack"
        specs = [
            EvalSpec(
                eval_id=f"{arch}_adaptive",
                arch=arch,
                label=f"{prefix} (Adaptive CRN Gate)",
                ckpt_path=ckpt_path,
                halt_mode="learned_gate",
                num_loops=4,
            ),
            EvalSpec(
                eval_id=f"{arch}_L4",
                arch=arch,
                label=f"{prefix} (Fixed L=4)",
                ckpt_path=ckpt_path,
                halt_mode="fixed",
                num_loops=4,
            ),
            EvalSpec(
                eval_id=f"{arch}_L2",
                arch=arch,
                label=f"{prefix} (Fixed L=2)",
                ckpt_path=ckpt_path,
                halt_mode="fixed",
                num_loops=2,
            ),
            EvalSpec(
                eval_id=f"{arch}_L1",
                arch=arch,
                label=f"{prefix} (Fixed L=1)",
                ckpt_path=ckpt_path,
                halt_mode="fixed",
                num_loops=1,
            ),
        ]
        if include_coda_conv and arch == "m6_deltaloop_b2":
            specs.append(
                EvalSpec(
                    eval_id=f"{arch}_coda_conv",
                    arch=arch,
                    label=f"{prefix} (Latent L2 Conv tau=0.22)",
                    ckpt_path=ckpt_path,
                    halt_mode="coda_conv",
                    num_loops=4,
                    coda_conv_tau=0.22,
                )
            )
        return specs

    num_loops = 4 if arch == "m1_rdvla_loop4" else 1
    return [
        EvalSpec(
            eval_id=arch,
            arch=arch,
            label=labels.get(arch, arch),
            ckpt_path=ckpt_path,
            halt_mode="fixed",
            num_loops=num_loops,
        )
    ]


# =============================================================================
# 1. Atomic Miyabi Checkpoint Sync (`<arch>_metrics.json` Guard)
# =============================================================================


def _run_miyabi_cmd(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Execute a read-only shell command on `miyabi-g1` via ControlMaster or `ssh-miyabi`."""
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


def verify_checkpoint_compatible(ckpt_path: Path) -> bool:
    """Verify a downloaded `.pt` file is complete and loads strictly into `DeltaLoopVLAPolicy`."""
    import torch
    from models.deltaloop_vla import DeltaLoopVLAConfig, DeltaLoopVLAPolicy

    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not payload.get("completed", False):
            return False
        cfg = DeltaLoopVLAConfig.from_dict(payload["model_config"])
        model = DeltaLoopVLAPolicy(cfg)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return True
    except Exception as exc:
        print(f"[sync] Rejecting incompatible/incomplete checkpoint {ckpt_path}: {exc}", flush=True)
        return False


def sync_ready_checkpoints_from_miyabi(mirror_dir: Path) -> dict[str, Any]:
    """Atomically sync any `<arch>.pt` whose `<arch>_metrics.json` already exists locally or on Miyabi."""
    import shutil

    mirror_dir.mkdir(parents=True, exist_ok=True)
    newly_synced: list[str] = []

    # 1. Fast read-only mirror from local `checkpoints/robomme/cycle2a/` if `vla-trainer` already synced it
    local_train_dir = REPO_ROOT / "checkpoints" / "robomme" / "cycle2a"
    if local_train_dir.exists():
        for mpath in sorted(local_train_dir.glob("*_metrics.json")):
            if mpath.name.startswith(".") or mpath.name.startswith("_"):
                continue
            arch = mpath.name[: -len("_metrics.json")]
            src_ckpt = local_train_dir / f"{arch}.pt"
            dst_ckpt = mirror_dir / f"{arch}.pt"
            dst_metrics = mirror_dir / mpath.name
            if dst_ckpt.exists() and dst_metrics.exists() and dst_ckpt.stat().st_size > 1_000_000:
                continue
            if not src_ckpt.exists() or src_ckpt.stat().st_size < 1_000_000:
                continue
            if verify_checkpoint_compatible(src_ckpt):
                tmp_m = mirror_dir / f".{mpath.name}.tmp"
                tmp_c = mirror_dir / f".{arch}.pt.tmp"
                shutil.copy2(mpath, tmp_m)
                shutil.copy2(src_ckpt, tmp_c)
                tmp_m.replace(dst_metrics)
                tmp_c.replace(dst_ckpt)
                size_mb = dst_ckpt.stat().st_size / (1024**2)
                print(
                    f"[sync] Verified & staged `{dst_ckpt.name}` ({size_mb:.1f} MiB) from local cycle2a/ -> eval_mirror!",
                    flush=True,
                )
                newly_synced.append(arch)

    probe_script = (
        f"ls -1 {MIYABI_CKPT_DIR}/*_metrics.json 2>/dev/null || true; "
        f"echo '__SUMMARY__'; "
        f"ls -1 {MIYABI_CKPT_DIR}/cycle2a_train_summary.json 2>/dev/null || true; "
        f"echo '__DONE_FLAG__'; "
        f"grep -c '\\[ALL_MODELS_COMPLETE\\]' {MIYABI_TRAIN_LOG} 2>/dev/null || true; "
        f"echo '__QSTAT__'; "
        f"/usr/local/bin/qstat 2>/dev/null || true"
    )
    probe = _run_miyabi_cmd(probe_script, timeout=30)
    if probe.returncode != 0:
        print(f"[sync] Warning: Miyabi probe returned {probe.returncode}: {probe.stderr.strip()}", flush=True)
        return {"synced_archs": newly_synced, "all_train_done": False, "pbs_running": True}

    parts = probe.stdout.split("__SUMMARY__")
    metrics_lines = [ln.strip() for ln in parts[0].splitlines() if ln.strip().endswith("_metrics.json")]
    rest1 = parts[1] if len(parts) > 1 else ""
    parts2 = rest1.split("__DONE_FLAG__")
    has_summary = "cycle2a_train_summary.json" in parts2[0]
    rest2 = parts2[1] if len(parts2) > 1 else ""
    parts3 = rest2.split("__QSTAT__")
    done_count_str = parts3[0].strip().splitlines()[0].strip() if parts3[0].strip() else "0"
    qstat_out = parts3[1] if len(parts3) > 1 else ""

    all_train_done = done_count_str.isdigit() and int(done_count_str) > 0
    pbs_running = ("RUNNING" in qstat_out) or ("QUEUED" in qstat_out) or ("TRANSIT" in qstat_out)

    if has_summary:
        subprocess.run(
            [
                "rsync",
                "-az",
                f"miyabi-g1:{MIYABI_CKPT_DIR}/cycle2a_train_summary.json",
                str(mirror_dir / "cycle2a_train_summary.json"),
            ],
            check=False,
            timeout=30,
        )

    for remote_metrics in metrics_lines:
        fname = Path(remote_metrics).name
        arch = fname[: -len("_metrics.json")]
        local_ckpt = mirror_dir / f"{arch}.pt"
        local_metrics = mirror_dir / fname

        if local_ckpt.exists() and local_metrics.exists():
            continue

        print(f"[sync] Detected `{fname}` on Miyabi -> pulling `{arch}.pt` atomically...", flush=True)
        tmp_ckpt = mirror_dir / f".{arch}.pt.tmp"
        tmp_metrics = mirror_dir / f".{fname}.tmp"

        r1 = subprocess.run(
            ["rsync", "-az", f"miyabi-g1:{MIYABI_CKPT_DIR}/{fname}", str(tmp_metrics)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        r2 = subprocess.run(
            ["rsync", "-az", f"miyabi-g1:{MIYABI_CKPT_DIR}/{arch}.pt", str(tmp_ckpt)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if r1.returncode != 0 or r2.returncode != 0 or not tmp_ckpt.exists():
            print(f"[sync] rsync failed for {arch}: {r1.stderr} {r2.stderr}", flush=True)
            tmp_ckpt.unlink(missing_ok=True)
            tmp_metrics.unlink(missing_ok=True)
            continue

        if not verify_checkpoint_compatible(tmp_ckpt):
            tmp_ckpt.unlink(missing_ok=True)
            tmp_metrics.unlink(missing_ok=True)
            continue

        tmp_metrics.replace(local_metrics)
        tmp_ckpt.replace(local_ckpt)
        size_mb = local_ckpt.stat().st_size / (1024**2)
        print(f"[sync] Verified & staged `{local_ckpt.name}` ({size_mb:.1f} MiB) in eval_mirror!", flush=True)
        newly_synced.append(arch)

    return {
        "synced_archs": newly_synced,
        "all_train_done": all_train_done,
        "pbs_running": pbs_running,
    }


def list_local_mirrored_archs(mirror_dir: Path) -> list[str]:
    """Return mirrored architectures in priority evaluation order."""
    if not mirror_dir.exists():
        return []
    present = {
        p.stem
        for p in mirror_dir.glob("*.pt")
        if not p.name.startswith(".") and not p.name.startswith("_")
    }
    ordered = [a for a in PRIORITY_MODEL_ORDER if a in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


# =============================================================================
# 2. Isolated Worker Subprocess Shard Execution
# =============================================================================


def _install_physical_bound_sampler(policy: Any) -> None:
    """Replace hardcoded `x.clamp(-1.5, 1.5)` with physical-delta-aware bounds.

    `norm_stats.json` in `robomme_siglip_v1` normalized `quantile_delta` by 1-step joint deltas
    (`q99 ~ 0.03-0.05 rad`), so legitimate 20-step chunk targets span `[-15, +15]` in normalized
    units while `DeltaLoopVLAPolicy.sample_action_chunk` clamped to `[-1.5, +1.5]` (truncating 90%
    of arm motion). Mapping `2.0x` the official 20-step `MME_VLA_NORM_STATS` physical range into
    `policy.normalizer`'s normalized coordinates preserves 100% of learned 20-step trajectories
    while still bounding unphysical jumps (and keeping gripper dim 7 in `[-1.5, +1.5]`).
    """
    import types
    import torch
    from robomme_eval.interface import MME_VLA_NORM_STATS

    norm = policy.normalizer
    mme_low = np.asarray(MME_VLA_NORM_STATS["actions"]["q01"], dtype=np.float32) * 2.0
    mme_high = np.asarray(MME_VLA_NORM_STATS["actions"]["q99"], dtype=np.float32) * 2.0
    span = norm.action_high - norm.action_low + 1e-6
    norm_min = (mme_low - norm.action_low) / span * 2.0 - 1.0
    norm_max = (mme_high - norm.action_low) / span * 2.0 - 1.0
    norm_min[7] = -1.5
    norm_max[7] = 1.5

    clamp_min_t = torch.tensor(norm_min, device=policy.device, dtype=torch.float32).view(1, 1, 8)
    clamp_max_t = torch.tensor(norm_max, device=policy.device, dtype=torch.float32).view(1, 1, 8)

    @torch.no_grad()
    def _sample_action_chunk_unclamped(
        self_model: Any,
        ctx_tokens: torch.Tensor,
        ctx_mask: torch.Tensor,
        mem_tokens: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
        *,
        ode_steps: int = 10,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        bsz = ctx_tokens.shape[0]
        dev = ctx_tokens.device
        x = torch.randn(
            (bsz, self_model.cfg.action_horizon, self_model.cfg.action_dim),
            generator=generator,
            device=dev,
            dtype=torch.float32,
        )
        dt_step = 1.0 / max(1, int(ode_steps))
        for s_i in range(max(1, int(ode_steps))):
            tau = torch.full((bsz,), s_i * dt_step, device=dev, dtype=torch.float32)
            v = self_model.action_expert(
                x, tau, ctx_tokens, ctx_mask, mem_tokens=mem_tokens, mem_mask=mem_mask
            )
            x = x + dt_step * v
        return torch.max(torch.min(x, clamp_max_t), clamp_min_t)

    policy.model.sample_action_chunk = types.MethodType(_sample_action_chunk_unclamped, policy.model)


def run_worker_shard(args: argparse.Namespace) -> None:
    """Execute one `(task, ep_start..ep_end)` shard inside an isolated SAPIEN/CUDA process."""
    os.environ["DISPLAY"] = ""
    import torch
    from models.robomme_policy_wrapper import RoboMMEPolicyWrapper
    from robomme_eval.runner import evaluate_policy

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    policy = RoboMMEPolicyWrapper(
        args.checkpoint,
        device=args.device,
        dtype=dtype,
        num_loops=args.num_loops,
        halt_mode=args.halt_mode,
        halt_threshold=args.halt_threshold,
        coda_conv_tau=args.coda_conv_tau,
        ode_steps=args.ode_steps,
        seed=args.seed,
    )
    _install_physical_bound_sampler(policy)
    ep_ids = list(range(args.ep_start, args.ep_end))
    res = evaluate_policy(
        policy,
        tasks=(args.task,),
        dataset=args.dataset,
        episode_ids=ep_ids,
        model_seeds=(args.seed,),
        exec_horizon=args.exec_horizon,
        max_steps=args.max_steps,
        output_json=args.shard_json,
        resume=True,
        verbose=False,
    )
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / (1024**2), 1)
        if torch.cuda.is_available()
        else 0.0
    )
    shard_path = Path(args.shard_json)
    payload = json.loads(shard_path.read_text())
    payload["worker_telemetry"] = {
        "eval_id": args.eval_id,
        "arch": policy.model.arch,
        "halt_mode": policy.halt_mode,
        "num_loops": args.num_loops,
        "peak_vram_mb": peak_vram_mb,
        "state_size_bytes": policy.model.state_size_bytes(),
    }
    shard_path.write_text(json.dumps(payload, indent=2))
    ov = res["summary"]["overall"]
    print(
        f"SHARD_OK eval={args.eval_id} task={args.task}[{args.ep_start:02d}..{args.ep_end-1:02d}] "
        f"n={ov['num_episodes']} SR={ov['success_rate']*100:.1f}% "
        f"L_mean={ov['mean_loop_depth']:.2f} ms/chunk={ov['mean_ms_per_chunk']:.1f} "
        f"vram_mb={peak_vram_mb:.1f}",
        flush=True,
    )


# =============================================================================
# 3. Loop-Depth Trajectory Diagnostics & Parallel Spec Orchestrator
# =============================================================================


def compute_loop_trajectory_diagnostics(episodes: Sequence[EpisodeResult]) -> dict[str, Any]:
    """Analyze per-chunk loop-depth `L_t` trajectories across tasks, phases, and outcomes."""
    all_chunks = [ld for ep in episodes for ld in ep.loop_depths]
    if not all_chunks:
        return {}

    arr = np.asarray(all_chunks, dtype=np.float64)
    hist = {
        f"L={k}": round(float(np.mean(np.round(arr) == k)), 4)
        for k in (1, 2, 3, 4)
    }

    early_vals: list[float] = []
    mid_vals: list[float] = []
    late_vals: list[float] = []
    video_chunk0_vals: list[float] = []
    video_later_vals: list[float] = []
    succ_ep_means: list[float] = []
    fail_ep_means: list[float] = []

    by_task_traj: dict[str, dict[str, Any]] = {}
    for task in CYCLE2A_TASKS_DISPLAY:
        t_eps = [ep for ep in episodes if ep.task == task and ep.loop_depths]
        if not t_eps:
            continue
        t_chunks = np.asarray([ld for ep in t_eps for ld in ep.loop_depths], dtype=np.float64)
        c0 = np.asarray([ep.loop_depths[0] for ep in t_eps], dtype=np.float64)
        t_early, t_mid, t_late = [], [], []
        for ep in t_eps:
            lds = ep.loop_depths
            m = len(lds)
            if m == 1:
                t_early.extend(lds)
                t_mid.extend(lds)
                t_late.extend(lds)
            else:
                i1 = max(1, m // 3)
                i2 = max(i1 + 1, (2 * m) // 3) if m >= 3 else m
                t_early.extend(lds[:i1])
                t_mid.extend(lds[i1:i2] if i2 > i1 else lds[:i1])
                t_late.extend(lds[i2:] if m > i2 else lds[-1:])

            if ep.demo_frames > 0:
                video_chunk0_vals.append(lds[0])
                if len(lds) > 1:
                    video_later_vals.extend(lds[1:])

            if ep.success:
                succ_ep_means.append(ep.mean_loop_depth)
            else:
                fail_ep_means.append(ep.mean_loop_depth)

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
            "frac_L1": round(float(np.mean(np.round(t_chunks) == 1)), 4),
            "frac_L2": round(float(np.mean(np.round(t_chunks) == 2)), 4),
            "frac_L3": round(float(np.mean(np.round(t_chunks) == 3)), 4),
            "frac_L4": round(float(np.mean(np.round(t_chunks) == 4)), 4),
        }

    return {
        "total_chunks": int(arr.size),
        "mean_L": round(float(arr.mean()), 4),
        "std_L": round(float(arr.std()), 4),
        "depth_histogram": hist,
        "phase_profile": {
            "early_third_mean_L": round(float(np.mean(early_vals)), 4) if early_vals else 0.0,
            "mid_third_mean_L": round(float(np.mean(mid_vals)), 4) if mid_vals else 0.0,
            "late_third_mean_L": round(float(np.mean(late_vals)), 4) if late_vals else 0.0,
            "video_demo_chunk0_mean_L": round(float(np.mean(video_chunk0_vals)), 4)
            if video_chunk0_vals
            else 0.0,
            "video_demo_subsequent_mean_L": round(float(np.mean(video_later_vals)), 4)
            if video_later_vals
            else 0.0,
        },
        "outcome_profile": {
            "success_episodes_mean_L": round(float(np.mean(succ_ep_means)), 4)
            if succ_ep_means
            else None,
            "failed_episodes_mean_L": round(float(np.mean(fail_ep_means)), 4)
            if fail_ep_means
            else None,
        },
        "per_task": by_task_traj,
    }


def _shard_is_complete(shard_path: Path, expected_eps: int) -> bool:
    if not shard_path.exists():
        return False
    try:
        data = json.loads(shard_path.read_text())
        return len(data.get("episodes", [])) >= expected_eps
    except Exception:
        return False


def evaluate_spec_parallel(
    spec: EvalSpec,
    *,
    results_dir: Path,
    python_bin: Path,
    tasks: Sequence[str] = CYCLE2A_TASKS_LPT,
    dataset: str = "val",
    episodes_per_task: int = 20,
    shard_size: int = 10,
    seed: int = 7,
    exec_horizon: int = 16,
    max_steps: int = 1300,
    workers: int = 3,
    device: str = "cuda",
) -> Path:
    """Evaluate `spec` across all `len(tasks) * episodes_per_task` episodes via parallel shards."""
    results_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = results_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{spec.eval_id}.json"

    expected_total = len(tasks) * episodes_per_task
    if out_json.exists():
        try:
            existing = json.loads(out_json.read_text())
            if len(existing.get("episodes", [])) >= expected_total:
                ov = existing["summary"]["overall"]
                print(
                    f"[eval] Skipping `{spec.eval_id}` (already complete: "
                    f"{expected_total} eps, SR={ov['success_rate']*100:.2f}%)",
                    flush=True,
                )
                return out_json
        except Exception:
            pass

    shard_jobs: list[tuple[str, int, int, Path]] = []
    for task in tasks:
        for ep_s in range(0, episodes_per_task, shard_size):
            ep_e = min(episodes_per_task, ep_s + shard_size)
            s_path = shards_dir / f"{spec.eval_id}__{task}__{ep_s:02d}_{ep_e-1:02d}.json"
            shard_jobs.append((task, ep_s, ep_e, s_path))

    pending = [jb for jb in shard_jobs if not _shard_is_complete(jb[3], jb[2] - jb[1])]
    t0 = time.perf_counter()
    print(
        f"\n================================================================================\n"
        f"[eval] Starting `{spec.eval_id}` ({spec.label})\n"
        f"  ckpt={spec.ckpt_path.name} | halt_mode={spec.halt_mode} | num_loops={spec.num_loops} | "
        f"shards={len(pending)}/{len(shard_jobs)} pending (workers={workers})\n"
        f"================================================================================",
        flush=True,
    )

    active: list[tuple[subprocess.Popen[str], tuple[str, int, int, Path], int, float]] = []
    queue = [(jb, 0) for jb in pending]
    done_count = len(shard_jobs) - len(pending)
    max_worker_vram_mb = 0.0

    env = os.environ.copy()
    env["DISPLAY"] = ""
    env["PYTHONUNBUFFERED"] = "1"
    py_path_entries = [str(SRC_ROOT), str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        py_path_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_path_entries)

    while queue or active:
        while queue and len(active) < workers:
            jb, attempt = queue.pop(0)
            task, ep_s, ep_e, s_path = jb
            cmd = [
                str(python_bin),
                str(Path(__file__).resolve()),
                "--worker-shard",
                "--checkpoint",
                str(spec.ckpt_path),
                "--eval-id",
                spec.eval_id,
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
                "--device",
                device,
                "--shard-json",
                str(s_path),
            ]
            if spec.num_loops is not None:
                cmd.extend(["--num-loops", str(spec.num_loops)])

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            active.append((proc, jb, attempt, time.perf_counter()))

        time.sleep(0.5)
        still_active = []
        for proc, jb, attempt, t_launch in active:
            ret = proc.poll()
            if ret is None:
                still_active.append((proc, jb, attempt, t_launch))
                continue

            out_text = proc.stdout.read() if proc.stdout is not None else ""
            task, ep_s, ep_e, s_path = jb
            dt_s = time.perf_counter() - t_launch

            if ret == 0 and _shard_is_complete(s_path, ep_e - ep_s):
                done_count += 1
                s_data = json.loads(s_path.read_text())
                vram_mb = float(s_data.get("worker_telemetry", {}).get("peak_vram_mb", 0.0))
                max_worker_vram_mb = max(max_worker_vram_mb, vram_mb)
                s_ov = s_data["summary"]["overall"]
                print(
                    f"  [{done_count:2d}/{len(shard_jobs):2d}] {task:<18s} "
                    f"ep={ep_s:02d}..{ep_e-1:02d} SR={s_ov['success_rate']*100:5.1f}% "
                    f"L={s_ov['mean_loop_depth']:.2f} ms/ch={s_ov['mean_ms_per_chunk']:5.1f} "
                    f"vram={vram_mb:5.0f}MB wall={dt_s:5.1f}s",
                    flush=True,
                )
            elif attempt < 2:
                print(
                    f"  [retry {attempt+1}/2] {task}[{ep_s:02d}..{ep_e-1:02d}] exited code={ret}; retrying...\n"
                    f"  tail: {out_text[-400:].strip()}",
                    flush=True,
                )
                queue.insert(0, (jb, attempt + 1))
            else:
                raise RuntimeError(
                    f"Shard {spec.eval_id} {task}[{ep_s}..{ep_e-1}] failed after 3 attempts (code={ret}):\n{out_text}"
                )
        active = still_active

    # Merge all shard episodes in canonical order
    task_rank = {t: idx for idx, t in enumerate(CYCLE2A_TASKS_DISPLAY)}
    merged_eps: list[EpisodeResult] = []
    seen_keys: set[tuple[str, str, int, int]] = set()
    state_size_bytes = 0

    for _, _, _, s_path in shard_jobs:
        s_data = json.loads(s_path.read_text())
        state_size_bytes = int(s_data.get("worker_telemetry", {}).get("state_size_bytes", state_size_bytes))
        max_worker_vram_mb = max(
            max_worker_vram_mb,
            float(s_data.get("worker_telemetry", {}).get("peak_vram_mb", 0.0)),
        )
        for raw_ep in s_data.get("episodes", []):
            ep = EpisodeResult(**raw_ep)
            key = (ep.task, ep.dataset, ep.episode_id, ep.model_seed)
            if key not in seen_keys:
                seen_keys.add(key)
                merged_eps.append(ep)

    merged_eps.sort(key=lambda e: (task_rank.get(e.task, 999), e.episode_id, e.model_seed))
    summary = summarize_episodes(merged_eps, n_boot=10000, seed=0)
    loop_diag = compute_loop_trajectory_diagnostics(merged_eps)
    wall_total_s = time.perf_counter() - t0

    final_payload = {
        "eval_spec": {
            "eval_id": spec.eval_id,
            "arch": spec.arch,
            "label": spec.label,
            "checkpoint": str(spec.ckpt_path),
            "halt_mode": spec.halt_mode,
            "num_loops": spec.num_loops,
            "halt_threshold": spec.halt_threshold,
            "coda_conv_tau": spec.coda_conv_tau,
            "ode_steps": spec.ode_steps,
            "state_size_bytes": state_size_bytes,
            "max_worker_vram_mb": round(max_worker_vram_mb, 1),
            "workers": workers,
            "eval_wall_s": round(wall_total_s, 2),
        },
        "config": {
            "tasks": list(CYCLE2A_TASKS_DISPLAY),
            "dataset": dataset,
            "episodes_per_task": episodes_per_task,
            "model_seeds": [seed],
            "exec_horizon": exec_horizon,
            "max_steps": max_steps,
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
        f"[DONE] `{spec.eval_id}` ({len(merged_eps)} eps in {wall_total_s/60:.2f}m): "
        f"Overall SR = {ov['success_rate']*100:.2f}% "
        f"[{ov['success_ci95_low']*100:.2f}%, {ov['success_ci95_high']*100:.2f}%] | "
        f"L_mean = {ov['mean_loop_depth']:.2f} | max_worker_VRAM = {max_worker_vram_mb:.0f} MB",
        flush=True,
    )
    return out_json


# =============================================================================
# 4. Paired Bootstrap Comparisons & `SUMMARY_CYCLE2A.md` Generator
# =============================================================================


def _fmt_pct(val: float) -> str:
    return f"{val * 100.0:.1f}%"


def _fmt_sr_ci(summary_block: dict[str, Any]) -> str:
    sr = summary_block["success_rate"] * 100.0
    lo = summary_block["success_ci95_low"] * 100.0
    hi = summary_block["success_ci95_high"] * 100.0
    return f"**{sr:.1f}%** `[{lo:.1f}, {hi:.1f}]`"


def _fmt_state_bytes(n_bytes: int) -> str:
    if n_bytes <= 0:
        return "`0 B`"
    return f"`{n_bytes / 1024.0:.1f} KB`"


def generate_summary_and_comparisons(
    results_dir: Path,
    mirror_dir: Path,
) -> tuple[Path, Path]:
    """Build `paired_comparisons.json` and `SUMMARY_CYCLE2A.md` from all completed `<eval_id>.json` runs."""
    canonical_order = [
        "m0_memoryless",
        "m1_rdvla_loop4",
        "m2_framesamp_modul",
        "m3_gdn_b1_l1",
        "m4_gdn_b2_l1",
        "m5_gdn2layer_b2_l1",
        "m6_deltaloop_b2_L1",
        "m6_deltaloop_b2_L2",
        "m6_deltaloop_b2_L4",
        "m6_deltaloop_b2_adaptive",
        "m6_deltaloop_b2_coda_conv",
        "m6b_deltaloop_wb_L1",
        "m6b_deltaloop_wb_L2",
        "m6b_deltaloop_wb_L4",
        "m6b_deltaloop_wb_adaptive",
    ]

    loaded: dict[str, dict[str, Any]] = {}
    episodes_map: dict[str, list[EpisodeResult]] = {}
    for jpath in sorted(results_dir.glob("*.json")):
        if jpath.name in ("paired_comparisons.json",) or jpath.name.startswith("."):
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

    ordered_ids = [eid for eid in canonical_order if eid in loaded] + sorted(
        set(loaded.keys()) - set(canonical_order)
    )

    # Load training metrics per arch if mirrored
    train_meta: dict[str, dict[str, Any]] = {}
    for mfile in mirror_dir.glob("*_metrics.json"):
        try:
            mdata = json.loads(mfile.read_text())
            train_meta[mdata["arch"]] = mdata
        except Exception:
            pass

    # Predefined controlled scientific hypothesis pairs `(name, run_A, run_B, question)`
    candidate_pairs = [
        (
            "H0a_M1_RDVLA_Loop4_vs_M0_Memoryless",
            "m1_rdvla_loop4",
            "m0_memoryless",
            "Memoryless L=4 latent looping (RD-VLA control) vs L=1 Markov baseline (0 B state)",
        ),
        (
            "H0b_M2_FrameSamp_vs_M0_Memoryless",
            "m2_framesamp_modul",
            "m0_memoryless",
            "32-slot Perceptual Frame-Sampling buffer vs Markov Flow-Matching baseline",
        ),
        (
            "H0c_M4_GDN_L1_vs_M0_Memoryless",
            "m4_gdn_b2_l1",
            "m0_memoryless",
            "Single-pass Householder GDN beta in (0,2) vs Markov Flow-Matching baseline",
        ),
        (
            "H1a_M6_Adapt_vs_M0_Memoryless",
            "m6_deltaloop_b2_adaptive",
            "m0_memoryless",
            "Overall value of DeltaLoop-VLA over Markov Flow-Matching baseline",
        ),
        (
            "H1b_M6_Adapt_vs_M1_RDVLA_Loop4",
            "m6_deltaloop_b2_adaptive",
            "m1_rdvla_loop4",
            "Persistent GDN state + dynamic readout vs memoryless L=4 looping (RD-VLA control)",
        ),
        (
            "H2_M4_NegEig_b2_vs_M3_PosEig_b1",
            "m4_gdn_b2_l1",
            "m3_gdn_b1_l1",
            "Householder negative eigenvalues beta in (0,2) vs standard beta in (0,1) at matched L=1",
        ),
        (
            "H3a_M6_Adapt_vs_M2_FrameSamp",
            "m6_deltaloop_b2_adaptive",
            "m2_framesamp_modul",
            "Constant-size O(1) GDN + dynamic loop vs 32-slot Perceptual Frame-Sampling buffer",
        ),
        (
            "H3b_M4_GDN_L1_vs_M2_FrameSamp",
            "m4_gdn_b2_l1",
            "m2_framesamp_modul",
            "Single-pass Householder GDN (L=1) vs 32-slot Perceptual Frame-Sampling buffer",
        ),
        (
            "H4a_M6_L4_vs_M6_L1_IntraCkpt",
            "m6_deltaloop_b2_L4",
            "m6_deltaloop_b2_L1",
            "Intra-checkpoint multi-hop dynamic readout gain (L=4 vs L=1 on identical weights)",
        ),
        (
            "H4b_M6_L2_vs_M6_L1_IntraCkpt",
            "m6_deltaloop_b2_L2",
            "m6_deltaloop_b2_L1",
            "Intra-checkpoint 2-hop vs 1-hop dynamic readout gain (L=2 vs L=1)",
        ),
        (
            "H4c_M6_L4_vs_M4_SinglePass_GDN",
            "m6_deltaloop_b2_L4",
            "m4_gdn_b2_l1",
            "Weight-tied L=4 dynamic re-querying vs dedicated single-pass L=1 Neg-Eig GDN",
        ),
        (
            "H4d_M6_Adapt_vs_M4_SinglePass_GDN",
            "m6_deltaloop_b2_adaptive",
            "m4_gdn_b2_l1",
            "Adaptive DeltaLoop-VLA vs dedicated single-pass L=1 Neg-Eig GDN",
        ),
        (
            "H4e_M6_L4_vs_M5_2Layer_IsoState",
            "m6_deltaloop_b2_L4",
            "m5_gdn2layer_b2_l1",
            "Weight-tied L=4 read-looping vs 2-Layer stacked Iso-State GDN (L=1)",
        ),
        (
            "H4f_M6_Adapt_vs_M5_2Layer_IsoState",
            "m6_deltaloop_b2_adaptive",
            "m5_gdn2layer_b2_l1",
            "Adaptive DeltaLoop-VLA vs 2-Layer stacked Iso-State GDN (L=1)",
        ),
        (
            "H5a_M6_Adapt_vs_M6_L4",
            "m6_deltaloop_b2_adaptive",
            "m6_deltaloop_b2_L4",
            "Stage-2 CRN Adaptive Halting compute-accuracy parity vs fixed L=4",
        ),
        (
            "H5b_M6_Adapt_vs_M6_CodaConv",
            "m6_deltaloop_b2_adaptive",
            "m6_deltaloop_b2_coda_conv",
            "Stage-2 CRN Marginal-Gain Gate vs training-free Latent L2 Convergence (Reviewer C2)",
        ),
        (
            "H6a_M6b_WB_Adapt_vs_M6_Adapt",
            "m6b_deltaloop_wb_adaptive",
            "m6_deltaloop_b2_adaptive",
            "Chunk-boundary Coda write-back (c_wb, g=0) ablation (Adaptive)",
        ),
        (
            "H6b_M6b_WB_L4_vs_M6_L4",
            "m6b_deltaloop_wb_L4",
            "m6_deltaloop_b2_L4",
            "Chunk-boundary Coda write-back (c_wb, g=0) ablation (Fixed L=4)",
        ),
    ]

    comparisons: dict[str, Any] = {}
    for key_name, id_a, id_b, desc in candidate_pairs:
        if id_a in episodes_map and id_b in episodes_map:
            try:
                comp = compare_policy_runs(episodes_map[id_a], episodes_map[id_b], n_boot=10000, seed=0)
                comp["run_a"] = id_a
                comp["run_b"] = id_b
                comp["description"] = desc
                comparisons[key_name] = comp
            except ValueError:
                pass

    apply_holm_bonferroni(comparisons)
    comp_path = results_dir / "paired_comparisons.json"
    comp_path.write_text(json.dumps(comparisons, indent=2))

    # Render Markdown Summary
    lines: list[str] = []
    lines.append("# Cycle 2A — Controlled Multi-Suite RoboMME Evaluation Summary")
    lines.append("")
    lines.append(
        f"- **Protocol:** 8 Representative RoboMME Tasks (`2 tasks × 4 memory suites`), "
        f"`val` split (`20` episodes/task = **`160` matched episodes per evaluation run**, `model_seed=7`, "
        f"`exec_horizon=16`, `max_steps=1300`, 10-step Flow-Matching Euler ODE)."
    )
    lines.append(
        f"- **Evaluated Runs Completed:** `{len(ordered_ids)}` (`results/robomme/cycle2a/*.json`)"
    )
    lines.append("")

    # Table 1: Overall & Suite-Level Table
    lines.append("## 1. Overall & Per-Suite Success Rates (`N = 160` Matched Episodes)")
    lines.append("")
    lines.append(
        "| Evaluation Run | Params | State Size | Train FM Loss | Mean $L_t$ (`p95`) | Latency (`ms/ch`) | "
        "Overall SR (`95% CI`) | Permanence (`40`) | Counting (`40`) | Reference (`40`) | Imitation (`40`) |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )

    for eid in ordered_ids:
        d = loaded[eid]
        sp = d["eval_spec"]
        arch = sp["arch"]
        ov = d["summary"]["overall"]
        suites = d["summary"]["suites"]
        tm = train_meta.get(arch, {})
        tot_p = tm.get("parameter_breakdown", {}).get("total")
        p_str = f"`{tot_p / 1e6:.2f}M`" if tot_p else "—"
        fm_val = tm.get("final_log", {}).get("fm_loss")
        fm_str = f"`{fm_val:.4f}`" if isinstance(fm_val, (int, float)) else "—"
        st_bytes = int(sp.get("state_size_bytes") or tm.get("state_size_bytes") or 0)

        s_cols = [
            _fmt_pct(suites[s]["success_rate"]) if s in suites else "—"
            for s in CYCLE2A_SUITES_ORDER
        ]
        lines.append(
            f"| **{sp['label']}** (`{eid}`) | {p_str} | {_fmt_state_bytes(st_bytes)} | {fm_str} | "
            f"`{ov['mean_loop_depth']:.2f}` (`{ov['p95_loop_depth']:.1f}`) | `{ov['mean_ms_per_chunk']:.1f}` | "
            f"{_fmt_sr_ci(ov)} | {s_cols[0]} | {s_cols[1]} | {s_cols[2]} | {s_cols[3]} |"
        )

    lines.append("")
    lines.append("## 2. Per-Task Success Rates (`20` Episodes / Task)")
    lines.append("")
    t_headers = " | ".join(f"`{t}`" for t in CYCLE2A_TASKS_DISPLAY)
    lines.append(f"| Evaluation Run | Overall | {t_headers} |")
    lines.append("| :--- | :---: | " + " | ".join(":---:" for _ in CYCLE2A_TASKS_DISPLAY) + " |")

    for eid in ordered_ids:
        d = loaded[eid]
        ov_sr = _fmt_pct(d["summary"]["overall"]["success_rate"])
        tasks_s = d["summary"]["tasks"]
        t_cells = " | ".join(
            _fmt_pct(tasks_s[t]["success_rate"]) if t in tasks_s else "—"
            for t in CYCLE2A_TASKS_DISPLAY
        )
        lines.append(f"| `{eid}` | **{ov_sr}** | {t_cells} |")

    # Section 3: Paired Bootstrap Hypothesis Tests
    if comparisons:
        lines.append("")
        lines.append("## 3. Paired Bootstrap Hypothesis Tests (`10,000` Resamples on `N = 160` Matched Episodes)")
        lines.append("")
        lines.append(
            "| Hypothesis / Comparison (`A - B`) | $\\text{SR}_A$ | $\\text{SR}_B$ | "
            "$\\Delta \\text{SR}$ (`95% Paired CI`) | Two-Sided $p$ | Holm $p_{\\text{Holm}}$ | Permanence $\\Delta$ | Counting $\\Delta$ | Reference $\\Delta$ | Imitation $\\Delta$ |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for key_name, comp in comparisons.items():
            ov_d = comp["overall_success_diff"]
            diff_pct = ov_d["mean_diff"] * 100.0
            lo_pct = ov_d["ci_low"] * 100.0
            hi_pct = ov_d["ci_high"] * 100.0
            pval = ov_d["p_value_two_sided"]
            pholm = ov_d.get("p_holm", 1.0)
            suite_diffs = []
            for s in CYCLE2A_SUITES_ORDER:
                if s in comp.get("suites", {}):
                    sd = comp["suites"][s]["mean_diff"] * 100.0
                    suite_diffs.append(f"`{sd:+.1f}%`")
                else:
                    suite_diffs.append("—")
            lines.append(
                f"| **{key_name}**: `{comp['run_a']}` vs `{comp['run_b']}` | "
                f"`{ov_d['mean_a']*100:.1f}%` | `{ov_d['mean_b']*100:.1f}%` | "
                f"**`{diff_pct:+.2f}%`** `[{lo_pct:+.2f}%, {hi_pct:+.2f}%]` | "
                f"`{pval:.4f}` | `{pholm:.4f}` | {suite_diffs[0]} | {suite_diffs[1]} | {suite_diffs[2]} | {suite_diffs[3]} |"
            )

    # Section 4: Adaptive Halting & Loop Trajectory Diagnostics
    adaptive_ids = [
        eid
        for eid in ordered_ids
        if loaded[eid]["eval_spec"]["halt_mode"] in ("learned_gate", "coda_conv")
    ]
    if adaptive_ids:
        lines.append("")
        lines.append("## 4. Adaptive Loop-Depth ($L_t$) Trajectory Diagnostics")
        lines.append("")
        for eid in adaptive_ids:
            ld = loaded[eid].get("loop_diagnostics", {})
            if not ld:
                continue
            hist = ld.get("depth_histogram", {})
            phase = ld.get("phase_profile", {})
            lines.append(f"### `{eid}` ({loaded[eid]['eval_spec']['label']})")
            lines.append(
                f"- **Overall Chunk Loop Allocation (`{ld.get('total_chunks', 0)}` chunks):** "
                f"Mean $L_t = {ld.get('mean_L', 0.0):.3f} \\pm {ld.get('std_L', 0.0):.3f}$ | "
                f"Histogram: `L=1: {hist.get('L=1', 0)*100:.1f}%`, `L=2: {hist.get('L=2', 0)*100:.1f}%`, "
                f"`L=3: {hist.get('L=3', 0)*100:.1f}%`, `L=4: {hist.get('L=4', 0)*100:.1f}%`"
            )
            lines.append(
                f"- **Trajectory Phase Profile:** Early `[0..33%]` = `{phase.get('early_third_mean_L', 0):.3f}`, "
                f"Mid `[33..67%]` = `{phase.get('mid_third_mean_L', 0):.3f}`, "
                f"Late `[67..100%]` = `{phase.get('late_third_mean_L', 0):.3f}` | "
                f"Video-Demo Tasks: `Chunk 0 (post-demo)` = `{phase.get('video_demo_chunk0_mean_L', 0):.3f}` vs "
                f"`Subsequent` = `{phase.get('video_demo_subsequent_mean_L', 0):.3f}`"
            )
            lines.append("")
            lines.append(
                "| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |"
            )
            lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for t in CYCLE2A_TASKS_DISPLAY:
                pt = ld.get("per_task", {}).get(t)
                if not pt:
                    continue
                t_sr = loaded[eid]["summary"]["tasks"].get(t, {}).get("success_rate", 0.0)
                lines.append(
                    f"| `{t}` | {TASK_TO_SUITE.get(t, '')} | `{t_sr*100:.1f}%` | "
                    f"`{pt['mean_L']:.2f}` | `{pt['chunk0_mean_L']:.2f}` | `{pt['early_mean_L']:.2f}` | "
                    f"`{pt['mid_mean_L']:.2f}` | `{pt['late_mean_L']:.2f}` | "
                    f"`{pt['frac_L1']*100:.1f}%` | `{pt['frac_L2']*100:.1f}%` | "
                    f"`{pt['frac_L3']*100:.1f}%` | `{pt['frac_L4']*100:.1f}%` |"
                )
            lines.append("")

    md_path = results_dir / "SUMMARY_CYCLE2A.md"
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, comp_path


# =============================================================================
# 5. CLI & Streaming Watch Loop
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Parallel Local-GPU Evaluator & Streamer for Cycle 2A")
    p.add_argument("--mirror-dir", type=Path, default=DEFAULT_MIRROR_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--dataset", default="val", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=20, help="Episodes per task (default: 20 -> 160/model)")
    p.add_argument("--shard-size", type=int, default=10, help="Episodes per isolated worker process")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--exec-horizon", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=1300)
    p.add_argument("--workers", type=int, default=3, help="Concurrent GPU rollout subprocesses (keep VRAM <= 8 GB)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--only-archs", nargs="+", default=None)
    p.add_argument("--no-sync", action="store_true", help="Do not query/rsync from Miyabi; evaluate local mirror only")
    p.add_argument("--no-coda-conv", action="store_true", help="Skip m6 coda_conv baseline")
    p.add_argument("--watch", action="store_true", help="Stream & evaluate continuously until Miyabi Cycle 2A finishes")
    p.add_argument("--poll-seconds", type=int, default=25)
    p.add_argument("--min-models", type=int, default=7, help="Minimum models before --watch can exit on training completion")
    p.add_argument("--expected-models", type=int, default=8, help="Target model count for immediate exit once all evaluated")

    # Internal worker-shard flags
    p.add_argument("--worker-shard", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--checkpoint", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--eval-id", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--task", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--ep-start", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--ep-end", type=int, default=10, help=argparse.SUPPRESS)
    p.add_argument("--num-loops", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--halt-mode", type=str, default="fixed", help=argparse.SUPPRESS)
    p.add_argument("--halt-threshold", type=float, default=0.5, help=argparse.SUPPRESS)
    p.add_argument("--coda-conv-tau", type=float, default=0.08, help=argparse.SUPPRESS)
    p.add_argument("--ode-steps", type=int, default=10, help=argparse.SUPPRESS)
    p.add_argument("--shard-json", type=str, default=None, help=argparse.SUPPRESS)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.worker_shard:
        run_worker_shard(args)
        return

    args.mirror_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    while True:
        sync_info = {"synced_archs": [], "all_train_done": False, "pbs_running": False}
        if not args.no_sync:
            sync_info = sync_ready_checkpoints_from_miyabi(args.mirror_dir)

        mirrored_archs = list_local_mirrored_archs(args.mirror_dir)
        if args.only_archs is not None:
            allowed = set(args.only_archs)
            mirrored_archs = [a for a in mirrored_archs if a in allowed]

        ran_any = False
        re_sort_needed = False
        for arch in mirrored_archs:
            ckpt_path = args.mirror_dir / f"{arch}.pt"
            specs = build_eval_specs_for_arch(
                arch,
                ckpt_path,
                include_coda_conv=(not args.no_coda_conv),
            )
            for spec in specs:
                out_json = args.results_dir / f"{spec.eval_id}.json"
                already_done = False
                if out_json.exists():
                    try:
                        d = json.loads(out_json.read_text())
                        already_done = len(d.get("episodes", [])) >= (len(CYCLE2A_TASKS_LPT) * args.episodes)
                    except Exception:
                        already_done = False
                if already_done:
                    continue

                evaluate_spec_parallel(
                    spec,
                    results_dir=args.results_dir,
                    python_bin=args.python_bin,
                    tasks=CYCLE2A_TASKS_LPT,
                    dataset=args.dataset,
                    episodes_per_task=args.episodes,
                    shard_size=args.shard_size,
                    seed=args.seed,
                    exec_horizon=args.exec_horizon,
                    max_steps=args.max_steps,
                    workers=args.workers,
                    device=args.device,
                )
                md_path, _ = generate_summary_and_comparisons(args.results_dir, args.mirror_dir)
                print(f"[summary] Updated {md_path}", flush=True)
                ran_any = True

                # Re-sync after every completed spec so newly arrived priority models are spotted immediately!
                if not args.no_sync:
                    post_sync = sync_ready_checkpoints_from_miyabi(args.mirror_dir)
                    if post_sync["synced_archs"]:
                        re_sort_needed = True
                        break
            if re_sort_needed:
                break

        # Always refresh summary if any results exist
        if list(args.results_dir.glob("*.json")):
            generate_summary_and_comparisons(args.results_dir, args.mirror_dir)

        if not args.watch:
            break

        mirrored_all = list_local_mirrored_archs(args.mirror_dir)
        all_specs_done = True
        for arch in mirrored_all:
            for sp in build_eval_specs_for_arch(
                arch,
                args.mirror_dir / f"{arch}.pt",
                include_coda_conv=(not args.no_coda_conv),
            ):
                oj = args.results_dir / f"{sp.eval_id}.json"
                if not oj.exists():
                    all_specs_done = False
                    break

        if all_specs_done and len(mirrored_all) >= args.expected_models:
            print(
                f"\n[COMPLETE] All {len(mirrored_all)} expected models synced and 100% evaluated! "
                f"Final report: {args.results_dir / 'SUMMARY_CYCLE2A.md'}",
                flush=True,
            )
            break

        if (
            all_specs_done
            and len(mirrored_all) >= args.min_models
            and (sync_info["all_train_done"] or not sync_info["pbs_running"])
        ):
            print(
                f"\n[COMPLETE] Miyabi training finished and all {len(mirrored_all)} models are 100% evaluated! "
                f"Final report: {args.results_dir / 'SUMMARY_CYCLE2A.md'}",
                flush=True,
            )
            break

        if not ran_any:
            print(
                f"[watch] Mirrored={len(mirrored_all)}/{args.expected_models} ({mirrored_all}), "
                f"pbs_running={sync_info['pbs_running']}; sleeping {args.poll_seconds}s...",
                flush=True,
            )
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
