#!/usr/bin/env python3
"""Closed-loop headless RoboMME evaluator for PyTorch `π0.5`, `FrameSamp+Modul`, and `DeltaMem-π0.5`.

Runs on the local RTX 5080 (`external/robomme_benchmark/.venv`, `DISPLAY=""`) with crash-safe
JSON checkpointing under `results/phase2/`, per-suite/per-task bootstrap CIs, latency & peak
VRAM telemetry, optional `Mem-CFG` (`--mem-cfg-scale`), and live `k`-session un-reset
cross-episode interference testing (`--interference-k`).
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import zlib

os.environ.setdefault("DISPLAY", "")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pi05 import Pi05RoboMMEPolicy  # noqa: E402
from pi05.policy import resolve_tokenizer_path  # noqa: E402
from pi05.model import load_pi05_checkpoint  # noqa: E402
from robomme_eval.interface import PolicyChunkOutput, ROBOMME_SUITES, ROBOMME_TASKS, StatefulPolicy  # noqa: E402
from robomme_eval.metrics import EpisodeResult, summarize_episodes  # noqa: E402
from robomme_eval.runner import evaluate_episode, evaluate_policy  # noqa: E402


class ZeroOraclePromptRoutedPolicy(StatefulPolicy):
    """Zero-oracle `P2-M10` specialist router over public user `task_goal` (`0` privileged inputs!).

    Routes the 6 `<row_y, col_x>` reveal/highlight/repick tasks (`should_route_to_symbolic_specialist`)
    to `specialist_policy` (`symbolic_grounded_subgoal_pt + P2-M9-Sym`) and the 10 continuous/counting/
    order/imitation tasks to `primary_policy` (`v3_s250` or `P2-M7B s300 ordinal_only`), sharing the
    bit-identical frozen `SigLIP` vision tower (`< 12.6 GiB` VRAM) with full `k=3` un-reset stream parity.
    """

    def __init__(
        self,
        primary_policy: Pi05RoboMMEPolicy,
        specialist_policy: Pi05RoboMMEPolicy,
        route_mode: str = "default",
        *,
        offload_inactive: bool = False,
    ) -> None:
        from pi05_mem.deltaground import (
            should_route_to_symbolic_specialist,
            should_route_to_symbolic_specialist_8task,
        )

        self.primary = primary_policy
        self.specialist = specialist_policy
        self._target_device = primary_policy.device
        self._offload_inactive = bool(offload_inactive) and (self._target_device.type == "cuda")
        self._active_name: str = "primary"
        if route_mode == "highlight":
            # Public-prompt-only: route ONLY goals mentioning "highlight" (PickHighlight) to the specialist.
            self._route_fn = lambda g: bool(g) and ("highlight" in str(g).lower())
        elif route_mode == "8task":
            self._route_fn = should_route_to_symbolic_specialist_8task
        else:
            self._route_fn = should_route_to_symbolic_specialist
        self._suppress_reset = False
        if self._offload_inactive:
            self._move_subpolicy(self.specialist, torch.device("cpu"))
            self._move_subpolicy(self.primary, self._target_device)

    def _move_subpolicy(self, pol: Pi05RoboMMEPolicy, dev: torch.device) -> None:
        vt = self.primary.model.vision_tower
        pol.device = dev
        pol.model.to(dev)
        # Keep shared SigLIP vision_tower permanently on target GPU!
        vt.to(self._target_device)
        pol._generator = torch.Generator(device=dev)
        pol._generator.manual_seed(pol._seed)
        ad = getattr(pol, "memory_adapter", None)
        if ad is not None and hasattr(ad, "to_device"):
            ad.to_device(dev)

    @property
    def suppress_reset(self) -> bool:
        return self._suppress_reset

    @suppress_reset.setter
    def suppress_reset(self, val: bool) -> None:
        self._suppress_reset = bool(val)
        self.primary.suppress_reset = bool(val)
        self.specialist.suppress_reset = bool(val)

    def _select(self, task_goal: str) -> tuple[Pi05RoboMMEPolicy, Pi05RoboMMEPolicy]:
        want_spec = bool(self._route_fn(task_goal))
        want_name = "specialist" if want_spec else "primary"
        if self._offload_inactive and (not self._suppress_reset) and want_name != self._active_name:
            old_p = self.specialist if self._active_name == "specialist" else self.primary
            new_p = self.specialist if want_spec else self.primary
            self._move_subpolicy(old_p, torch.device("cpu"))
            torch.cuda.empty_cache()
            self._move_subpolicy(new_p, self._target_device)
            self._active_name = want_name
        if want_spec:
            return self.specialist, self.primary
        return self.primary, self.specialist

    def set_seed(self, seed: int) -> None:
        self.primary.set_seed(seed)
        self.specialist.set_seed(seed)

    def force_reset(self) -> None:
        self.primary.force_reset()
        self.specialist.force_reset()

    def reset(self) -> None:
        if self._suppress_reset:
            return
        self.force_reset()

    def ingest_demo_frames(
        self,
        front_rgb_list: Any,
        wrist_rgb_list: Any,
        state_list: Any,
        task_goal: str,
    ) -> None:
        active, inactive = self._select(task_goal)
        active.ingest_demo_frames(front_rgb_list, wrist_rgb_list, state_list, task_goal)
        if self._suppress_reset:
            inactive.ingest_demo_frames(front_rgb_list, wrist_rgb_list, state_list, task_goal)

    def ingest_step_history(
        self,
        front_rgb_list: Any,
        wrist_rgb_list: Any,
        state_list: Any,
        task_goal: str,
    ) -> None:
        active, inactive = self._select(task_goal)
        active.ingest_step_history(front_rgb_list, wrist_rgb_list, state_list, task_goal)
        if self._suppress_reset:
            inactive.ingest_step_history(front_rgb_list, wrist_rgb_list, state_list, task_goal)

    def predict_chunk(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        state: np.ndarray,
        task_goal: str,
        *,
        step_idx: int = 0,
        exec_start_idx: int = 0,
    ) -> PolicyChunkOutput:
        active, inactive = self._select(task_goal)
        if self._suppress_reset:
            inactive.ingest_step_history([front_rgb], [wrist_rgb], [state], task_goal)
        out = active.predict_chunk(
            front_rgb,
            wrist_rgb,
            state,
            task_goal,
            step_idx=step_idx,
            exec_start_idx=exec_start_idx,
        )
        out.aux_info["routed_specialist"] = bool(active is self.specialist)
        return out


def parse_tasks(spec: str) -> tuple[str, ...]:
    if spec.strip().lower() in ("all", "16", "*"):
        return ROBOMME_TASKS
    if spec.strip() in ROBOMME_SUITES:
        return ROBOMME_SUITES[spec.strip()]
    parts = [p.strip() for p in spec.replace(",", " ").split() if p.strip()]
    for p in parts:
        if p not in ROBOMME_TASKS:
            raise ValueError(f"Unknown RoboMME task {p!r}. Valid tasks: {ROBOMME_TASKS}")
    return tuple(parts)


def load_optional_memory_adapter(
    adapter_spec: str | None,
    adapter_ckpt: Path | None,
    policy_model: Any,
    device: str,
    hop_mode: str = "auto",
) -> Any | None:
    """Dynamically load `@pi05-trainer`'s memory adapter (`module.path:factory_fn`) when provided."""
    if not adapter_spec and adapter_ckpt is None:
        return None
    spec = adapter_spec or "pi05_mem.adapter:build_eval_adapter"
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    builder = getattr(mod, fn_name)
    import inspect

    kwargs: dict[str, Any] = {"ckpt_path": adapter_ckpt, "pi05_model": policy_model, "device": device}
    if "hop_mode" in inspect.signature(builder).parameters:
        kwargs["hop_mode"] = hop_mode
    ad = builder(**kwargs)
    if (
        ad is not None
        and hasattr(ad, "stage_adapter")
        and hasattr(ad, "pointer_adapter")
        and hasattr(ad, "_stage_latch")
        and not hasattr(ad, "_pending_sub_idx")
    ):
        def _union_select(num_frames: int, *, task_goal: str, is_demo: bool, total_demo_len: int | None = None) -> list[int]:
            sel_s = ad.stage_adapter.select_write_indices(
                num_frames, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len
            )
            sel_p = ad.pointer_adapter.select_write_indices(
                num_frames, task_goal=task_goal, is_demo=is_demo, total_demo_len=total_demo_len
            )
            union = sorted(set(sel_s) | set(sel_p))
            pos = {idx: k for k, idx in enumerate(union)}
            ad._sub_s = [pos[idx] for idx in sel_s]
            ad._sub_p = [pos[idx] for idx in sel_p]
            return union

        def _union_ingest(
            front_tokens_256: torch.Tensor,
            wrist_tokens_256: torch.Tensor | None,
            raw_states: np.ndarray,
            norm_states: np.ndarray,
            *,
            task_goal: str,
            is_demo: bool,
        ) -> None:
            prev_sess_s = ad.stage_adapter.session_start_frame
            prev_exec_s = ad.stage_adapter.session_exec_start_frame
            sub_s = getattr(ad, "_sub_s", [])
            sub_p = getattr(ad, "_sub_p", [])
            if sub_s:
                idx_s = torch.tensor(sub_s, device=front_tokens_256.device, dtype=torch.long)
                ws = wrist_tokens_256[idx_s] if wrist_tokens_256 is not None else None
                ad.stage_adapter.ingest_selected_frames(
                    front_tokens_256[idx_s], ws, raw_states[sub_s], norm_states[sub_s], task_goal=task_goal, is_demo=is_demo
                )
            if sub_p:
                idx_p = torch.tensor(sub_p, device=front_tokens_256.device, dtype=torch.long)
                wp = wrist_tokens_256[idx_p] if wrist_tokens_256 is not None else None
                ad.pointer_adapter.ingest_selected_frames(
                    front_tokens_256[idx_p], wp, raw_states[sub_p], norm_states[sub_p], task_goal=task_goal, is_demo=is_demo
                )
            if ad.stage_adapter.session_start_frame != prev_sess_s or ad.stage_adapter.session_exec_start_frame != prev_exec_s:
                ad._stage_latch.reset()
                if getattr(ad, "corm", None) is not None:
                    ad.corm.reset()

        ad.select_write_indices = _union_select
        ad.ingest_selected_frames = _union_ingest
    if (
        ad is not None
        and hasattr(ad, "pi05")
        and getattr(ad, "restore_lora_mode", None) == "none"
        and not getattr(ad, "dynamic_policy_lora_on_subgoal", False)
        and any(getattr(m, "has_policy_lora", False) for m in ad.pi05.paligemma_with_expert.modules())
    ):
        from pi05_mem.deltaground_vlm import set_vlm_lora_mode

        ad.restore_lora_mode = "policy"
        set_vlm_lora_mode(ad.pi05, "policy")
    return ad


def run_interference_evaluation(
    policy: Any,
    tasks: tuple[str, ...],
    *,
    policy_name: str,
    dataset: str,
    episodes_per_task: int,
    start_episode: int,
    interference_k: int,
    exec_horizon: int,
    max_steps: int,
    seed: int,
    output_path: Path,
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate target episodes after `k-1` un-reset distractor episodes from other tasks."""
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    completed: dict[tuple[str, str, int, int], EpisodeResult] = {}
    if resume and output_path.exists():
        prev = json.loads(output_path.read_text())
        for raw_ep in prev.get("episodes", []):
            ep = EpisodeResult(**raw_ep)
            completed[(ep.task, ep.dataset, ep.episode_id, ep.model_seed)] = ep
        if completed:
            print(f"[eval_pi05] Resumed {len(completed)} k={interference_k} episodes from {output_path}")

    results: list[EpisodeResult] = list(completed.values())
    policy.suppress_reset = True

    for task in tasks:
        other_tasks = [t for t in ROBOMME_TASKS if t != task]
        probe_builder = BenchmarkEnvBuilder(
            env_id=task,
            dataset=dataset,
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )
        total_avail = probe_builder.get_episode_num()
        del probe_builder
        gc.collect()
        end_ep = min(total_avail, start_episode + episodes_per_task)

        for ep_id in range(start_episode, end_ep):
            key = (task, dataset, ep_id, seed)
            if key in completed:
                continue

            task_crc = zlib.crc32(task.encode("utf-8")) % 99991
            ep_rng = np.random.default_rng(seed * 1_000_003 + task_crc + ep_id)
            policy.force_reset()
            distractors = ep_rng.choice(other_tasks, size=interference_k - 1, replace=False).tolist()
            for d_task in distractors:
                d_builder = BenchmarkEnvBuilder(
                    env_id=d_task,
                    dataset=dataset,
                    action_space="joint_angle",
                    gui_render=False,
                    max_steps=min(240, max_steps),
                )
                d_ep = int(ep_rng.integers(0, min(20, d_builder.get_episode_num())))
                _ = evaluate_episode(
                    d_builder,
                    policy,
                    task=d_task,
                    dataset=dataset,
                    episode_id=d_ep,
                    model_seed=seed,
                    exec_horizon=exec_horizon,
                    max_steps=min(240, max_steps),
                )
                del d_builder
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            builder = BenchmarkEnvBuilder(
                env_id=task,
                dataset=dataset,
                action_space="joint_angle",
                gui_render=False,
                max_steps=max_steps,
            )
            ep_res = evaluate_episode(
                builder,
                policy,
                task=task,
                dataset=dataset,
                episode_id=ep_id,
                model_seed=seed,
                exec_horizon=exec_horizon,
                max_steps=max_steps,
            )
            del builder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            completed[key] = ep_res
            results.append(ep_res)
            status = "PASS" if ep_res.success else "FAIL"
            print(
                f"[k={interference_k} {task} ep={ep_id:02d}] {status} | steps={ep_res.steps} | "
                f"distractors={distractors} | ms/chunk={ep_res.mean_ms_per_chunk:.1f}",
                flush=True,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    {
                        "policy_name": policy_name,
                        "config": {
                            "interference_k": interference_k,
                            "dataset": dataset,
                            "tasks": list(tasks),
                            "episodes_per_task": episodes_per_task,
                            "start_episode": start_episode,
                            "model_seeds": [seed],
                            "exec_horizon": exec_horizon,
                            "max_steps": max_steps,
                        },
                        "summary": summarize_episodes(results),
                        "episodes": [r.to_dict() for r in results],
                    },
                    indent=2,
                )
                + "\n"
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    policy.suppress_reset = False
    payload = {
        "policy_name": policy_name,
        "config": {
            "interference_k": interference_k,
            "dataset": dataset,
            "tasks": list(tasks),
            "episodes_per_task": episodes_per_task,
            "start_episode": start_episode,
            "model_seeds": [seed],
            "exec_horizon": exec_horizon,
            "max_steps": max_steps,
        },
        "summary": summarize_episodes(results),
        "episodes": [r.to_dict() for r in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _merge_shard_jsons(
    shard_paths: list[Path],
    output_path: Path,
    *,
    policy_name: str,
    tasks: tuple[str, ...],
    args: argparse.Namespace,
    wall_time_sec: float,
) -> dict[str, Any] | None:
    """Atomically merge per-episode results from `output_path` and `shard_paths` in canonical task order."""
    by_key: dict[tuple[str, str, int, int], EpisodeResult] = {}
    shard_hw: list[dict[str, Any]] = []

    sources = ([output_path] if (not args.no_resume and output_path.exists()) else []) + shard_paths
    for sp in sources:
        if not sp.exists():
            continue
        try:
            raw = json.loads(sp.read_text())
        except Exception:
            continue
        if "hardware_telemetry" in raw and sp in shard_paths:
            shard_hw.append(raw["hardware_telemetry"])
        for ep_d in raw.get("episodes", []):
            ep = EpisodeResult(**ep_d)
            if ep.task in tasks and args.start_episode <= ep.episode_id < args.start_episode + args.episodes:
                by_key[(ep.task, ep.dataset, ep.episode_id, ep.model_seed)] = ep

    if not by_key:
        return None

    task_order = {t: idx for idx, t in enumerate(ROBOMME_TASKS)}
    ordered_eps = sorted(
        by_key.values(),
        key=lambda e: (task_order.get(e.task, 999), e.episode_id, e.model_seed),
    )

    cfg: dict[str, Any] = {
        "tasks": list(tasks),
        "dataset": args.dataset,
        "model_seeds": [args.seed],
        "exec_horizon": args.exec_horizon,
        "max_steps": args.max_steps,
        "start_episode": args.start_episode,
        "episodes_per_task": args.episodes,
        "parallel_workers": args.parallel_workers,
    }
    if args.interference_k > 1:
        cfg["interference_k"] = args.interference_k

    payload: dict[str, Any] = {
        "policy_name": policy_name,
        "config": cfg,
        "summary": summarize_episodes(ordered_eps),
        "episodes": [e.to_dict() for e in ordered_eps],
    }
    if shard_hw:
        payload["hardware_telemetry"] = {
            "device_name": shard_hw[0].get("device_name", "cuda"),
            "parallel_workers": len(shard_paths),
            "per_worker_model_vram_mib": max(h.get("model_only_vram_mib", 0.0) for h in shard_hw),
            "per_worker_peak_vram_mib": max(h.get("peak_eval_with_sapien_vram_mib", 0.0) for h in shard_hw),
            "estimated_total_peak_vram_mib": round(
                sum(h.get("peak_eval_with_sapien_vram_mib", 0.0) for h in shard_hw), 2
            ),
            "mem_cfg_scale": args.mem_cfg_scale,
            "num_inference_steps": args.num_steps,
            "precision": args.precision,
            "wall_time_sec": round(wall_time_sec, 2),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(output_path)
    return payload


def run_parallel_sharded(
    tasks: tuple[str, ...],
    policy_name: str,
    args: argparse.Namespace,
) -> None:
    """Run `N` task-sharded GPU worker subprocesses with crash-safe preemption & live JSON merging."""
    n_workers = min(args.parallel_workers, len(tasks))
    shard_tasks_list = [tasks[i::n_workers] for i in range(n_workers)]
    shard_paths = [
        args.output.with_name(f"{args.output.stem}.shard{i}{args.output.suffix}")
        for i in range(n_workers)
    ]

    if args.no_resume:
        for sp in shard_paths:
            sp.unlink(missing_ok=True)
    else:
        # Seed shard JSONs from existing `args.output` + any prior `.shard*.json` so resume skips 100% of finished eps
        merged_existing = _merge_shard_jsons(
            shard_paths, args.output, policy_name=policy_name, tasks=tasks, args=args, wall_time_sec=0.0
        )
        if merged_existing is not None:
            all_eps = merged_existing["episodes"]
            for i, stasks in enumerate(shard_tasks_list):
                st_set = set(stasks)
                sub_eps = [e for e in all_eps if e["task"] in st_set]
                shard_paths[i].parent.mkdir(parents=True, exist_ok=True)
                shard_paths[i].write_text(
                    json.dumps({"policy_name": policy_name, "episodes": sub_eps}, indent=2) + "\n"
                )
            print(
                f"[eval_pi05:parallel={n_workers}] Seeded {len(all_eps)} completed episodes across {n_workers} shards",
                flush=True,
            )

    t0 = time.perf_counter()
    procs: list[subprocess.Popen[str]] = []
    terminated = False

    def _handle_sig(signum: int, _frame: Any) -> None:
        nonlocal terminated
        terminated = True
        print(f"\n[eval_pi05:parallel] Caught signal {signum}; stopping workers & merging completed episodes...", flush=True)
        for p in procs:
            if p.poll() is None:
                p.terminate()

    old_sigterm = signal.signal(signal.SIGTERM, _handle_sig)
    old_sigint = signal.signal(signal.SIGINT, _handle_sig)

    try:
        env = os.environ.copy()
        env.setdefault("DISPLAY", "")
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        if args.specialist_ckpt_dir is not None and args.interference_k <= 1:
            env.setdefault("PI05_OFFLOAD_INACTIVE_SPECIALIST", "1")

        for i, stasks in enumerate(shard_tasks_list):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--ckpt-dir",
                str(args.ckpt_dir),
                "--policy-name",
                policy_name,
                "--tasks",
                ",".join(stasks),
                "--dataset",
                args.dataset,
                "--episodes",
                str(args.episodes),
                "--start-episode",
                str(args.start_episode),
                "--exec-horizon",
                str(args.exec_horizon),
                "--max-steps",
                str(args.max_steps),
                "--num-steps",
                str(args.num_steps),
                "--mem-cfg-scale",
                str(args.mem_cfg_scale),
                "--noise-temp",
                str(args.noise_temp),
                "--best-of-k",
                str(args.best_of_k),
                "--rtc-weight",
                str(args.rtc_weight),
                "--ode-solver",
                args.ode_solver,
                "--precision",
                args.precision,
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--hop-mode",
                args.hop_mode,
                "--interference-k",
                str(args.interference_k),
                "--parallel-workers",
                "1",
                "--output",
                str(shard_paths[i]),
            ]
            if args.mem_mod_mode is not None:
                cmd.extend(["--mem-mod-mode", args.mem_mod_mode])
            if args.tokenizer_path is not None:
                cmd.extend(["--tokenizer-path", str(args.tokenizer_path)])
            if args.memory_adapter is not None:
                cmd.extend(["--memory-adapter", args.memory_adapter])
            if args.memory_ckpt is not None:
                cmd.extend(["--memory-ckpt", str(args.memory_ckpt)])
            if args.specialist_ckpt_dir is not None:
                cmd.extend(["--specialist-ckpt-dir", str(args.specialist_ckpt_dir)])
            if args.specialist_memory_adapter is not None:
                cmd.extend(["--specialist-memory-adapter", args.specialist_memory_adapter])
            if args.specialist_memory_ckpt is not None:
                cmd.extend(["--specialist-memory-ckpt", str(args.specialist_memory_ckpt)])
            if args.specialist_hop_mode is not None:
                cmd.extend(["--specialist-hop-mode", args.specialist_hop_mode])
            if args.specialist_route != "default":
                cmd.extend(["--specialist-route", args.specialist_route])
            if args.antithetic:
                cmd.append("--antithetic")
            if args.save_videos:
                cmd.append("--save-videos")

            print(f"[eval_pi05:shard{i}] Launching tasks={list(stasks)} -> {shard_paths[i].name}", flush=True)
            procs.append(subprocess.Popen(cmd, env=env))
            if i < n_workers - 1:
                time.sleep(25.0)

        while any(p.poll() is None for p in procs):
            time.sleep(4.0)
            _merge_shard_jsons(
                shard_paths,
                args.output,
                policy_name=policy_name,
                tasks=tasks,
                args=args,
                wall_time_sec=time.perf_counter() - t0,
            )

        rcodes = [p.wait() for p in procs]
        final_payload = _merge_shard_jsons(
            shard_paths,
            args.output,
            policy_name=policy_name,
            tasks=tasks,
            args=args,
            wall_time_sec=time.perf_counter() - t0,
        )
        if any(rc != 0 for rc in rcodes) or terminated:
            n_saved = len(final_payload["episodes"]) if final_payload else 0
            print(
                f"[eval_pi05:parallel] Exited with codes {rcodes} (saved {n_saved} episodes to {args.output})",
                flush=True,
            )
            sys.exit(1 if any(rc != 0 for rc in rcodes) else 0)

        for sp in shard_paths:
            sp.unlink(missing_ok=True)

        assert final_payload is not None
        summary = final_payload["summary"]
        overall = summary["overall"]
        hw = final_payload.get("hardware_telemetry", {})
        print("\n=== Parallel Sharded Evaluation Summary ===", flush=True)
        print(
            f"Policy: {policy_name} | Split: {args.dataset} | Workers: {n_workers} | "
            f"Episodes: {overall['num_episodes']} | SR: {overall['success_rate']*100:.2f}% "
            f"(95% CI [{overall['success_ci95_low']*100:.2f}%, {overall['success_ci95_high']*100:.2f}%]) | "
            f"Mean Chunk: {overall['mean_ms_per_chunk']:.1f} ms | "
            f"Wall: {time.perf_counter() - t0:.1f}s | Est Total VRAM: {hw.get('estimated_total_peak_vram_mib', 0.0):.1f} MiB",
            flush=True,
        )
        for suite_name, s_info in summary.get("suites", {}).items():
            n_ep = s_info["num_episodes"]
            n_ok = round(s_info["success_rate"] * n_ep)
            print(f"  Suite {suite_name:11s}: {s_info['success_rate']*100:5.2f}% ({n_ok}/{n_ep})", flush=True)
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)
        for p in procs:
            if p.poll() is None:
                p.kill()
        _merge_shard_jsons(
            shard_paths,
            args.output,
            policy_name=policy_name,
            tasks=tasks,
            args=args,
            wall_time_sec=time.perf_counter() - t0,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PyTorch pi0.5 / DeltaMem-pi0.5 on RoboMME.")
    parser.add_argument("--ckpt-dir", type=Path, required=True, help="Converted PyTorch pi0.5 checkpoint dir.")
    parser.add_argument("--policy-name", type=str, default=None, help="Human-readable run name for JSON summary.")
    parser.add_argument("--tasks", type=str, default="all", help="'all', suite name, or task list.")
    parser.add_argument("--dataset", "--split", dest="dataset", choices=("val", "test", "train"), default="test")
    parser.add_argument("--episodes", "--episodes-per-task", dest="episodes", type=int, default=5, help="Max episodes per task.")
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--exec-horizon", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=1300)
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-matching ODE NFE budget.")
    parser.add_argument("--ode-solver", choices=("euler", "midpoint", "heun"), default="euler", help="Flow-ODE solver.")
    parser.add_argument("--noise-temp", type=float, default=1.0, help="Initial Gaussian noise temperature tau.")
    parser.add_argument("--antithetic", action="store_true", help="BS=2 [+eps, -eps] antithetic ODE chunk averaging.")
    parser.add_argument("--best-of-k", type=int, default=1, help="Select 1 sharp medoid+RTC trajectory out of K (0 mode avg).")
    parser.add_argument("--rtc-weight", type=float, default=0.25, help="RTC overlap continuity weight for --best-of-k.")
    parser.add_argument("--mem-mod-mode", choices=("none", "modul_residual", "mme_vla_exact"), default=None)
    parser.add_argument("--mem-cfg-scale", type=float, default=1.0, help="Memory Classifier-Free Guidance weight.")
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--memory-adapter", type=str, default=None, help="Optional module:builder for DeltaMem.")
    parser.add_argument("--memory-ckpt", "--adapter-ckpt", dest="memory_ckpt", type=Path, default=None, help="Optional Stage A/B/C DeltaMem ckpt.")
    parser.add_argument("--hop-mode", choices=("auto", "l1", "l2", "l3"), default="auto", help="Loop hop mode (Stage C / DeltaGround).")
    parser.add_argument("--specialist-ckpt-dir", "--specialist-ckpt", dest="specialist_ckpt_dir", type=Path, default=None, help="Optional zero-oracle P2-M10 symbolic specialist ckpt dir.")
    parser.add_argument("--specialist-memory-adapter", type=str, default=None, help="Optional module:builder for P2-M10 specialist adapter.")
    parser.add_argument("--specialist-memory-ckpt", type=Path, default=None, help="Optional ckpt for P2-M10 specialist adapter.")
    parser.add_argument("--specialist-hop-mode", choices=("auto", "l1", "l2", "l3"), default=None, help="Optional hop mode override for P2-M10 specialist.")
    parser.add_argument("--specialist-route", choices=("default", "8task", "highlight"), default="default", help="Specialist routing rule: 'default' (6-task), '8task' (all 8 Permanence+Reference tasks), or 'highlight'-only.")
    parser.add_argument("--interference-k", type=int, default=1, help="Un-reset session length k (1 = standard).")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Number of parallel GPU shard processes.")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing output JSON.")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path under results/phase2/.")
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    policy_name = args.policy_name or args.ckpt_dir.name

    if args.parallel_workers > 1 and len(tasks) > 1:
        run_parallel_sharded(tasks, policy_name, args)
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    print(
        f"Loading Pi05RoboMMEPolicy `{policy_name}` from {args.ckpt_dir} "
        f"(precision={args.precision}, device={args.device}, mem_cfg={args.mem_cfg_scale}) ...",
        flush=True,
    )
    policy = Pi05RoboMMEPolicy(
        args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        precision=args.precision,
        mem_mod_mode=args.mem_mod_mode,
        num_inference_steps=args.num_steps,
        mem_cfg_scale=args.mem_cfg_scale,
        noise_temp=args.noise_temp,
        antithetic=args.antithetic,
        best_of_k=args.best_of_k,
        rtc_weight=args.rtc_weight,
        ode_solver=args.ode_solver,
        seed=args.seed,
    )
    adapter = load_optional_memory_adapter(
        args.memory_adapter, args.memory_ckpt, policy.model, args.device, hop_mode=args.hop_mode
    )
    if adapter is not None:
        policy.memory_adapter = adapter
        policy.framesamp_buffer = None

    eval_policy: Any = policy
    if args.specialist_ckpt_dir is not None:
        print(
            f"Loading P2-M10 zero-oracle specialist from {args.specialist_ckpt_dir} "
            f"(sharing frozen SigLIP vision_tower, adapter={args.specialist_memory_adapter}) ...",
            flush=True,
        )
        import weakref

        offload_spec = os.environ.get("PI05_OFFLOAD_INACTIVE_SPECIALIST", "0") == "1" and args.interference_k <= 1
        spec_init_dev = "cpu" if offload_spec else args.device
        spec_model, spec_stats = load_pi05_checkpoint(
            args.specialist_ckpt_dir,
            device="cpu",
            precision="bfloat16" if args.precision == "bfloat16" else "float32",
        )
        del spec_model.vision_tower
        policy.model.vision_tower._host_perceptual_pi05 = weakref.ref(policy.model)  # type: ignore[attr-defined]
        spec_model.vision_tower = policy.model.vision_tower
        gc.collect()
        specialist_policy = Pi05RoboMMEPolicy(
            spec_model,
            norm_stats=spec_stats,
            tokenizer_path=args.tokenizer_path or resolve_tokenizer_path(None, args.specialist_ckpt_dir),
            device=spec_init_dev,
            precision=args.precision,
            num_inference_steps=args.num_steps,
            mem_cfg_scale=args.mem_cfg_scale,
            noise_temp=args.noise_temp,
            antithetic=args.antithetic,
            best_of_k=args.best_of_k,
            rtc_weight=args.rtc_weight,
            ode_solver=args.ode_solver,
            seed=args.seed,
        )
        if offload_spec:
            policy.model.vision_tower.to(args.device)
        spec_adapter = load_optional_memory_adapter(
            args.specialist_memory_adapter,
            args.specialist_memory_ckpt,
            specialist_policy.model,
            spec_init_dev,
            hop_mode=args.specialist_hop_mode or args.hop_mode,
        )
        if spec_adapter is not None:
            specialist_policy.memory_adapter = spec_adapter
            specialist_policy.framesamp_buffer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_policy = ZeroOraclePromptRoutedPolicy(
            policy, specialist_policy, route_mode=args.specialist_route, offload_inactive=offload_spec
        )

    model_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    print(f"Policy ready in {time.perf_counter() - t0:.2f}s (model VRAM = {model_vram_mb:.1f} MiB)", flush=True)

    if args.interference_k > 1:
        payload = run_interference_evaluation(
            eval_policy,
            tasks,
            policy_name=policy_name,
            dataset=args.dataset,
            episodes_per_task=args.episodes,
            start_episode=args.start_episode,
            interference_k=args.interference_k,
            exec_horizon=args.exec_horizon,
            max_steps=args.max_steps,
            seed=args.seed,
            output_path=args.output,
            resume=not args.no_resume,
        )
    else:
        ep_ids = list(range(args.start_episode, args.start_episode + args.episodes))
        video_dir = args.output.with_suffix("") / "videos" if args.save_videos else None
        payload = evaluate_policy(
            eval_policy,
            tasks=tasks,
            dataset=args.dataset,
            episode_ids=ep_ids,
            model_seeds=(args.seed,),
            exec_horizon=args.exec_horizon,
            max_steps=args.max_steps,
            output_json=args.output,
            save_videos_dir=video_dir,
            resume=not args.no_resume,
            verbose=True,
        )
        payload["policy_name"] = policy_name

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    )
    payload["hardware_telemetry"] = {
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "model_only_vram_mib": round(model_vram_mb, 2),
        "peak_eval_with_sapien_vram_mib": round(peak_vram_mb, 2),
        "mem_cfg_scale": args.mem_cfg_scale,
        "num_inference_steps": args.num_steps,
        "precision": args.precision,
        "wall_time_sec": round(time.perf_counter() - t0, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")

    summary = payload["summary"]
    overall = summary["overall"]
    print("\n=== Evaluation Summary ===", flush=True)
    print(
        f"Policy: {policy_name} | Split: {args.dataset} | Episodes: {overall['num_episodes']} | "
        f"SR: {overall['success_rate']*100:.2f}% "
        f"(95% CI [{overall['success_ci95_low']*100:.2f}%, {overall['success_ci95_high']*100:.2f}%]) | "
        f"Mean Chunk: {overall['mean_ms_per_chunk']:.1f} ms | Peak VRAM: {peak_vram_mb:.1f} MiB",
        flush=True,
    )
    for suite_name, s_info in summary.get("suites", {}).items():
        n_ep = s_info["num_episodes"]
        n_ok = round(s_info["success_rate"] * n_ep)
        print(
            f"  Suite {suite_name:11s}: {s_info['success_rate']*100:5.2f}% ({n_ok}/{n_ep})",
            flush=True,
        )


if __name__ == "__main__":
    main()
