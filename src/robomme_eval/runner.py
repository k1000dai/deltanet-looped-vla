"""Headless ManiSkill3 / RoboMME rollout evaluator for stateful & adaptive-loop visuomotor policies."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .interface import (
    DummyStatefulLoopedPolicy,
    ROBOMME_TASKS,
    StatefulPolicy,
    TASK_TO_SUITE,
    pack_robomme_state,
    to_uint8_rgb,
)
from .metrics import EpisodeResult, compare_policy_runs, summarize_episodes


def _set_model_rng_seed(seed: int) -> None:
    """Set model-side RNG seeds without modifying ManiSkill3's internal episode seed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _EpisodeVideoWriter:
    """Lightweight side-by-side `(front_rgb | wrist_rgb)` MP4 recorder with red border on demo frames."""

    def __init__(self, fps: int = 20) -> None:
        self.fps = fps
        self.frames: list[np.ndarray] = []

    def add_frame(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray, is_demo: bool = False) -> None:
        frame = np.hstack([front_rgb, wrist_rgb]).astype(np.uint8)
        if is_demo:
            frame[:6, :, :] = (255, 0, 0)
            frame[-6:, :, :] = (255, 0, 0)
            frame[:, :6, :] = (255, 0, 0)
            frame[:, -6:, :] = (255, 0, 0)
        self.frames.append(frame)

    def save(self, out_path: Path) -> None:
        if not self.frames:
            return
        import imageio

        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out_path), self.frames, fps=self.fps)
        self.frames.clear()


def evaluate_episode(
    env_builder: Any,
    policy: StatefulPolicy,
    *,
    task: str,
    dataset: str,
    episode_id: int,
    model_seed: int = 7,
    exec_horizon: int = 16,
    max_steps: int = 1300,
    video_path: Path | None = None,
) -> EpisodeResult:
    """Run a single RoboMME episode with strict stateful reset, demo-frame streaming, and chunked execution."""
    if exec_horizon < 1:
        raise ValueError(f"exec_horizon must be >= 1, got {exec_horizon}")

    t_ep_start = time.perf_counter()
    _set_model_rng_seed(model_seed)
    policy.set_seed(model_seed)
    policy.reset()

    env = env_builder.make_env_for_episode(episode_id)
    recorder = _EpisodeVideoWriter() if video_path is not None else None

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
        if not init_front or len(init_front) != len(init_wrist) or len(init_front) != len(init_states):
            raise RuntimeError(
                f"Malformed reset obs lists on {task} ep {episode_id}: "
                f"front={len(init_front)}, wrist={len(init_wrist)}, state={len(init_states)}"
            )

        # Indices [0 .. exec_start_idx - 1] are conditioning video demo frames; [-1] is first execution frame.
        exec_start_idx = len(init_front) - 1
        if recorder is not None:
            for idx, (fr, wr) in enumerate(zip(init_front, init_wrist)):
                recorder.add_frame(fr, wr, is_demo=(idx < exec_start_idx))

        if exec_start_idx > 0:
            t_demo0 = time.perf_counter()
            policy.ingest_demo_frames(
                init_front[:-1],
                init_wrist[:-1],
                init_states[:-1],
                task_goal,
            )
            demo_ingest_ms = (time.perf_counter() - t_demo0) * 1000.0
        else:
            demo_ingest_ms = 0.0

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
                step_idx=exec_start_idx + steps_executed,
                exec_start_idx=exec_start_idx,
            )
            dt_ms = (time.perf_counter() - t_inf0) * 1000.0
            chunk_infer_ms.append(dt_ms)
            loop_depths.append(float(chunk_out.loop_depth))

            actions = np.asarray(chunk_out.actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 8 or actions.shape[0] < 1:
                raise ValueError(
                    f"Policy predict_chunk must return shape (T>=1, 8), got {actions.shape}"
                )
            if not np.isfinite(actions).all():
                raise ValueError(f"Non-finite action values predicted on {task} ep {episode_id}")

            chunk_to_run = actions[:exec_horizon]
            n_sub = chunk_to_run.shape[0]
            stop_episode = False

            for sub_i in range(n_sub):
                if steps_executed >= max_steps:
                    outcome = "timeout"
                    stop_episode = True
                    break

                obs, _reward, terminated, truncated, step_info = env.step(chunk_to_run[sub_i])
                steps_executed += 1

                if step_info is not None and step_info.get("status") == "error":
                    outcome = "error"
                    error_message = str(step_info.get("error_message", "unknown simulator error"))
                    success = False
                    stop_episode = True
                    break

                curr_front = to_uint8_rgb(obs["front_rgb_list"][-1])
                curr_wrist = to_uint8_rgb(obs["wrist_rgb_list"][-1])
                curr_state = pack_robomme_state(
                    obs["joint_state_list"][-1], obs["gripper_state_list"][-1]
                )
                if recorder is not None:
                    recorder.add_frame(curr_front, curr_wrist, is_demo=False)

                if terminated or truncated:
                    outcome = str(step_info.get("status", "unknown")) if step_info else "unknown"
                    success = outcome == "success"
                    stop_episode = True
                    break

                # If this is an intermediate step within the chunk (not the final step that becomes
                # the next chunk's `curr_*`), buffer it so recurrent policies can ingest every step.
                if sub_i < n_sub - 1 and steps_executed < max_steps:
                    inter_front.append(curr_front)
                    inter_wrist.append(curr_wrist)
                    inter_states.append(curr_state)

            if stop_episode:
                break

        if recorder is not None and video_path is not None:
            stem = video_path.stem + f"_{outcome}" + video_path.suffix
            recorder.save(video_path.with_name(stem))

        total_infer_ms = demo_ingest_ms + float(sum(chunk_infer_ms))
        mean_ms = float(np.mean(chunk_infer_ms)) if chunk_infer_ms else 0.0
        loops_arr = np.asarray(loop_depths, dtype=np.float64)

        return EpisodeResult(
            task=task,
            suite=TASK_TO_SUITE.get(task, "Unknown"),
            dataset=dataset,
            episode_id=episode_id,
            model_seed=model_seed,
            success=success,
            outcome=outcome,
            steps=steps_executed,
            demo_frames=exec_start_idx,
            num_chunks=len(chunk_infer_ms),
            mean_loop_depth=float(loops_arr.mean()) if loops_arr.size else 0.0,
            p95_loop_depth=float(np.percentile(loops_arr, 95.0)) if loops_arr.size else 0.0,
            loop_depths=loop_depths,
            mean_ms_per_chunk=mean_ms,
            total_infer_ms=total_infer_ms,
            episode_wall_s=time.perf_counter() - t_ep_start,
            task_goal=task_goal,
            difficulty=str(difficulty) if difficulty is not None else None,
            error_message=error_message,
        )
    finally:
        env.close()


def evaluate_policy(
    policy: StatefulPolicy,
    *,
    tasks: Sequence[str] = ROBOMME_TASKS,
    dataset: str = "val",
    episode_ids: Sequence[int] | None = None,
    max_episodes_per_task: int | None = None,
    model_seeds: Sequence[int] = (7,),
    exec_horizon: int = 16,
    max_steps: int = 1300,
    output_json: str | Path | None = None,
    save_videos_dir: str | Path | None = None,
    resume: bool = True,
    n_boot: int = 10000,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate a `StatefulPolicy` across RoboMME tasks, splits, and model seeds with bootstrap CIs."""
    # Ensure headless SAPIEN works cleanly without X11
    os.environ.setdefault("DISPLAY", "")
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    out_path = Path(output_json) if output_json is not None else None
    vid_dir = Path(save_videos_dir) if save_videos_dir is not None else None

    completed: dict[tuple[str, str, int, int], EpisodeResult] = {}
    if resume and out_path is not None and out_path.exists():
        prev = json.loads(out_path.read_text())
        for raw_ep in prev.get("episodes", []):
            ep = EpisodeResult(**raw_ep)
            completed[(ep.task, ep.dataset, ep.episode_id, ep.model_seed)] = ep
        if verbose and completed:
            print(f"[robomme_eval] Resumed {len(completed)} completed episodes from {out_path}")

    all_results: list[EpisodeResult] = list(completed.values())

    for task in tasks:
        if task not in ROBOMME_TASKS:
            raise ValueError(f"Unknown RoboMME task {task!r}; expected one of {ROBOMME_TASKS}")

        env_builder = BenchmarkEnvBuilder(
            env_id=task,
            dataset=dataset,
            action_space="joint_angle",
            gui_render=False,
            max_steps=max_steps,
        )
        total_avail = env_builder.get_episode_num()
        if episode_ids is not None:
            target_eps = [ep_i for ep_i in episode_ids if 0 <= ep_i < total_avail]
        elif max_episodes_per_task is not None:
            target_eps = list(range(min(total_avail, max_episodes_per_task)))
        else:
            target_eps = list(range(total_avail))

        for seed in model_seeds:
            for ep_id in target_eps:
                key = (task, dataset, ep_id, seed)
                if key in completed:
                    continue

                vid_path = (
                    vid_dir / task / f"seed{seed}_ep{ep_id:03d}.mp4"
                    if vid_dir is not None
                    else None
                )
                ep_res = evaluate_episode(
                    env_builder,
                    policy,
                    task=task,
                    dataset=dataset,
                    episode_id=ep_id,
                    model_seed=seed,
                    exec_horizon=exec_horizon,
                    max_steps=max_steps,
                    video_path=vid_path,
                )
                completed[key] = ep_res
                all_results.append(ep_res)

                if verbose:
                    print(
                        f"[{task} | {dataset} | seed={seed} | ep={ep_id:02d}] "
                        f"outcome={ep_res.outcome:<7s} success={int(ep_res.success)} "
                        f"demo_frames={ep_res.demo_frames:3d} steps={ep_res.steps:4d} "
                        f"chunks={ep_res.num_chunks:3d} L_mean={ep_res.mean_loop_depth:.2f} "
                        f"ms/chunk={ep_res.mean_ms_per_chunk:.2f} wall={ep_res.episode_wall_s:.2f}s"
                    )

                if out_path is not None:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "config": {
                            "tasks": list(tasks),
                            "dataset": dataset,
                            "model_seeds": list(model_seeds),
                            "exec_horizon": exec_horizon,
                            "max_steps": max_steps,
                        },
                        "summary": summarize_episodes(all_results, n_boot=n_boot),
                        "episodes": [r.to_dict() for r in all_results],
                    }
                    out_path.write_text(json.dumps(payload, indent=2))

    summary = summarize_episodes(all_results, n_boot=n_boot)
    final_payload = {
        "config": {
            "tasks": list(tasks),
            "dataset": dataset,
            "model_seeds": list(model_seeds),
            "exec_horizon": exec_horizon,
            "max_steps": max_steps,
        },
        "summary": summary,
        "episodes": [r.to_dict() for r in all_results],
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(final_payload, indent=2))
    return final_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Headless RoboMME stateful policy evaluator")
    p.add_argument(
        "--tasks",
        nargs="+",
        default=["ButtonUnmaskSwap", "VideoUnmaskSwap"],
        help="RoboMME task IDs (or 'all' for all 16 tasks)",
    )
    p.add_argument("--dataset", default="val", choices=["train", "val", "test"])
    p.add_argument("--episodes", type=int, default=2, help="Max episodes per task")
    p.add_argument("--seeds", type=int, nargs="+", default=[7], help="Model RNG seeds")
    p.add_argument("--exec-horizon", type=int, default=16, help="Executed actions per chunk")
    p.add_argument("--pred-horizon", type=int, default=16, help="Predicted actions per chunk (dummy policy)")
    p.add_argument("--max-loops", type=int, default=4, help="Max loop depth (dummy policy)")
    p.add_argument("--max-steps", type=int, default=1300, help="Max environment steps per episode")
    p.add_argument("--output-json", type=str, default=None, help="Optional JSON path to save results")
    p.add_argument("--save-videos-dir", type=str, default=None, help="Optional directory for rollout MP4s")
    p.add_argument(
        "--compare-dummy-baseline",
        action="store_true",
        help="Also run a 1-loop dummy baseline and print paired bootstrap 95%% CI verification",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    tasks = ROBOMME_TASKS if args.tasks == ["all"] else tuple(args.tasks)

    policy = DummyStatefulLoopedPolicy(
        pred_horizon=max(args.pred_horizon, args.exec_horizon),
        max_loops=args.max_loops,
    )
    res = evaluate_policy(
        policy,
        tasks=tasks,
        dataset=args.dataset,
        max_episodes_per_task=args.episodes,
        model_seeds=args.seeds,
        exec_horizon=args.exec_horizon,
        max_steps=args.max_steps,
        output_json=args.output_json,
        save_videos_dir=args.save_videos_dir,
        resume=False,
    )
    ov = res["summary"]["overall"]
    print(
        f"\n[Summary] episodes={ov['num_episodes']} "
        f"SR={ov['success_rate']*100:.1f}% (95% CI [{ov['success_ci95_low']*100:.1f}%, {ov['success_ci95_high']*100:.1f}%]) "
        f"mean_steps={ov['mean_steps']:.1f} mean_demo_frames={ov['mean_demo_frames']:.1f} "
        f"L_mean={ov['mean_loop_depth']:.2f} (p95={ov['p95_loop_depth']:.2f}) "
        f"ms/chunk={ov['mean_ms_per_chunk']:.2f}"
    )

    if args.compare_dummy_baseline:
        base_policy = DummyStatefulLoopedPolicy(
            pred_horizon=max(args.pred_horizon, args.exec_horizon),
            max_loops=1,
        )
        res_base = evaluate_policy(
            base_policy,
            tasks=tasks,
            dataset=args.dataset,
            max_episodes_per_task=args.episodes,
            model_seeds=args.seeds,
            exec_horizon=args.exec_horizon,
            max_steps=args.max_steps,
            resume=False,
            verbose=False,
        )
        eps_a = [EpisodeResult(**d) for d in res["episodes"]]
        eps_b = [EpisodeResult(**d) for d in res_base["episodes"]]
        comp = compare_policy_runs(eps_a, eps_b)
        print(f"[Paired Bootstrap Verification] {json.dumps(comp, indent=2)}")


if __name__ == "__main__":
    main()
