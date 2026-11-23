"""Evaluation metrics, suite aggregation, and paired bootstrap 95% confidence intervals for RoboMME."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Sequence

import numpy as np

from .interface import ROBOMME_SUITES, TASK_TO_SUITE


def exact_mcnemar_two_sided_p(n10: int, n01: int) -> float:
    """Exact two-sided binomial sign test on discordant episode pairs `(n10, n01)`."""
    n_disc = int(n10) + int(n01)
    if n_disc <= 0:
        return 1.0
    k_min = min(int(n10), int(n01))
    tail = sum(math.comb(n_disc, j) for j in range(k_min + 1)) * (0.5 ** n_disc)
    return float(min(1.0, 2.0 * tail))


@dataclass
class EpisodeResult:
    """Detailed per-episode rollout telemetry."""

    task: str
    suite: str
    dataset: str
    episode_id: int
    model_seed: int
    success: bool
    outcome: str
    steps: int
    demo_frames: int
    num_chunks: int
    mean_loop_depth: float
    p95_loop_depth: float
    loop_depths: list[float] = field(default_factory=list)
    mean_ms_per_chunk: float = 0.0
    total_infer_ms: float = 0.0
    episode_wall_s: float = 0.0
    task_goal: str = ""
    difficulty: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskEvalSummary:
    """Aggregated metrics for a single RoboMME task or suite."""

    name: str
    num_episodes: int
    success_rate: float
    success_ci95_low: float
    success_ci95_high: float
    mean_steps: float
    mean_demo_frames: float
    mean_loop_depth: float
    p50_loop_depth: float
    p95_loop_depth: float
    mean_ms_per_chunk: float
    p95_ms_per_chunk: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_ci(
    values: Sequence[float | bool | int],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute sample mean and percentile bootstrap `(1 - alpha)` confidence interval `(mean, low, high)`."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(arr.mean())
    if arr.size == 1 or np.all(arr == arr[0]):
        return mean_val, mean_val, mean_val

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[boot_idx].mean(axis=1)
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_val, low, high


def paired_bootstrap_ci(
    values_a: Sequence[float | bool | int],
    values_b: Sequence[float | bool | int],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Compute paired bootstrap `(1 - alpha)` CI for `A - B` across matched episodes/seeds.

    Returns a dictionary containing `mean_a`, `mean_b`, `mean_diff`, `ci95_low`, `ci95_high`,
    `p_value_two_sided`, and `n_pairs`.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired bootstrap requires identical shapes, got {a.shape} vs {b.shape}")
    if a.size == 0:
        return {
            "n_pairs": 0,
            "mean_a": 0.0,
            "mean_b": 0.0,
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value_two_sided": 1.0,
            "p_value_one_sided_gt0": 1.0,
            "p_value_one_sided_lt0": 1.0,
        }

    diffs = a - b
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    mean_diff = float(diffs.mean())

    if diffs.size == 1 or np.all(diffs == diffs[0]):
        mc_floor = float(1.0 / (n_boot + 1.0))
        p_val = 1.0 if mean_diff == 0.0 else mc_floor
        p_gt0 = mc_floor if mean_diff > 0.0 else 1.0
        p_lt0 = mc_floor if mean_diff < 0.0 else 1.0
        return {
            "n_pairs": int(diffs.size),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_diff": mean_diff,
            "ci_low": mean_diff,
            "ci_high": mean_diff,
            "p_value_two_sided": round(p_val, 6),
            "p_bootstrap_phipson_smyth_2tail": round(1.0 if mean_diff == 0.0 else min(1.0, 2.0 * mc_floor), 6),
            "p_value_one_sided_gt0": round(p_gt0, 6),
            "p_value_one_sided_lt0": round(p_lt0, 6),
        }

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_diffs = diffs[boot_idx].mean(axis=1)
    low = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))

    # Finite-sample Monte Carlo bound (Phipson & Smyth 2010): never emit 0.0 from finite B draws
    r_le = int(np.sum(boot_diffs <= 0.0))
    r_ge = int(np.sum(boot_diffs >= 0.0))
    r_min = min(r_le, r_ge)
    mc_floor = float(1.0 / (n_boot + 1.0))
    p_le_zero = mc_floor if r_le == 0 else float(r_le / n_boot)
    p_ge_zero = mc_floor if r_ge == 0 else float(r_ge / n_boot)
    p_two_sided = mc_floor if r_min == 0 else float(min(1.0, 2.0 * (r_min / n_boot)))
    p_ps_2tail = float(min(1.0, 2.0 * (r_min + 1.0) / (n_boot + 1.0)))

    return {
        "n_pairs": int(diffs.size),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "ci_low": low,
        "ci_high": high,
        "p_value_two_sided": round(p_two_sided, 6),
        "p_bootstrap_phipson_smyth_2tail": round(p_ps_2tail, 6),
        "p_value_one_sided_gt0": round(p_le_zero, 6),
        "p_value_one_sided_lt0": round(p_ge_zero, 6),
    }


def _aggregate_group(
    name: str,
    episodes: Sequence[EpisodeResult],
    n_boot: int = 10000,
    seed: int = 0,
) -> TaskEvalSummary:
    if not episodes:
        return TaskEvalSummary(
            name=name,
            num_episodes=0,
            success_rate=0.0,
            success_ci95_low=0.0,
            success_ci95_high=0.0,
            mean_steps=0.0,
            mean_demo_frames=0.0,
            mean_loop_depth=0.0,
            p50_loop_depth=0.0,
            p95_loop_depth=0.0,
            mean_ms_per_chunk=0.0,
            p95_ms_per_chunk=0.0,
        )

    sr, sr_low, sr_high = bootstrap_ci([float(ep.success) for ep in episodes], n_boot=n_boot, seed=seed)
    all_loops = [ld for ep in episodes for ld in ep.loop_depths]
    if not all_loops:
        all_loops = [ep.mean_loop_depth for ep in episodes]
    loops_arr = np.asarray(all_loops, dtype=np.float64)
    chunk_ms_arr = np.asarray([ep.mean_ms_per_chunk for ep in episodes], dtype=np.float64)

    return TaskEvalSummary(
        name=name,
        num_episodes=len(episodes),
        success_rate=sr,
        success_ci95_low=sr_low,
        success_ci95_high=sr_high,
        mean_steps=float(np.mean([ep.steps for ep in episodes])),
        mean_demo_frames=float(np.mean([ep.demo_frames for ep in episodes])),
        mean_loop_depth=float(loops_arr.mean()) if loops_arr.size else 0.0,
        p50_loop_depth=float(np.percentile(loops_arr, 50.0)) if loops_arr.size else 0.0,
        p95_loop_depth=float(np.percentile(loops_arr, 95.0)) if loops_arr.size else 0.0,
        mean_ms_per_chunk=float(chunk_ms_arr.mean()) if chunk_ms_arr.size else 0.0,
        p95_ms_per_chunk=float(np.percentile(chunk_ms_arr, 95.0)) if chunk_ms_arr.size else 0.0,
    )


def summarize_episodes(
    episodes: Sequence[EpisodeResult],
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize episode results at overall, memory-suite, and individual-task granularity."""
    by_task: dict[str, list[EpisodeResult]] = {}
    by_suite: dict[str, list[EpisodeResult]] = {suite: [] for suite in ROBOMME_SUITES}

    for ep in episodes:
        by_task.setdefault(ep.task, []).append(ep)
        suite = ep.suite or TASK_TO_SUITE.get(ep.task, "Unknown")
        by_suite.setdefault(suite, []).append(ep)

    task_summaries = {
        task: _aggregate_group(task, task_eps, n_boot=n_boot, seed=seed).to_dict()
        for task, task_eps in by_task.items()
    }
    suite_summaries = {
        suite: _aggregate_group(suite, suite_eps, n_boot=n_boot, seed=seed).to_dict()
        for suite, suite_eps in by_suite.items()
        if suite_eps
    }
    overall_summary = _aggregate_group("overall", list(episodes), n_boot=n_boot, seed=seed).to_dict()

    return {
        "overall": overall_summary,
        "suites": suite_summaries,
        "tasks": task_summaries,
    }


def compare_policy_runs(
    episodes_a: Sequence[EpisodeResult],
    episodes_b: Sequence[EpisodeResult],
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute paired bootstrap 95% CIs (`A - B`) aligned by `(task, dataset, episode_id, model_seed)`."""
    map_a = {(e.task, e.dataset, e.episode_id, e.model_seed): e for e in episodes_a}
    map_b = {(e.task, e.dataset, e.episode_id, e.model_seed): e for e in episodes_b}
    common_keys = sorted(set(map_a).intersection(map_b))
    if not common_keys:
        raise ValueError("No overlapping (task, dataset, episode_id, model_seed) episodes between run A and run B")

    paired_a = [map_a[k] for k in common_keys]
    paired_b = [map_b[k] for k in common_keys]

    ov_diff = paired_bootstrap_ci(
        [float(e.success) for e in paired_a],
        [float(e.success) for e in paired_b],
        n_boot=n_boot,
        seed=seed,
    )
    n10 = sum(1 for ea, eb in zip(paired_a, paired_b) if bool(ea.success) and not bool(eb.success))
    n01 = sum(1 for ea, eb in zip(paired_a, paired_b) if (not bool(ea.success)) and bool(eb.success))
    p_mcn = exact_mcnemar_two_sided_p(n10, n01)
    ov_diff["discordant_a_only"] = int(n10)
    ov_diff["discordant_b_only"] = int(n01)
    ov_diff["p_bootstrap_two_sided"] = float(ov_diff["p_value_two_sided"])
    ov_diff["p_mcnemar_two_sided"] = float(f"{p_mcn:.6g}")
    ov_diff["p_mcnemar_two_sided_4dp"] = round(p_mcn, 4) if p_mcn >= 1e-4 else float(f"{p_mcn:.4g}")

    out: dict[str, Any] = {
        "overall_success_diff": ov_diff,
        "overall_loop_depth_diff": paired_bootstrap_ci(
            [e.mean_loop_depth for e in paired_a],
            [e.mean_loop_depth for e in paired_b],
            n_boot=n_boot,
            seed=seed,
        ),
        "suites": {},
        "tasks": {},
    }

    for suite in ROBOMME_SUITES:
        idx = [i for i, e in enumerate(paired_a) if e.suite == suite]
        if idx:
            out["suites"][suite] = paired_bootstrap_ci(
                [float(paired_a[i].success) for i in idx],
                [float(paired_b[i].success) for i in idx],
                n_boot=n_boot,
                seed=seed,
            )

    tasks_present = sorted({e.task for e in paired_a})
    for task in tasks_present:
        idx = [i for i, e in enumerate(paired_a) if e.task == task]
        out["tasks"][task] = paired_bootstrap_ci(
            [float(paired_a[i].success) for i in idx],
            [float(paired_b[i].success) for i in idx],
            n_boot=n_boot,
            seed=seed,
        )

    return out


def compare_did_policy_runs(
    episodes_a_k3: Sequence[EpisodeResult],
    episodes_a_k0: Sequence[EpisodeResult],
    episodes_b_k3: Sequence[EpisodeResult],
    episodes_b_k0: Sequence[EpisodeResult],
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute paired bootstrap Difference-in-Differences `(A_k3 - A_k0) - (B_k3 - B_k0)` on matched episodes."""
    map_ak3 = {(e.task, e.dataset, e.episode_id, e.model_seed): e for e in episodes_a_k3}
    map_ak0 = {(e.task, e.dataset, e.episode_id, e.model_seed): e for e in episodes_a_k0}
    map_bk3 = {(e.task, e.dataset, e.episode_id, e.model_seed): e for e in episodes_b_k3}
    map_bk0 = {(e.task, e.dataset, e.episode_id, e.model_seed): e for e in episodes_b_k0}
    common_keys = sorted(set(map_ak3) & set(map_ak0) & set(map_bk3) & set(map_bk0))
    if not common_keys:
        raise ValueError("No overlapping (task, dataset, episode_id, model_seed) episodes across all 4 DiD runs")

    delta_a = [float(map_ak3[k].success) - float(map_ak0[k].success) for k in common_keys]
    delta_b = [float(map_bk3[k].success) - float(map_bk0[k].success) for k in common_keys]
    ld_delta_a = [map_ak3[k].mean_loop_depth - map_ak0[k].mean_loop_depth for k in common_keys]
    ld_delta_b = [map_bk3[k].mean_loop_depth - map_bk0[k].mean_loop_depth for k in common_keys]
    suites_k = [map_ak3[k].suite for k in common_keys]
    tasks_k = [map_ak3[k].task for k in common_keys]

    out: dict[str, Any] = {
        "overall_success_diff": paired_bootstrap_ci(delta_a, delta_b, n_boot=n_boot, seed=seed),
        "overall_loop_depth_diff": paired_bootstrap_ci(ld_delta_a, ld_delta_b, n_boot=n_boot, seed=seed),
        "suites": {},
        "tasks": {},
    }

    for suite in ROBOMME_SUITES:
        idx = [i for i, s in enumerate(suites_k) if s == suite]
        if idx:
            out["suites"][suite] = paired_bootstrap_ci(
                [delta_a[i] for i in idx],
                [delta_b[i] for i in idx],
                n_boot=n_boot,
                seed=seed,
            )

    for task in sorted(set(tasks_k)):
        idx = [i for i, t in enumerate(tasks_k) if t == task]
        out["tasks"][task] = paired_bootstrap_ci(
            [delta_a[i] for i in idx],
            [delta_b[i] for i in idx],
            n_boot=n_boot,
            seed=seed,
        )

    return out


def apply_holm_bonferroni(
    comparisons: dict[str, dict[str, Any]],
    alpha: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Annotate a family of hypothesis test dicts in-place with step-down Holm-Bonferroni `p_holm`."""
    valid_keys = [
        k
        for k, v in comparisons.items()
        if isinstance(v, dict) and isinstance(v.get("overall_success_diff"), dict)
    ]
    m = len(valid_keys)
    if m == 0:
        return comparisons

    # 1. Compute reference pure-bootstrap step-down Holm-Bonferroni (both 1/(B+1) and 2/(B+1) bounds)
    indexed_boot = [
        (
            float(
                comparisons[k]["overall_success_diff"].get(
                    "p_bootstrap_two_sided",
                    comparisons[k]["overall_success_diff"].get("p_value_two_sided", 1.0),
                )
            ),
            idx,
            k,
        )
        for idx, k in enumerate(valid_keys)
    ]
    indexed_boot.sort(key=lambda item: (item[0], item[1]))
    prev_boot = 0.0
    boot_holm_map: dict[str, float] = {}
    boot_ps2_map: dict[str, float] = {}
    for rank_idx, (raw_p, _, k) in enumerate(indexed_boot):
        adj_b = float(min(1.0, max(prev_boot, (m - rank_idx) * raw_p)))
        prev_boot = adj_b
        boot_holm_map[k] = round(adj_b, 6)
        ps2 = float(comparisons[k]["overall_success_diff"].get("p_bootstrap_phipson_smyth_2tail", raw_p))
        boot_ps2_map[k] = round(min(1.0, (m - rank_idx) * ps2), 6)

    # 2. Promote zero-gain/loss discordant pairs ((n10 + n01) >= 8 and min(n10, n01) == 0) to exact
    #    two-sided McNemar as primary p_value_two_sided BEFORE the primary step-down Holm pass,
    #    so p_holm across the entire family is 100% step-down self-consistent.
    for k in valid_keys:
        ov = comparisons[k]["overall_success_diff"]
        n10 = ov.get("discordant_a_only")
        n01 = ov.get("discordant_b_only")
        if n10 is not None and n01 is not None and (int(n10) + int(n01)) >= 8 and (int(n10) == 0 or int(n01) == 0):
            ov["primary_test"] = "exact_mcnemar"
            ov["p_value_two_sided"] = float(ov["p_mcnemar_two_sided_4dp"])
            comparisons[k]["primary_test"] = "exact_mcnemar"
        else:
            ov["primary_test"] = "paired_bootstrap"
            comparisons[k]["primary_test"] = "paired_bootstrap"

    # 3. Primary unified step-down Holm-Bonferroni pass over `p_value_two_sided`
    indexed_primary = [
        (float(comparisons[k]["overall_success_diff"].get("p_value_two_sided", 1.0)), idx, k)
        for idx, k in enumerate(valid_keys)
    ]
    indexed_primary.sort(key=lambda item: (item[0], item[1]))

    prev_adj = 0.0
    for rank_idx, (raw_p, _, k) in enumerate(indexed_primary):
        adj_p = float(min(1.0, max(prev_adj, (m - rank_idx) * raw_p)))
        prev_adj = adj_p
        sig = bool(adj_p < alpha)
        p_rounded = round(adj_p, 6)
        ov = comparisons[k]["overall_success_diff"]

        comparisons[k]["p_holm"] = p_rounded
        comparisons[k]["p_holm_bootstrap"] = boot_holm_map[k]
        comparisons[k]["p_holm_bootstrap_ps2tail"] = boot_ps2_map[k]
        comparisons[k]["holm_significant_005"] = sig
        comparisons[k]["holm_bootstrap_significant_005"] = bool(boot_holm_map[k] < alpha)
        comparisons[k]["holm_family_size"] = m

        ov["p_holm"] = p_rounded
        ov["p_holm_bootstrap"] = boot_holm_map[k]
        ov["p_holm_bootstrap_ps2tail"] = boot_ps2_map[k]
        ov["holm_significant_005"] = sig
        ov["holm_bootstrap_significant_005"] = bool(boot_holm_map[k] < alpha)
        ov["holm_family_size"] = m

        if ov.get("primary_test") == "exact_mcnemar":
            ph_exact = round(min(1.0, (m - rank_idx) * float(ov["p_mcnemar_two_sided"])), 6)
            comparisons[k]["p_holm_mcnemar"] = p_rounded
            comparisons[k]["p_holm_mcnemar_exact"] = ph_exact
            comparisons[k]["holm_mcnemar_significant_005"] = sig
            ov["p_holm_mcnemar"] = p_rounded
            ov["p_holm_mcnemar_exact"] = ph_exact
            ov["holm_mcnemar_significant_005"] = sig

    return comparisons
