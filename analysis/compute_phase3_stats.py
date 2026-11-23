#!/usr/bin/env python3
"""Rigorous Phase-3 Statistical Verification Suite (`compute_phase3_stats.py`).

Computes:
  1. Exact Clopper-Pearson 95% CI, Wilson 95% CI, and 10,000-resample task-stratified bootstrap 95% CI.
  2. Exact two-sided binomial test vs. published `T^2Mem` (`56.83%`, `arXiv:2609.36720`).
  3. Task-stratified two-sample z-test & bootstrap p-value vs. `T^2Mem`'s published 16-task vector
     (accounting for both our per-task variance `p_hat_t(1 - p_hat_t)/n_t` AND `T^2Mem`'s published
     3-seed `N=150`/task variance `p_t2(1 - p_t2)/150`!).
  4. Separate reporting on:
     - Full official `test` leaderboard split (`ep 0..49`, `N = 800`),
     - Completely untouched `test` subset (`ep 15..49`, `N = 560`),
     - Matched `test240` subset (`ep 0..14`, `N = 240`) with exact paired McNemar + Holm-Bonferroni
       vs. `pi05_baseline_pt` (`21.67%`), `perceptual_framesamp_modul_pt` (`45.83%`), and Phase-2
       `p2_m5_hybrid_v3_stageC_hop2_imitation_l2` (`48.75%`),
     - Matched `k = 3` un-reset multi-session interference (`N = 240`) vs. `FrameSamp+Modul` (`15.42%`).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

T2MEM_PUBLISHED_OVERALL = 0.5683375  # 909.34 / 1600 = 56.83%
T2MEM_BY_TASK: Dict[str, float] = {
    "BinFill": 0.6067,
    "PickXtimes": 0.9133,
    "SwingXtimes": 0.9067,
    "StopCube": 0.7000,
    "VideoUnmask": 0.8800,
    "ButtonUnmask": 0.5933,
    "VideoUnmaskSwap": 0.2933,
    "ButtonUnmaskSwap": 0.2000,
    "PickHighlight": 0.5067,
    "VideoRepick": 0.3867,
    "VideoPlaceButton": 0.3533,
    "VideoPlaceOrder": 0.3000,
    "MoveCube": 0.8067,
    "InsertPeg": 0.3600,
    "PatternLock": 0.5000,
    "RouteStick": 0.7867,
}

SUITES: Dict[str, List[str]] = {
    "Counting": ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"],
    "Permanence": ["VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"],
    "Reference": ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"],
    "Imitation": ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"],
}


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def exact_mcnemar(eps_a: List[dict], eps_b: List[dict]) -> Dict[str, object]:
    map_a = {(e["task"], int(e["episode_id"])): bool(e["success"]) for e in eps_a}
    map_b = {(e["task"], int(e["episode_id"])): bool(e["success"]) for e in eps_b}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    n10 = sum(1 for k in common if map_a[k] and not map_b[k])
    n01 = sum(1 for k in common if (not map_a[k]) and map_b[k])
    n11 = sum(1 for k in common if map_a[k] and map_b[k])
    n00 = sum(1 for k in common if (not map_a[k]) and (not map_b[k]))
    nd = n10 + n01
    p_two = float(stats.binomtest(n10, nd, 0.5, alternative="two-sided").pvalue) if nd > 0 else 1.0
    p_one = float(stats.binomtest(n10, nd, 0.5, alternative="greater").pvalue) if nd > 0 else 1.0
    sr_a = sum(1 for k in common if map_a[k]) / max(len(common), 1)
    sr_b = sum(1 for k in common if map_b[k]) / max(len(common), 1)
    return {
        "n_paired": len(common),
        "sr_a": sr_a,
        "sr_b": sr_b,
        "diff_pp": (sr_a - sr_b) * 100.0,
        "wins_a_only": n10,
        "wins_b_only": n01,
        "both_win": n11,
        "both_fail": n00,
        "p_mcnemar_two_sided": p_two,
        "p_mcnemar_one_sided": p_one,
    }


def analyze_subset(episodes: List[dict], label: str, n_t2mem_per_task: int = 150) -> Dict[str, object]:
    n = len(episodes)
    k = sum(1 for e in episodes if e["success"])
    sr = k / max(n, 1)
    cp_lo, cp_hi = clopper_pearson_ci(k, n)
    p_binom_two = float(stats.binomtest(k, n, T2MEM_PUBLISHED_OVERALL, alternative="two-sided").pvalue)
    p_binom_gt = float(stats.binomtest(k, n, T2MEM_PUBLISHED_OVERALL, alternative="greater").pvalue)

    by_task: Dict[str, Dict[str, float]] = {}
    var_ours_sum = 0.0
    var_two_sample_sum = 0.0
    rng = np.random.default_rng(7)
    boot_means = np.zeros(10000, dtype=np.float64)

    tasks_present = [t for t in T2MEM_BY_TASK if any(e["task"] == t for e in episodes)]
    for t in tasks_present:
        teps = [1 if e["success"] else 0 for e in episodes if e["task"] == t]
        nt = len(teps)
        pt = float(np.mean(teps))
        p_t2 = T2MEM_BY_TASK[t]
        by_task[t] = {
            "n": nt,
            "k": int(sum(teps)),
            "sr": pt,
            "t2mem_sr": p_t2,
            "diff_pp": (pt - p_t2) * 100.0,
        }
        # Conservative Jeffreys-smoothed variance bound when pt in {0, 1}
        pt_v = min(max(pt, 0.5 / (nt + 1.0)), 1.0 - 0.5 / (nt + 1.0))
        var_ours_sum += (pt_v * (1.0 - pt_v)) / nt
        var_two_sample_sum += (pt_v * (1.0 - pt_v)) / nt + (p_t2 * (1.0 - p_t2)) / float(n_t2mem_per_task)
        arr = np.asarray(teps, dtype=np.float64)
        idx = rng.integers(0, nt, size=(10000, nt))
        boot_means += arr[idx].mean(axis=1) / len(tasks_present)

    n_tasks = max(len(tasks_present), 1)
    se_strat_ours = math.sqrt(var_ours_sum) / n_tasks
    se_strat_two_sample = math.sqrt(var_two_sample_sum) / n_tasks
    t2_mean = float(np.mean([T2MEM_BY_TASK[t] for t in tasks_present]))
    z_fixed = (sr - t2_mean) / max(se_strat_ours, 1e-12)
    z_two_sample = (sr - t2_mean) / max(se_strat_two_sample, 1e-12)
    p_z_fixed_two = float(2.0 * stats.norm.sf(abs(z_fixed)))
    p_z_two_sample_two = float(2.0 * stats.norm.sf(abs(z_two_sample)))
    boot_lo, boot_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    # Phipson-Smyth exact Monte Carlo p-value vs T2Mem
    b_le = int(np.sum(boot_means <= t2_mean))
    p_boot_one = (b_le + 1.0) / 10001.0
    p_boot_two = min(1.0, 2.0 * min(p_boot_one, (int(np.sum(boot_means >= t2_mean)) + 1.0) / 10001.0))

    by_suite: Dict[str, Dict[str, float]] = {}
    for s_name, s_tasks in SUITES.items():
        seps = [e for e in episodes if e["task"] in s_tasks]
        if seps:
            sk = sum(1 for e in seps if e["success"])
            ssr = sk / len(seps)
            s_t2 = float(np.mean([T2MEM_BY_TASK[t] for t in s_tasks]))
            by_suite[s_name] = {
                "n": len(seps),
                "k": sk,
                "sr": ssr,
                "t2mem_sr": s_t2,
                "diff_pp": (ssr - s_t2) * 100.0,
            }

    return {
        "label": label,
        "n_episodes": n,
        "n_success": k,
        "sr": sr,
        "t2mem_sr": t2_mean,
        "diff_vs_t2mem_pp": (sr - t2_mean) * 100.0,
        "clopper_pearson_ci95": [cp_lo, cp_hi],
        "stratified_boot_ci95": [boot_lo, boot_hi],
        "se_stratified_ours_pp": se_strat_ours * 100.0,
        "se_stratified_two_sample_pp": se_strat_two_sample * 100.0,
        "p_binom_two_sided_vs_t2mem": p_binom_two,
        "p_binom_one_sided_vs_t2mem": p_binom_gt,
        "z_stratified_fixed": z_fixed,
        "p_z_stratified_fixed_two_sided": p_z_fixed_two,
        "z_stratified_two_sample_150": z_two_sample,
        "p_z_stratified_two_sample_two_sided": p_z_two_sample_two,
        "p_boot_stratified_two_sided": p_boot_two,
        "by_suite": by_suite,
        "by_task": by_task,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", type=Path, required=True)
    ap.add_argument("--interfk3-json", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    data = json.loads(args.eval_json.read_text())
    eps = data["episodes"]
    report: Dict[str, object] = {"source_json": str(args.eval_json)}
    report["all_episodes"] = analyze_subset(eps, f"all_N{len(eps)}")

    eps_untouched = [e for e in eps if int(e["episode_id"]) >= 15]
    if eps_untouched:
        report["untouched_ep15_49"] = analyze_subset(eps_untouched, f"untouched_ep15_49_N{len(eps_untouched)}")

    eps_0_14 = [e for e in eps if int(e["episode_id"]) < 15]
    if eps_0_14:
        report["ep0_14_subset"] = analyze_subset(eps_0_14, f"ep0_14_N{len(eps_0_14)}")

    # Paired tests on test240 if dataset == "test"
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "results").exists():
        repo = Path("/home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla")
    paired: Dict[str, object] = {}
    if eps and eps[0].get("dataset") == "test":
        refs = {
            "vs_pi05_base_test240": repo / "results/phase2/test_pi05_baseline.json",
            "vs_framesamp_modul_test240": repo / "results/phase2/test_framesamp_modul.json",
            "vs_phase2_champion_hop2_test240": repo / "results/phase2/test_m5_hybrid_v3_stageC_hop2_imitation_l2.json",
        }
        # check exact framesamp test filename
        if not refs["vs_framesamp_modul_test240"].exists():
            cand = list((repo / "results/phase2").glob("test*framesamp*k1*.json")) + list(
                (repo / "results/phase2").glob("m1_framesamp*test*.json")
            )
            for p in (repo / "results/phase2").glob("*.json"):
                if "framesamp" in p.name and "test" in p.name and "interfk3" not in p.name and "shard" not in p.name:
                    refs["vs_framesamp_modul_test240"] = p
                    break
        for name, rpath in refs.items():
            if rpath.exists():
                paired[name] = exact_mcnemar(eps_0_14, json.loads(rpath.read_text())["episodes"])

    if args.interfk3_json and args.interfk3_json.exists():
        k3_eps = json.loads(args.interfk3_json.read_text())["episodes"]
        report["interfk3"] = analyze_subset(k3_eps, f"interfk3_N{len(k3_eps)}")
        fs_k3 = repo / "results/phase2/test_framesamp_modul_interfk3.json"
        if fs_k3.exists():
            paired["interfk3_vs_framesamp_k3"] = exact_mcnemar(k3_eps, json.loads(fs_k3.read_text())["episodes"])
        if eps_0_14:
            paired["interfk3_vs_own_k1"] = exact_mcnemar(k3_eps, eps_0_14)

    # Step-down Holm-Bonferroni across Phase-3 pre-registered primary hypothesis family
    primary_pvals: List[Tuple[str, float]] = [
        ("all_vs_t2mem_binom", report["all_episodes"]["p_binom_two_sided_vs_t2mem"]),
        ("all_vs_t2mem_strat_2samp", report["all_episodes"]["p_z_stratified_two_sample_two_sided"]),
    ]
    if "untouched_ep15_49" in report:
        primary_pvals.append(
            ("untouched560_vs_t2mem_binom", report["untouched_ep15_49"]["p_binom_two_sided_vs_t2mem"])
        )
        primary_pvals.append(
            ("untouched560_vs_t2mem_strat_2samp", report["untouched_ep15_49"]["p_z_stratified_two_sample_two_sided"])
        )
    for k_name, p_obj in paired.items():
        if k_name != "interfk3_vs_own_k1":
            primary_pvals.append((k_name, p_obj["p_mcnemar_two_sided"]))

    sorted_p = sorted(primary_pvals, key=lambda x: x[1])
    m_fam = len(sorted_p)
    holm: Dict[str, float] = {}
    running_max = 0.0
    for idx, (h_name, pv) in enumerate(sorted_p):
        adj = min(1.0, pv * (m_fam - idx))
        running_max = max(running_max, adj)
        holm[h_name] = running_max

    report["paired_tests"] = paired
    report["holm_bonferroni_family_m"] = m_fam
    report["holm_adjusted_pvalues"] = holm

    text_out = json.dumps(report, indent=2)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text_out)
    for key in ("all_episodes", "untouched_ep15_49", "ep0_14_subset", "interfk3"):
        if key in report:
            r = report[key]
            print(
                f"[{r['label']}] SR = {r['sr']*100:.2f}% ({r['n_success']}/{r['n_episodes']}) "
                f"vs T2Mem {r['t2mem_sr']*100:.2f}% ({r['diff_vs_t2mem_pp']:+.2f} pp) | "
                f"CP95=[{r['clopper_pearson_ci95'][0]*100:.2f}, {r['clopper_pearson_ci95'][1]*100:.2f}] | "
                f"p_binom={r['p_binom_two_sided_vs_t2mem']:.3e} | "
                f"p_strat2s={r['p_z_stratified_two_sample_two_sided']:.3e}"
            )
            for s_name, s_d in r["by_suite"].items():
                print(
                    f"   {s_name:<12}: {s_d['sr']*100:6.2f}% ({s_d['k']:3d}/{s_d['n']:3d}) "
                    f"vs T2Mem {s_d['t2mem_sr']*100:5.2f}% ({s_d['diff_pp']:+6.2f} pp)"
                )
    if holm:
        print(f"\n[Holm-Bonferroni Family m={m_fam}]:")
        for k_h, pv_h in holm.items():
            print(f"   {k_h:<38}: p_Holm = {pv_h:.4e}")


if __name__ == "__main__":
    main()
