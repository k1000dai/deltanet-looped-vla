#!/usr/bin/env python3
"""Phase-2 Publication Figure & Statistical Analysis Suite (`DeltaMem-π0.5` on 16-Task RoboMME).

Strictly CPU-only (`CUDA_VISIBLE_DEVICES=""`). Reads `results/phase2/*.json` (read-only!) and
renders 300-DPI PNG + vector PDF (`pdf.fonttype = 42`) figures into `--out-dir` (default:
`docs/figures_phase2`), plus optional JSON statistical summary (`--stats-out`):
  1. `fig_p2_1_leaderboard_and_suites.{png,pdf}`
  2. `fig_p2_2_per_task_heatmap.{png,pdf}`
  3. `fig_p2_3_efficiency_pareto_and_state.{png,pdf}`
  4. `fig_p2_4_interference_and_parity.{png,pdf}`

Automatically discovers M1 parity JSONs (`m1_*.json`), M3/M4 validation runs (`val*.json`,
`*stage*.json`, `*deltamem*.json`), M5 official `test` runs (`test_*.json`), and live `k=3`
un-reset cross-session interference controls (`*interfk*.json`), computing exact two-sided
McNemar p-values, Phipson-Smyth finite-sample paired bootstrap bounds (`2/(B+1)` floor, zero
`0.0` p-values), and step-down Holm-Bonferroni multiplicity-adjusted `p_holm`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "results" / "phase2"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "figures_phase2"

SUITES: Tuple[str, ...] = ("Counting", "Permanence", "Reference", "Imitation")
TASKS_ORDERED: Tuple[str, ...] = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "ButtonUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)
TASK_SHORT: Dict[str, str] = {
    "BinFill": "BinFill",
    "PickXtimes": "PickX",
    "SwingXtimes": "SwingX",
    "StopCube": "StopCube",
    "VideoUnmask": "VidUmsk",
    "ButtonUnmask": "BtnUmsk",
    "VideoUnmaskSwap": "VidSwap",
    "ButtonUnmaskSwap": "BtnSwap",
    "PickHighlight": "PickHL",
    "VideoRepick": "VidRepk",
    "VideoPlaceButton": "VidPlcBtn",
    "VideoPlaceOrder": "VidPlcOrd",
    "MoveCube": "MoveCube",
    "InsertPeg": "InsPeg",
    "PatternLock": "PatLock",
    "RouteStick": "RteStick",
}

# Primary-source verified official 16-task RoboMME `test` numbers (`docs/SURVEY_PHASE2.md` §1.1)
PUBLISHED_BASELINES: Dict[str, Dict[str, Any]] = {
    "pi05_pub": {
        "label": "π0.5 Base (Pub.)",
        "overall": 17.93,
        "suites": {"Counting": 28.78, "Permanence": 17.00, "Reference": 17.17, "Imitation": 8.78},
        "tasks": [
            30.00, 42.89, 35.56, 6.67,
            20.44, 22.22, 18.67, 6.67,
            11.33, 0.44, 31.11, 25.78,
            26.00, 1.56, 2.89, 4.67,
        ],
        "state_mib": 0.001,
        "latency_ms": 184.4,
        "color": "#7F7F7F",
    },
    "rmt_modul_pub": {
        "label": "RMT+Modul (Pub.)",
        "overall": 20.17,
        "suites": {"Counting": 34.14, "Permanence": 15.78, "Reference": 21.11, "Imitation": 9.67},
        "tasks": [
            33.33, 60.78, 37.78, 4.67,
            31.11, 11.78, 17.78, 2.44,
            17.11, 4.22, 32.00, 31.11,
            24.67, 2.21, 3.78, 8.00,
        ],
        "state_mib": 1.00,
        "latency_ms": 225.0,
        "color": "#A6761D",
    },
    "ttt_expert_pub": {
        "label": "TTT+Expert (Pub.)",
        "overall": 22.35,
        "suites": {"Counting": 36.00, "Permanence": 22.95, "Reference": 19.67, "Imitation": 10.78},
        "tasks": [
            34.89, 63.78, 41.33, 4.00,
            31.78, 22.44, 19.56, 18.00,
            12.22, 9.56, 34.00, 22.89,
            33.56, 0.89, 3.11, 5.56,
        ],
        "state_mib": 0.51,
        "latency_ms": 240.0,
        "color": "#8C6BB1",
    },
    "groundsg_qwen_pub": {
        "label": "GroundSG+QwenVL",
        "overall": 32.70,
        "suites": {"Counting": 38.00, "Permanence": 39.34, "Reference": 31.56, "Imitation": 21.89},
        "tasks": [
            52.00, 92.67, 7.33, 0.00,
            88.67, 24.00, 30.67, 14.00,
            15.11, 25.33, 54.00, 31.78,
            71.56, 3.33, 6.67, 6.00,
        ],
        "state_mib": 0.25,
        "latency_ms": 1850.0,
        "color": "#66A61E",
    },
    "memer_pub": {
        "label": "MemER (Qwen3-4B)",
        "overall": 42.38,
        "suites": {"Counting": 48.83, "Permanence": 53.17, "Reference": 38.00, "Imitation": 29.50},
        "tasks": [
            56.67, 79.33, 59.33, 0.00,
            81.33, 72.00, 38.00, 21.33,
            70.67, 25.33, 30.00, 26.00,
            82.67, 6.67, 16.67, 12.00,
        ],
        "state_mib": 6.00,
        "latency_ms": 10135.6,
        "color": "#999999",
    },
    "framesamp_pub": {
        "label": "FrameSamp+Modul (Pub.)",
        "overall": 44.51,
        "suites": {"Counting": 65.22, "Permanence": 25.11, "Reference": 36.33, "Imitation": 51.39},
        "tasks": [
            39.56, 87.33, 92.00, 42.00,
            32.67, 25.11, 24.44, 18.22,
            22.89, 30.44, 60.00, 32.00,
            77.78, 7.56, 53.56, 66.67,
        ],
        "state_mib": 4.00,
        "latency_ms": 440.0,
        "color": "#D55E00",
    },
    "dual_stream_pub": {
        "label": "Dual-Stream (Mask-Sup.*)",
        "overall": 49.22,
        "suites": {"Counting": 70.83, "Permanence": 27.75, "Reference": 45.31, "Imitation": 53.00},
        "tasks": [
            42.00, 93.33, 92.67, 55.33,
            35.00, 28.00, 23.00, 25.00,
            32.67, 35.00, 74.00, 39.56,
            83.33, 8.00, 52.00, 68.67,
        ],
        "state_mib": 3.20,
        "latency_ms": 380.0,
        "color": "#56B4E9",
    },
    "t2mem_pub": {
        "label": "T²Mem (36.4 MiB, Pub.)",
        "overall": 56.83,
        "suites": {"Counting": 78.17, "Permanence": 49.17, "Reference": 38.67, "Imitation": 61.34},
        "tasks": [
            60.67, 91.33, 90.67, 70.00,
            88.00, 59.33, 29.33, 20.00,
            50.67, 38.67, 35.33, 30.00,
            80.67, 36.00, 50.00, 78.67,
        ],
        "state_mib": 36.35,
        "latency_ms": 146.6,
        "color": "#E69F00",
    },
    "oracle_pub": {
        "label": "GroundSG+Oracle*",
        "overall": 84.08,
        "suites": {"Counting": 83.86, "Permanence": 93.31, "Reference": 95.17, "Imitation": 63.98},
        "tasks": [
            85.78, 100.0, 100.0, 49.67,
            98.78, 95.00, 99.22, 80.22,
            83.33, 97.33, 100.0, 100.0,
            87.78, 15.56, 97.00, 55.56,
        ],
        "state_mib": 0.05,
        "latency_ms": 185.0,
        "color": "#4D4D4D",
    },
}

INTERFERENCE_PUBLISHED: Dict[str, List[float]] = {
    "FrameSamp-Modul (2606.22338)": [45.3, 38.4, 30.0, 19.3],
    "TokenDrop-Modul (2606.22338)": [35.3, 31.3, 24.2, 20.7],
    "GroundSG+QwenVL (2606.22338)": [34.2, 32.0, 26.4, 26.0],
    "MemER (2606.22338)": [32.2, 27.8, 22.9, 21.3],
    "Recurrent-TTT-Expert (2606.22338)": [18.0, 16.4, 14.9, 16.4],
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 7.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _save_both(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def exact_mcnemar_two_sided(b_only: int, c_only: int) -> float:
    """Exact two-sided binomial McNemar p-value on discordant pairs `(n = b + c)`."""
    n = b_only + c_only
    if n == 0:
        return 1.0
    k = min(b_only, c_only)
    log2_n = n * math.log(2.0)
    cum = 0.0
    for i in range(k + 1):
        log_c = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        cum += math.exp(log_c - log2_n)
    return min(1.0, 2.0 * cum)


def paired_test_episodes(
    run_a: Dict[str, Any],
    run_b: Dict[str, Any],
    n_boot: int = 9999,
    seed: int = 20261116,
) -> Optional[Dict[str, Any]]:
    """Compute matched-episode paired bootstrap (`2/(B+1)` floor) & exact McNemar test."""
    eps_a = {
        (ep["task"], ep.get("dataset", ""), int(ep["episode_id"])): 1.0 if ep["success"] else 0.0
        for ep in run_a.get("episodes", [])
    }
    eps_b = {
        (ep["task"], ep.get("dataset", ""), int(ep["episode_id"])): 1.0 if ep["success"] else 0.0
        for ep in run_b.get("episodes", [])
    }
    common = sorted(set(eps_a.keys()) & set(eps_b.keys()))
    if len(common) < 16:
        return None
    va = np.array([eps_a[k] for k in common], dtype=np.float64)
    vb = np.array([eps_b[k] for k in common], dtype=np.float64)
    diff = va - vb
    n = len(common)
    a_only = int(np.sum((va == 1.0) & (vb == 0.0)))
    b_only = int(np.sum((va == 0.0) & (vb == 1.0)))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975]).tolist()
    r_le0 = int(np.sum(boot_means <= 0.0))
    r_ge0 = int(np.sum(boot_means >= 0.0))
    p_boot_ps2 = min(1.0, 2.0 * (min(r_le0, r_ge0) + 1.0) / (n_boot + 1.0))
    p_mcnemar = exact_mcnemar_two_sided(a_only, b_only)
    zero_discordant_cell = (a_only == 0 or b_only == 0) and (a_only + b_only > 0)
    primary_test = "exact_mcnemar" if zero_discordant_cell else "paired_bootstrap_ps2"
    primary_p = p_mcnemar if zero_discordant_cell else p_boot_ps2
    return {
        "n_pairs": n,
        "mean_a": float(va.mean()),
        "mean_b": float(vb.mean()),
        "mean_diff": float(diff.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "discordant_a_only": a_only,
        "discordant_b_only": b_only,
        "primary_test": primary_test,
        "p_value_two_sided": float(primary_p),
        "p_mcnemar_two_sided": float(p_mcnemar),
        "p_bootstrap_ps2_two_sided": float(p_boot_ps2),
    }


def apply_holm_bonferroni(tests: Dict[str, Dict[str, Any]]) -> None:
    """In-place step-down Holm-Bonferroni multiplicity adjustment across all hypothesis tests."""
    ordered = sorted(tests.items(), key=lambda kv: kv[1]["p_value_two_sided"])
    m = len(ordered)
    running_max = 0.0
    for rank_idx, (_, entry) in enumerate(ordered):
        adj = min(1.0, entry["p_value_two_sided"] * (m - rank_idx))
        running_max = max(running_max, adj)
        entry["p_holm"] = float(running_max)
        entry["holm_significant_005"] = bool(running_max < 0.05)


def is_complete_or_full_suite_run(data: Dict[str, Any], min_episodes: int = 32) -> bool:
    """Filter out mid-run or preempted checkpoint evaluations that have not completed all configured tasks."""
    ov = data.get("summary", {}).get("overall", {})
    n_done = int(ov.get("num_episodes", 0))
    cfg = data.get("config", {})
    cfg_tasks = cfg.get("tasks", TASKS_ORDERED)
    n_tasks = len(cfg_tasks)
    ep_per_task = cfg.get("episodes_per_task")
    if ep_per_task is not None:
        expected = int(ep_per_task) * n_tasks
        if n_done < max(min_episodes, expected):
            return False
    summary_tasks = data.get("summary", {}).get("tasks", {})
    if summary_tasks and n_tasks > 0:
        counts = [int(summary_tasks.get(t, {}).get("num_episodes", 0)) for t in cfg_tasks]
        if min(counts) == 0 or min(counts) != max(counts):
            return False
    return n_done >= min_episodes


def load_phase2_runs(eval_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load completed closed-loop evaluation JSONs in `results/phase2/` (read-only)."""
    runs: Dict[str, Dict[str, Any]] = {}
    parity_summary: Optional[Dict[str, Any]] = None
    if not eval_dir.is_dir():
        return runs, parity_summary
    for p in sorted(eval_dir.glob("*.json")):
        if ".shard" in p.name:
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if p.name == "m1_parity_summary.json":
            parity_summary = data
            continue
        if isinstance(data, dict) and "summary" in data and "overall" in data["summary"]:
            if is_complete_or_full_suite_run(data):
                runs[p.stem] = data
    runs = dict(
        sorted(
            runs.items(),
            key=lambda kv: (
                kv[1].get("config", {}).get("dataset", "val") == "test",
                re.sub(r"^unified_", "", kv[0]),
            ),
        )
    )
    return runs, parity_summary


def _extract_run_modifiers(stem: str, data: Dict[str, Any]) -> str:
    """Extract concise version (`v2`), step (`s2.5k`), and `Mem-CFG` (`CFG1.35`) modifiers when present."""
    s_low = stem.lower()
    mods: List[str] = []
    s_ver_clean = re.sub(
        r"selective_v\d+[a-z]*|sel_v\d+[a-z]*|v\d+[a-z]*stage|v\d+[a-z]*ptr|m9v\d+[a-z]*|m9_v\d+[a-z]*|m9(?:sym|hyb)v\d+[a-z]*|m10v\d+[a-z]*|m11v\d+[a-z]*|m7[de]_?v\d+[a-z]*",
        "selective",
        s_low,
    )
    m_ver = re.search(r"(?:_|-|b\d|a\d|c|t\d)v(\d+)(?:_|$)", s_ver_clean)
    if m_ver:
        mods.append(f"v{m_ver.group(1)}")
    m_step = re.search(r"_s(\d{3,6})(?:_|$)", s_low)
    if m_step:
        st_val = int(m_step.group(1))
        mods.append(f"s{st_val/1000:.1f}k".replace(".0k", "k") if st_val >= 1000 else f"s{st_val}")
    m_tok = re.search(r"_(\d{2,4}t)(?:_|$)", s_low)
    if m_tok:
        mods.append(m_tok.group(1))
    if "anc1" in s_low or "a100" in s_low:
        mods.append("anc1")
    elif "a9375" in s_low:
        mods.append("anc.94")
    elif "a0875" in s_low or "anc0875" in s_low:
        mods.append("anc.88")
    if "noshock" in s_low:
        mods.append("NoShock")
    if re.search(r"(?:^|_)(?:online|onl|on100)(?:_|$)", s_low):
        mods.append("Onl")
    if "latch" in s_low:
        mods.append("Latch")
    if "6task" in s_low or "6_task" in s_low:
        mods.append("6Task")
    elif "ordinal_only" in s_low or "ordonly" in s_low or "ord_only" in s_low:
        mods.append("OrdOnly")
    elif "selective_v2" in s_low or "sel_v2" in s_low:
        mods.append("Sel-v2")
    elif "selective" in s_low or "_sel_" in s_low or s_low.endswith("_sel"):
        mods.append("Sel")
    elif "conf65" in s_low:
        mods.append("Conf.65")
    elif "conf50" in s_low:
        mods.append("Conf.50")
    m_m12 = re.search(r"m12(v\d+[a-z]*)", s_low)
    if m_m12:
        mods.append(f"M12{m_m12.group(1)}")
    elif "m12" in s_low or "reactive" in s_low:
        mods.append("M12")
    m_m11 = re.search(r"m11(v\d+[a-z]*)", s_low)
    if m_m11:
        mods.append(f"M11{m_m11.group(1)}")
    elif "m11" in s_low or "continuous_vlm" in s_low or "cont_vlm" in s_low:
        mods.append("M11")
    if "hr16" in s_low:
        mods.append("HR16")
    elif "pyr" in s_low:
        mods.append("Pyr")
    elif re.search(r"m9v\d+adv", s_low):
        mods.append("Adv")
    m_m10 = re.search(r"m10(v\d+[a-z]*)", s_low)
    if m_m10:
        mods.append(f"M10{m_m10.group(1)}")
    elif "m10" in s_low or "specialist" in s_low:
        mods.append("M10")
    if "decoupled" in s_low or "decoup" in s_low or "v1stage" in s_low:
        mods.append("Decoup")
    if "cross" in s_low or "xbb" in s_low:
        mods.append("Cross")
    if "fixed252" in s_low or "fix252" in s_low or "fix381" in s_low:
        mods.append("Fix252")
    m_m7ev = re.search(r"m7e_?(v\d+[a-z]*)(?:_|$)", s_low)
    m_m7dv = re.search(r"m7d_?(v2[a-z]*)(?:_|$)", s_low)
    if "m9" in s_low or "dgvlm" in s_low:
        if m_m7ev:
            mods.append(f"M7E{m_m7ev.group(1)}")
        elif "m7e" in s_low:
            mods.append("M7E")
        elif m_m7dv:
            mods.append(f"M7D{m_m7dv.group(1)}")
        elif "m7d" in s_low:
            mods.append("M7D")
        elif "m7c" in s_low:
            mods.append("M7C")
        elif "m7b" in s_low:
            mods.append("M7B")
        m_m9v = re.search(r"m9(?:_?(?:sym|hyb)_?)?(v\d+[a-z]?)(?:_|hr|pyr|adv|$)", s_low)
        if m_m9v:
            if "sym" in s_low:
                mods.append(f"M9{m_m9v.group(1)}-Sym")
            elif "hyb" in s_low:
                mods.append(f"M9{m_m9v.group(1)}-Hyb")
            else:
                mods.append(f"M9{m_m9v.group(1)}")
        elif "sym" in s_low:
            mods.append("M9-Sym")
        elif "hyb" in s_low and not ("hybrid" in s_low and "m9_hyb" not in s_low and "m9hyb" not in s_low and "m9v" not in s_low):
            mods.append("M9-Hyb")
        else:
            mods.append("M9-DGVLM")
    elif "m7a" in s_low:
        mods.append("M7A")
    elif "m7b" in s_low:
        mods.append("M7B")
    elif "m7c" in s_low:
        mods.append("M7C")
    elif m_m7ev:
        mods.append(f"M7E{m_m7ev.group(1)}")
    elif "m7e" in s_low:
        mods.append("M7E")
    elif m_m7dv:
        mods.append(f"M7D{m_m7dv.group(1)}")
    elif "m7d" in s_low:
        mods.append("M7D")
    elif "m8a" in s_low:
        mods.append("M8A-256t")
    elif "m8b" in s_low:
        mods.append("M8B-64t")
    elif "m8" in s_low or "vlmprefix" in s_low:
        mods.append("M8-VLM")
    if "peft" in s_low:
        mods.append("PEFT")
    if "wiseft" in s_low:
        mods.append("WiSE-FT")
    if "midpoint" in s_low or "heun" in s_low or "_mid_" in s_low:
        mods.append("Midpt")
    if "antithetic" in s_low or "_anti_" in s_low:
        mods.append("Anti")
    if "_lm" in s_low:
        mods.append("LM")
    m_hop = re.search(r"hop(\d+)", s_low)
    if m_hop:
        mods.append(f"Hop{m_hop.group(1)}")
    if "_auto_" in s_low:
        mods.append("Auto")
    m_loop = re.search(r"_(?:loop|l)(\d+)(?:_|$)", s_low)
    if "imitation_l2" in s_low or "imit_l2" in s_low:
        mods.append("ImitL2")
    elif m_loop:
        mods.append(f"L={m_loop.group(1)}")
    m_bok = re.search(r"(?:bok|medoid|bestof)(\d+)", s_low)
    if m_bok:
        mods.append(f"BoK{m_bok.group(1)}")
    elif "medoid" in s_low:
        mods.append("Medoid")
    m_nt = re.search(r"(?:nt|noisetemp)(\d{2,3})", s_low)
    if m_nt:
        nt_v = float(m_nt.group(1)) / (100.0 if len(m_nt.group(1)) >= 3 else 10.0)
        mods.append(f"NT{nt_v:.2f}".rstrip("0").rstrip("."))
    hw_cfg = data.get("hardware_telemetry", {}).get("mem_cfg_scale")
    m_cfg = re.search(r"(?:mem)?cfg(\d{2,4})", s_low)
    if hw_cfg is not None and abs(float(hw_cfg) - 1.0) > 1e-3:
        c_str = f"{float(hw_cfg):.2f}".rstrip("0").rstrip(".")
        mods.append(f"CFG{c_str}")
    elif m_cfg:
        raw_c = m_cfg.group(1)
        val_c = float(raw_c) / (100.0 if len(raw_c) >= 3 else 10.0)
        c_str = f"{val_c:.2f}".rstrip("0").rstrip(".")
        mods.append(f"CFG{c_str}")
    return (" " + " ".join(mods)) if mods else ""


def _classify_run_color_and_label(stem: str, data: Dict[str, Any]) -> Tuple[str, str, float]:
    """Return `(readable_label, hex_color, state_mib_estimate)` for a discovered run stem."""
    cfg = data.get("config", {})
    split = cfg.get("dataset", "val")
    n_ep = int(data["summary"]["overall"].get("num_episodes", 0))
    interf_k = int(cfg.get("interference_k", 0))
    k_tag = f", k={interf_k}" if interf_k > 1 else ""
    mods = _extract_run_modifiers(stem, data)
    s_lower = stem.lower()
    stg_short = ""
    for tag in ("stagec", "stageb2", "stageb1", "stagea1", "stagea0", "stageb", "stagea"):
        if tag in s_lower:
            stg_short = f" {tag.replace('stage', 'STG-').upper()}"
            break
    if "m12" in s_lower or ("reactive" in s_lower and "m11" not in s_lower):
        if "m11" in s_lower or "continuous_vlm" in s_lower or "cont_vlm" in s_lower:
            return f"Continuous-VLM-Hybrid*{stg_short}{mods} ({split} N={n_ep}{k_tag})", "#003C30", 3.08
        return f"Reactive-Specialist-π0.5*{stg_short}{mods} ({split} N={n_ep}{k_tag})", "#003C30", 3.08
    if "m11" in s_lower or "continuous_vlm" in s_lower or "cont_vlm" in s_lower:
        return f"Continuous-VLM-Hybrid*{stg_short}{mods} ({split} N={n_ep}{k_tag})", "#003C30", 3.08
    if "m10" in s_lower or "specialist" in s_lower:
        return f"Grounded-Specialist-π0.5*{stg_short}{mods} ({split} N={n_ep}{k_tag})", "#003C30", 2.94
    if any(k in s_lower for k in ("dgvlm", "p2_m9", "m9_", "m9v", "hr16", "pyr", "m7c", "m7d", "m7e")):
        if any(h in s_lower for h in ("m7b", "m7c", "m7d", "m7e", "hyb", "plus")):
            return f"Grounded-Hybrid-VLM*{stg_short}{mods} ({split} N={n_ep}{k_tag})", "#00441B", 2.92
        return f"DeltaGround-VLM*{stg_short}{mods} ({split} N={n_ep}{k_tag})", "#006D2C", 0.55
    if any(k in s_lower for k in ("vlmprefix", "p2_m8", "m8_", "m8a", "m8b")):
        m8_mib = 2.64 if ("m8b" in s_lower or "64t" in s_lower) else 3.39
        return f"VLM-Surprise-Hybrid*{stg_short}{mods} ({split} N={n_ep}{k_tag})", "#238B45", m8_mib
    if any(k in s_lower for k in ("grounded_hybrid", "p2_m7", "m7_", "m7a", "m7b", "m7c", "m7d", "m7e")):
        m7_col = "#005A32" if "m7a" in s_lower else "#00441B"
        return f"Grounded-Hybrid-π0.5*{stg_short}{mods} ({split} N={n_ep}{k_tag})", m7_col, 2.84
    if "deltaground" in s_lower or "p2_m6" in s_lower:
        return f"DeltaGround-π0.5*{stg_short}{mods} ({split} N={n_ep}{k_tag})", "#1B7837", 0.45
    if "symbolic_grounded" in s_lower or "groundsg" in s_lower:
        return f"GroundSG PT{mods} ({split} N={n_ep}{k_tag})", "#4D4D4D", 0.05
    if "hybrid" in s_lower or "p2_m5" in s_lower or "wiseft" in s_lower:
        hyb_mib = (
            1.39
            if "256t" in s_lower
            else (2.39 if any(x in s_lower for x in ("512t", "v3", "v4", "v5", "wiseft")) else 1.39)
        )
        col_h = "#542788" if any(x in s_lower for x in ("peft", "v4", "v5", "wiseft")) else "#7A0177"
        return f"DeltaMem-Hybrid*{stg_short}{mods} ({split} N={n_ep}{k_tag})", col_h, hyb_mib
    if "aligned" in s_lower or "p2_m4" in s_lower:
        return f"DeltaMem-Aligned*{stg_short}{mods} ({split} N={n_ep}{k_tag})", "#02818A", 0.89
    if "deltamem_loop" in s_lower or "p2_m3" in s_lower or "stagec" in s_lower:
        return f"DeltaMem-Loop{mods} ({split} N={n_ep}{k_tag})", "#C51B7D", 0.89
    if "deltamem" in s_lower or "p2_m2" in s_lower:
        stage = ""
        col = "#0072B2"
        for tag, c_hex in (
            ("stageb2", "#084594"),
            ("stageb1", "#0072B2"),
            ("stagea1", "#2171B5"),
            ("stagea0", "#2B8CBE"),
            ("stageb", "#0072B2"),
            ("stagea", "#2B8CBE"),
        ):
            if tag in s_lower:
                stage = f" {tag.replace('stage', 'Stg-').upper()}"
                col = "#084594" if any(v in mods for v in ("v2", "v3")) else c_hex
                break
        return f"DeltaMem-π0.5{stage}{mods} ({split} N={n_ep}{k_tag})", col, 0.89
    if "gdn2l" in s_lower or re.search(r"p2_m1(?:_|$)", s_lower):
        return f"π0.5 + 2L-GDN{mods} ({split} N={n_ep}{k_tag})", "#009E73", 0.39
    if "t2mem" in s_lower or "p2_b1" in s_lower:
        return f"π0.5 + T²Mem-1L{mods} ({split} N={n_ep}{k_tag})", "#CC79A7", 0.51
    if "framesamp" in s_lower:
        return f"FrameSamp+Modul PT ({split} N={n_ep}{k_tag})", "#D55E00", 4.00
    if "baseline" in s_lower or "pi05_base" in s_lower:
        return f"π0.5 Base PT ({split} N={n_ep}{k_tag})", "#4B4B4B", 0.001
    return f"{stem} ({split} N={n_ep}{k_tag})", "#333333", 0.89


def _stage_bucket_key(stem: str, data: Dict[str, Any]) -> str:
    """Group multiple step/CFG sweeps of the same architecture & stage into one summary bucket."""
    s_low = stem.lower()
    split = data.get("config", {}).get("dataset", "val")
    if "baseline" in s_low or "pi05_base" in s_low:
        return f"base:{split}"
    if "framesamp" in s_low:
        return f"framesamp:{split}"
    if "t2mem" in s_low or "p2_b1" in s_low:
        return f"t2mem:{split}"
    if "gdn2l" in s_low or re.search(r"p2_m1(?:_|$)", s_low):
        return f"gdn2l:{split}"
    if "m12" in s_low or "reactive" in s_low:
        return f"reactive_specialist:{split}"
    if "m11" in s_low or "continuous_vlm" in s_low or "cont_vlm" in s_low:
        return f"continuous_vlm_hybrid:{split}"
    if "m10" in s_low or "specialist" in s_low:
        return f"specialist:{split}"
    if any(k in s_low for k in ("dgvlm", "p2_m9", "m9_", "m9v", "hr16", "pyr", "m7c", "m7d", "m7e")):
        if "m7e" in s_low:
            return f"grounded_hybrid_vlm:m7e:{split}"
        if "m7d" in s_low:
            return f"grounded_hybrid_vlm:m7d:{split}"
        if any(h in s_low for h in ("m7c",)):
            return f"grounded_hybrid_vlm:m7c:{split}"
        if any(h in s_low for h in ("m7b", "hyb", "plus")):
            return f"grounded_hybrid_vlm:{split}"
        return f"deltaground_vlm:{split}"
    if any(k in s_low for k in ("vlmprefix", "p2_m8", "m8_", "m8a", "m8b")):
        m8_sub = "m8b" if ("m8b" in s_low or "64t" in s_low) else "m8a"
        return f"vlmprefix_hybrid:{m8_sub}:{split}"
    if any(k in s_low for k in ("grounded_hybrid", "p2_m7", "m7_", "m7a", "m7b", "m7c", "m7d", "m7e")):
        m7_sub = (
            "m7a"
            if "m7a" in s_low
            else ("m7e" if "m7e" in s_low else ("m7d" if "m7d" in s_low else ("m7c" if "m7c" in s_low else ("m7b" if "m7b" in s_low else "m7"))))
        )
        return f"grounded_hybrid:{m7_sub}:{split}"
    if "deltaground" in s_low or "p2_m6" in s_low:
        return f"deltaground:{split}"
    if "symbolic_grounded" in s_low or "groundsg" in s_low:
        return f"groundsg:{split}"
    if "hybrid" in s_low or "p2_m5" in s_low or "wiseft" in s_low:
        if "stagec" in s_low:
            hyb_stg = "stagec"
        elif "peft" in s_low or "v4" in s_low or "v5" in s_low:
            hyb_stg = "peft"
        elif "wiseft" in s_low:
            hyb_stg = "wiseft"
        elif "stageb" in s_low:
            hyb_stg = "stageb"
        else:
            hyb_stg = "stagea"
        return f"hybrid:{hyb_stg}:{split}"
    if "aligned" in s_low or "p2_m4" in s_low:
        return f"aligned:{split}"
    if "deltamem_loop" in s_low or "p2_m3" in s_low or "stagec" in s_low:
        return f"loop:{split}"
    for stg in ("stageb2", "stageb1", "stagea1", "stagea0", "stageb", "stagea"):
        if stg in s_low:
            return f"deltamem:{stg}:{split}"
    return f"other:{stem}:{split}"


def _select_representative_clean_runs(
    clean_runs: List[Tuple[str, Dict[str, Any]]],
    max_live: int = 4,
    dedup_stage: bool = True,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Select key baselines + highest-SR run per stage so grouped bar / scatter panels stay clean."""
    pool = list(clean_runs)
    if dedup_stage:
        best_by_bucket: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        for s, d in pool:
            bk = _stage_bucket_key(s, d)
            cand_key = (
                float(d["summary"]["overall"]["success_rate"]),
                -float(d["summary"]["overall"].get("mean_ms_per_chunk", 9999.0)),
            )
            if bk not in best_by_bucket:
                best_by_bucket[bk] = (s, d)
            else:
                prev_d = best_by_bucket[bk][1]
                prev_key = (
                    float(prev_d["summary"]["overall"]["success_rate"]),
                    -float(prev_d["summary"]["overall"].get("mean_ms_per_chunk", 9999.0)),
                )
                if cand_key > prev_key:
                    best_by_bucket[bk] = (s, d)
        pool = list(best_by_bucket.values())

    if len(pool) <= max_live:
        return pool
    chosen: List[Tuple[str, Dict[str, Any]]] = []
    seen = set()
    for pred in (
        lambda s: "baseline" in s.lower(),
        lambda s: "framesamp" in s.lower(),
    ):
        matches = [(s, d) for s, d in pool if pred(s)]
        if matches:
            matches.sort(
                key=lambda sd: (
                    sd[1].get("config", {}).get("dataset") == "test",
                    sd[1]["summary"]["overall"]["num_episodes"],
                ),
                reverse=True,
            )
            chosen.append(matches[0])
            for m_s, _ in matches:
                seen.add(m_s)
    rem = [(s, d) for s, d in pool if s not in seen]
    test_rem = [sd for sd in rem if sd[1].get("config", {}).get("dataset") == "test"]
    if test_rem and len(chosen) < max_live:
        test_rem.sort(key=lambda sd: float(sd[1]["summary"]["overall"]["success_rate"]), reverse=True)
        chosen.append(test_rem[0])
        seen.add(test_rem[0][0])
        rem = [(s, d) for s, d in pool if s not in seen]
    rem.sort(
        key=lambda sd: (
            float(sd[1]["summary"]["overall"]["success_rate"]),
            sd[1].get("config", {}).get("dataset") == "test",
        ),
        reverse=True,
    )
    for item in rem[: max(0, max_live - len(chosen))]:
        chosen.append(item)
    return chosen


def plot_fig1_leaderboard_and_suites(
    runs: Dict[str, Dict[str, Any]],
    tests: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Figure P2-1: (a) 16-Task Overall Leaderboard & Live PT Runs; (b) 4-Suite Breakdown."""
    clean_runs = [
        (s, d) for s, d in runs.items() if int(d.get("config", {}).get("interference_k", 0)) <= 1
    ]
    fig_h = max(5.8, 0.36 * (8 + len(clean_runs)) + 1.2)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.5, fig_h), gridspec_kw={"width_ratios": [1.08, 1.22]})

    pub_keys = [
        "pi05_pub",
        "rmt_modul_pub",
        "ttt_expert_pub",
        "groundsg_qwen_pub",
        "memer_pub",
        "framesamp_pub",
        "dual_stream_pub",
        "t2mem_pub",
    ]
    labels_a: List[str] = [PUBLISHED_BASELINES[k]["label"] for k in pub_keys]
    vals_a: List[float] = [PUBLISHED_BASELINES[k]["overall"] for k in pub_keys]
    colors_a: List[str] = [PUBLISHED_BASELINES[k]["color"] for k in pub_keys]
    err_low_a: List[float] = [0.0] * len(pub_keys)
    err_high_a: List[float] = [0.0] * len(pub_keys)
    hatches_a: List[str] = [""] * len(pub_keys)

    for stem, d in clean_runs:
        lbl, col, _ = _classify_run_color_and_label(stem, d)
        ov = d["summary"]["overall"]
        sr = float(ov["success_rate"]) * 100.0
        lo = max(0.0, sr - float(ov.get("success_ci95_low", ov["success_rate"])) * 100.0)
        hi = max(0.0, float(ov.get("success_ci95_high", ov["success_rate"])) * 100.0 - sr)
        labels_a.append(lbl)
        vals_a.append(sr)
        colors_a.append(col)
        err_low_a.append(lo)
        err_high_a.append(hi)
        hatches_a.append("//" if d.get("config", {}).get("dataset") == "val" else "")

    y_pos = np.arange(len(labels_a))
    bars = ax_a.barh(
        y_pos,
        vals_a,
        color=colors_a,
        edgecolor="#222222",
        linewidth=0.8,
        height=0.68,
        alpha=0.90,
    )
    for bar, h in zip(bars, hatches_a):
        if h:
            bar.set_hatch(h)

    has_ci = [i for i, (lo, hi) in enumerate(zip(err_low_a, err_high_a)) if (lo + hi) > 0]
    if has_ci:
        ax_a.errorbar(
            [vals_a[i] for i in has_ci],
            [y_pos[i] for i in has_ci],
            xerr=[[err_low_a[i] for i in has_ci], [err_high_a[i] for i in has_ci]],
            fmt="none",
            ecolor="#111111",
            elinewidth=1.1,
            capsize=2.8,
        )

    ax_a.axvline(44.51, color="#D55E00", linestyle=":", linewidth=1.2, label="MME-VLA Best (44.51%)")
    ax_a.axvline(56.83, color="#E69F00", linestyle="--", linewidth=1.4, label="T²Mem SOTA (56.83%)")
    ax_a.axvline(84.08, color="#4D4D4D", linestyle="-.", linewidth=1.1, label="GroundSG+Oracle* (84.08%)")

    for i, v in enumerate(vals_a):
        offset = err_high_a[i] + 1.0 if err_high_a[i] > 0 else 1.0
        ax_a.text(min(v + offset, 91.5), y_pos[i], f"{v:.1f}%", va="center", ha="left", fontsize=8.0, fontweight="bold")

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(labels_a, fontsize=8.2)
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 99.0)
    ax_a.set_xlabel("16-Task Success Rate (%)")
    ax_a.set_title("(a) 16-Task RoboMME Leaderboard & Live PyTorch Runs")
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.4)

    # --- Panel (b): 4-Suite Grouped Comparison (Overlap-Free Headroom) ---
    suite_series: List[Tuple[str, List[float], str, str]] = [
        ("π0.5 Base (Pub. 17.9%)", [PUBLISHED_BASELINES["pi05_pub"]["suites"][s] for s in SUITES], "#7F7F7F", ""),
        ("FrameSamp+Modul (Pub. 44.5%)", [PUBLISHED_BASELINES["framesamp_pub"]["suites"][s] for s in SUITES], "#D55E00", ""),
        ("Dual-Stream* (Pub. 49.2%)", [PUBLISHED_BASELINES["dual_stream_pub"]["suites"][s] for s in SUITES], "#56B4E9", ""),
        ("T²Mem (Pub. 56.8%)", [PUBLISHED_BASELINES["t2mem_pub"]["suites"][s] for s in SUITES], "#E69F00", ""),
    ]
    for stem, d in _select_representative_clean_runs(clean_runs, max_live=4):
        lbl, col, _ = _classify_run_color_and_label(stem, d)
        s_dict = d["summary"].get("suites", {})
        s_vals = [float(s_dict.get(s, {}).get("success_rate", 0.0)) * 100.0 for s in SUITES]
        hatch = "//" if d.get("config", {}).get("dataset") == "val" else ""
        suite_series.append((lbl, s_vals, col, hatch))

    n_ser = len(suite_series)
    x = np.arange(len(SUITES))
    width = min(0.13, 0.80 / max(1, n_ser))
    for idx, (s_lbl, s_vals, s_col, s_hatch) in enumerate(suite_series):
        pos = x + (idx - (n_ser - 1) / 2.0) * width
        b_list = ax_b.bar(pos, s_vals, width=width * 0.94, label=s_lbl, color=s_col, edgecolor="#222222", linewidth=0.7, alpha=0.90)
        if s_hatch:
            for b in b_list:
                b.set_hatch(s_hatch)
        for bx, bv in zip(pos, s_vals):
            if bv >= 4.0:
                ax_b.text(bx, bv + 1.2, f"{bv:.0f}", ha="center", va="bottom", fontsize=6.9, rotation=90)

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(SUITES, fontweight="bold")
    ax_b.set_ylim(0, 105.0)
    ax_b.set_ylabel("Suite Success Rate (%)")
    ax_b.set_title("(b) Cognitive Suite Profile (Hatched = Local val, Solid = Official test)")
    ax_b.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.94, fontsize=7.1)

    if "Test_FrameSamp_vs_Pi05Base" in tests:
        t_m1 = tests["Test_FrameSamp_vs_Pi05Base"]
        best_test_model: Optional[Tuple[str, Dict[str, Any]]] = None
        for s_tm, d_tm in clean_runs:
            if d_tm.get("config", {}).get("dataset") != "test":
                continue
            if any(b_tok in s_tm.lower() for b_tok in ("pi05_baseline", "framesamp_modul")):
                continue
            if best_test_model is None or float(d_tm["summary"]["overall"]["success_rate"]) > float(
                best_test_model[1]["summary"]["overall"]["success_rate"]
            ):
                best_test_model = (s_tm, d_tm)
        box_lines = [
            f"Official test Verify (N={t_m1['n_pairs']} test):",
            f"FrameSamp vs Base: +{t_m1['mean_diff']*100:.1f} pp (McN p={t_m1['p_mcnemar_two_sided']:.2e}, Holm p={t_m1['p_holm']:.4f})",
        ]
        if best_test_model is not None:
            tm_stem, tm_d = best_test_model
            tm_sr = float(tm_d["summary"]["overall"]["success_rate"]) * 100.0
            t_vs_b = tests.get(f"{tm_stem}_vs_test_pi05_baseline")
            t_vs_f = tests.get(f"{tm_stem}_vs_test_framesamp_modul")
            if t_vs_b is not None:
                fs_part = f", {t_vs_f['mean_diff']*100:+.1f} pp vs FS" if t_vs_f is not None else ""
                box_lines.append(
                    f"Best Model ({tm_sr:.1f}%): {t_vs_b['mean_diff']*100:+.1f} pp vs Base (Holm p={t_vs_b['p_holm']:.4f}{fs_part})"
                )
        ax_b.text(
            0.015,
            0.80,
            "\n".join(box_lines),
            transform=ax_b.transAxes,
            va="top",
            ha="left",
            fontsize=7.0,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F8F9FA", "edgecolor": "#BBBBBB", "alpha": 0.94},
        )
    elif "M1_FrameSamp_vs_Pi05Base_val80" in tests:
        t_m1 = tests["M1_FrameSamp_vs_Pi05Base_val80"]
        ax_b.text(
            0.015,
            0.79,
            f"M1 Paired Verify (N={t_m1['n_pairs']} val):\n"
            f"Δ = +{t_m1['mean_diff']*100:.1f} pp (two-sided p={t_m1['p_value_two_sided']:.4f}, Holm p={t_m1['p_holm']:.4f})",
            transform=ax_b.transAxes,
            va="top",
            ha="left",
            fontsize=7.3,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F8F9FA", "edgecolor": "#BBBBBB", "alpha": 0.94},
        )

    fig.suptitle(
        "Phase 2 (π0.5 3.44B Backbone): Primary-Source Leaderboard, Suite Bottlenecks, & PyTorch Verification",
        fontsize=11.6,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save_both(fig, out_dir, "fig_p2_1_leaderboard_and_suites")


def plot_fig2_per_task_heatmap(runs: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Figure P2-2: 16-Task × Methods Success-Rate Heatmap (Highlighting Swap & Reference Gaps)."""
    pub_order = ["pi05_pub", "ttt_expert_pub", "framesamp_pub", "dual_stream_pub", "t2mem_pub", "oracle_pub"]
    row_labels: List[str] = [PUBLISHED_BASELINES[k]["label"] for k in pub_order]
    matrix_rows: List[List[float]] = [list(PUBLISHED_BASELINES[k]["tasks"]) for k in pub_order]

    clean_runs = [
        (s, d) for s, d in runs.items() if int(d.get("config", {}).get("interference_k", 0)) <= 1
    ]
    for stem, d in clean_runs:
        lbl, _, _ = _classify_run_color_and_label(stem, d)
        t_map = d["summary"].get("tasks", {})
        row = [float(t_map.get(t, {}).get("success_rate", 0.0)) * 100.0 for t in TASKS_ORDERED]
        row_labels.append(lbl)
        matrix_rows.append(row)

    mat = np.array(matrix_rows, dtype=np.float64)
    avgs = mat.mean(axis=1, keepdims=True)
    mat_with_avg = np.concatenate([mat, avgs], axis=1)
    col_labels = [TASK_SHORT[t] for t in TASKS_ORDERED] + ["AVG"]

    fig_h = max(5.0, 0.54 * len(row_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(14.6, fig_h))
    im = ax.imshow(mat_with_avg, cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")

    for r in range(mat_with_avg.shape[0]):
        for c in range(mat_with_avg.shape[1]):
            val = mat_with_avg[r, c]
            txt_col = "white" if val >= 58.0 else "#111111"
            weight = "bold" if c == 16 else "normal"
            ax.text(c, r, f"{val:.0f}", ha="center", va="center", color=txt_col, fontsize=7.9, fontweight=weight)

    for b_col in (3.5, 7.5, 11.5, 15.5):
        ax.axvline(b_col, color="#111111", linewidth=1.5)
    ax.axhline(len(pub_order) - 0.5, color="#B22222", linestyle="--", linewidth=1.3)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=32, ha="right", fontweight="bold")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8.4)
    ax.grid(False)

    for s_idx, s_name in enumerate(SUITES):
        ax.text(s_idx * 4 + 1.5, -0.78, s_name, ha="center", va="bottom", fontsize=9.3, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.015)
    cbar.set_label("Task Success Rate (%)")
    ax.set_title(
        "16-Task RoboMME Success Matrix: Published Baselines (Top) vs. Live PyTorch Evaluations (Bottom)",
        pad=32,
    )
    fig.tight_layout()
    _save_both(fig, out_dir, "fig_p2_2_per_task_heatmap")


def plot_fig3_efficiency_pareto_and_state(runs: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    """Figure P2-3: (a) Latency vs. SR Pareto Frontier; (b) Episode Memory Footprint vs. SR."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.0, 5.3))

    # --- Panel (a): Chunk Inference Latency (log ms) vs. Success Rate ---
    pareto_pts = [
        ("π0.5 Base (Pub.)", 184.4, 17.93, "#7F7F7F", "o", (82.0, 10.2)),
        ("TTT+Expert (Pub.)", 240.0, 22.35, "#8C6BB1", "s", (82.0, 26.8)),
        ("FrameSamp+Modul (A5000 Pub.)", 440.0, 44.51, "#D55E00", "D", (82.0, 33.8)),
        ("MemER (A5000 Pub.)", 10135.6, 42.38, "#999999", "v", (4300.0, 45.2)),
        ("T²Mem (A5000 Pub.)", 146.6, 56.83, "#E69F00", "^", (82.0, 61.2)),
    ]
    for lbl, lat, sr, col, mk, (tx, ty) in pareto_pts:
        ax_a.scatter(lat, sr, s=95, color=col, marker=mk, edgecolor="#111111", linewidth=0.9, zorder=4)
        need_pub_leader = abs(ty - sr) > 3.5 or abs(math.log10(tx) - math.log10(lat)) > 0.12
        ax_a.annotate(
            lbl,
            (lat, sr),
            xytext=(tx, ty),
            fontsize=7.6,
            fontweight="bold",
            zorder=6,
            arrowprops={"arrowstyle": "-", "color": col, "lw": 0.75, "alpha": 0.60} if need_pub_leader else None,
        )

    # Keep non-overlapping boxes `(log10_x_lo, log10_x_hi, y_lo, y_hi)` in Panel (a)
    occupied_boxes: List[Tuple[float, float, float, float]] = [
        (math.log10(tx), math.log10(tx) + 0.042 * len(lbl), ty - 2.2, ty + 2.8)
        for lbl, _, _, _, _, (tx, ty) in pareto_pts
    ] + [
        (math.log10(lat) - 0.068, math.log10(lat) + 0.068, sr - 2.9, sr + 2.9)
        for _, lat, sr, _, _, _ in pareto_pts
    ]

    clean_runs = [
        (s, d) for s, d in runs.items() if int(d.get("config", {}).get("interference_k", 0)) <= 1
    ]
    # First scatter all clean stars and register their marker footprints so text never crosses any star
    for stem, d in clean_runs:
        _, col, _ = _classify_run_color_and_label(stem, d)
        ov = d["summary"]["overall"]
        lat = float(ov.get("mean_ms_per_chunk", 200.0))
        sr = float(ov["success_rate"]) * 100.0
        ax_a.scatter(lat, sr, s=130, color=col, marker="*", edgecolor="#111111", linewidth=0.9, zorder=5)
        occupied_boxes.append((math.log10(max(80.0, lat)) - 0.068, math.log10(max(80.0, lat)) + 0.068, sr - 2.9, sr + 2.9))

    # Label the representative (best-per-stage) runs without crossing any marker or prior label
    rep_runs = _select_representative_clean_runs(clean_runs, max_live=6, dedup_stage=True)
    rep_runs.sort(
        key=lambda sd: (
            sd[1].get("config", {}).get("dataset") == "test",
            float(sd[1]["summary"]["overall"]["success_rate"]),
        ),
        reverse=True,
    )
    for stem, d in rep_runs:
        lbl, col, _ = _classify_run_color_and_label(stem, d)
        pw = int(d.get("config", {}).get("parallel_workers", 1))
        if pw > 1:
            lbl = f"{lbl} [{pw}w]"
        ov = d["summary"]["overall"]
        lat = float(ov.get("mean_ms_per_chunk", 200.0))
        sr = float(ov["success_rate"]) * 100.0
        w_log = 0.042 * len(lbl)
        cand_offsets = [
            (0.22, 3.0),
            (0.22, -3.5),
            (0.22, 7.8),
            (0.22, -8.2),
            (0.22, 12.5),
            (0.22, -13.0),
            (0.10, 15.5),
            (0.10, 20.0),
            (0.22, -17.5),
            (-w_log - 0.06, 3.0),
            (-w_log - 0.06, -4.0),
        ]
        if "baseline" in stem.lower():
            cand_offsets = [(0.54, -17.2), (0.54, -12.5), (0.42, -17.2)] + cand_offsets
        elif "framesamp" in stem.lower():
            cand_offsets = [(0.24, 3.8), (0.24, -3.2)] + cand_offsets

        lx_pt = math.log10(max(80.0, lat))
        best_tx, best_ty = lat * 1.62, max(3.5, sr + 3.0)
        chosen_box_a: Optional[Tuple[float, float, float, float]] = None
        for dx_log, dy in cand_offsets:
            cx_lo = max(math.log10(78.0), min(math.log10(17500.0) - w_log, lx_pt + dx_log))
            cy = max(3.4, min(72.0, sr + dy))
            box = (cx_lo, cx_lo + w_log, cy - 2.3, cy + 2.8)
            collision = any(
                not (box[1] < ob[0] or box[0] > ob[1] or box[3] < ob[2] or box[2] > ob[3])
                for ob in occupied_boxes
            )
            if not collision:
                best_tx, best_ty = 10.0 ** cx_lo, cy
                chosen_box_a = box
                break
        if chosen_box_a is None:
            fb_lo = math.log10(max(78.0, best_tx))
            for step_k in range(1, 16):
                for sign_y in (1.0, -1.0):
                    cy_fb = max(3.4, min(72.0, sr + sign_y * 4.2 * step_k))
                    cand_fb = (fb_lo, fb_lo + w_log, cy_fb - 2.3, cy_fb + 2.8)
                    if not any(
                        not (cand_fb[1] < ob[0] or cand_fb[0] > ob[1] or cand_fb[3] < ob[2] or cand_fb[2] > ob[3])
                        for ob in occupied_boxes
                    ):
                        best_ty = cy_fb
                        chosen_box_a = cand_fb
                        break
                if chosen_box_a is not None:
                    break
            if chosen_box_a is None:
                chosen_box_a = (fb_lo, fb_lo + w_log, best_ty - 2.3, best_ty + 2.8)
        occupied_boxes.append(chosen_box_a)
        ax_a.annotate(
            lbl,
            (lat, sr),
            xytext=(best_tx, best_ty),
            fontsize=7.3,
            color=col,
            fontweight="bold",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": col, "lw": 0.80, "alpha": 0.70},
        )

    ax_a.set_xscale("log")
    ax_a.set_xlim(75.0, 18000.0)
    ax_a.set_ylim(0.0, 76.0)
    ax_a.axhline(56.83, color="#E69F00", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_a.set_xlabel("Foreground Latency per Chunk (ms, log; Pub.=A5000, PT=RTX 5080) ↓")
    ax_a.set_ylabel("16-Task Success Rate (%) ↑")
    ax_a.set_title("(a) Closed-Loop Speed–Accuracy Pareto Frontier")

    # --- Panel (b): Episode Fast-State Footprint (MiB, log scale) vs. Success Rate ---
    state_pts = [
        ("TTT+Expert (0.51 MiB)", 0.51, 22.35, "#8C6BB1", "s", (0.25, 26.5)),
        ("RMT+Modul (1.00 MiB)", 1.00, 20.17, "#A6761D", "o", (1.22, 18.2)),
        ("Dual-Stream* (~3.2 MiB)", 3.20, 49.22, "#56B4E9", "P", (3.65, 51.0)),
        ("FrameSamp+Modul (4.00 MiB)", 4.00, 44.51, "#D55E00", "D", (4.70, 45.8)),
        ("MemER (~6.0 MiB + VLM)", 6.00, 42.38, "#999999", "v", (7.00, 41.2)),
        ("T²Mem (36.35 MiB, 1,536 H200-h)", 36.35, 56.83, "#E69F00", "^", (11.5, 59.0)),
    ]
    for lbl, st_mib, sr, col, mk, (tx, ty) in state_pts:
        ax_b.scatter(st_mib, sr, s=95, color=col, marker=mk, edgecolor="#111111", linewidth=0.9, zorder=4)
        ax_b.annotate(lbl, (st_mib, sr), xytext=(tx, ty), fontsize=7.7, fontweight="bold")

    ax_b.axvspan(0.35, 0.95, color="#0072B2", alpha=0.10, label="DeltaMem-π0.5 State Band (0.39–0.89 MiB: 41–93× < T²Mem)")
    occ_b: List[Tuple[float, float, float, float]] = [
        (math.log10(tx), math.log10(tx) + 0.024 * len(lbl), ty - 1.9, ty + 2.5)
        for lbl, _, _, _, _, (tx, ty) in state_pts
    ] + [
        (math.log10(st_mib) - 0.06, math.log10(st_mib) + 0.06, sr - 2.2, sr + 2.2)
        for _, st_mib, sr, _, _, _ in state_pts
    ]
    for stem, d in clean_runs:
        if "baseline" in stem.lower():
            continue
        _, col, st_mib = _classify_run_color_and_label(stem, d)
        sr = float(d["summary"]["overall"]["success_rate"]) * 100.0
        ax_b.scatter(st_mib, sr, s=130, color=col, marker="*", edgecolor="#111111", linewidth=0.9, zorder=5)
        occ_b.append((math.log10(st_mib) - 0.065, math.log10(st_mib) + 0.065, sr - 2.2, sr + 2.2))

    # Label distinct non-FrameSamp architecture families in Panel (b) (1 label per state-footprint family, zero overlaps)
    best_by_state_fam: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for stem, d in _select_representative_clean_runs(clean_runs, max_live=8, dedup_stage=True):
        if any(k in stem.lower() for k in ("baseline", "framesamp")):
            continue
        lbl_f, _, st_f = _classify_run_color_and_label(stem, d)
        fam_key = f"{lbl_f.split()[0]}:{st_f:.2f}"
        cur_sc = (float(d["summary"]["overall"]["success_rate"]), "stageb" in stem.lower())
        if fam_key not in best_by_state_fam or cur_sc > (
            float(best_by_state_fam[fam_key][1]["summary"]["overall"]["success_rate"]),
            "stageb" in best_by_state_fam[fam_key][0].lower(),
        ):
            best_by_state_fam[fam_key] = (stem, d)

    for stem, d in best_by_state_fam.values():
        lbl, col, st_mib = _classify_run_color_and_label(stem, d)
        sr = float(d["summary"]["overall"]["success_rate"]) * 100.0
        fam_title = (lbl.split("*")[0] + "*") if "*" in lbl else lbl.split(" (")[0]
        short_b = f"{fam_title} ({st_mib:.2f} MiB)"
        w_log_b = 0.035 * len(short_b)
        lx_b = math.log10(st_mib)
        cands_b = [
            (0.16, 4.0),
            (0.16, -5.0),
            (0.16, 8.5),
            (0.16, -9.5),
            (0.16, 12.8),
            (0.16, -13.8),
            (0.24, 9.5),
            (0.24, -11.0),
        ]
        best_bx, best_by = st_mib * 1.45, max(5.5, sr + 8.0)
        chosen_box_b: Optional[Tuple[float, float, float, float]] = None
        for dx_l, dy_b in cands_b:
            cx_lo = max(math.log10(0.24), min(math.log10(68.0) - w_log_b, lx_b + dx_l))
            cy_b = max(5.5, min(68.0, sr + dy_b))
            box_b = (cx_lo, cx_lo + w_log_b, cy_b - 2.2, cy_b + 2.8)
            if not any(
                not (box_b[1] < ob[0] or box_b[0] > ob[1] or box_b[3] < ob[2] or box_b[2] > ob[3])
                for ob in occ_b
            ):
                best_bx, best_by = 10.0 ** cx_lo, cy_b
                chosen_box_b = box_b
                break
        if chosen_box_b is None:
            fb_b = math.log10(max(0.24, best_bx))
            for step_k in range(1, 16):
                for sign_y in (1.0, -1.0):
                    cy_fb = max(5.5, min(68.0, sr + sign_y * 3.8 * step_k))
                    cand_fb = (fb_b, fb_b + w_log_b, cy_fb - 2.2, cy_fb + 2.8)
                    if not any(
                        not (cand_fb[1] < ob[0] or cand_fb[0] > ob[1] or cand_fb[3] < ob[2] or cand_fb[2] > ob[3])
                        for ob in occ_b
                    ):
                        best_by = cy_fb
                        chosen_box_b = cand_fb
                        break
                if chosen_box_b is not None:
                    break
            if chosen_box_b is None:
                chosen_box_b = (fb_b, fb_b + w_log_b, best_by - 2.2, best_by + 2.8)
        occ_b.append(chosen_box_b)
        ax_b.annotate(
            short_b,
            (st_mib, sr),
            xytext=(best_bx, best_by),
            fontsize=7.3,
            color=col,
            fontweight="bold",
            zorder=6,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.86},
            arrowprops={"arrowstyle": "-", "color": col, "lw": 0.80, "alpha": 0.70},
        )

    ax_b.set_xscale("log")
    ax_b.set_xlim(0.22, 85.0)
    ax_b.set_ylim(0.0, 72.0)
    ax_b.axhline(56.83, color="#E69F00", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_b.set_xlabel("Episode Memory State Footprint (MiB, log scale) ↓")
    ax_b.set_ylabel("16-Task Success Rate (%) ↑")
    ax_b.set_title("(b) State Compression Efficiency vs. T²Mem (36.35 MiB)")
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=7.6)

    fig.tight_layout()
    _save_both(fig, out_dir, "fig_p2_3_efficiency_pareto_and_state")


def plot_fig4_interference_and_parity(
    runs: Dict[str, Dict[str, Any]],
    parity_summary: Optional[Dict[str, Any]],
    tests: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> None:
    """Figure P2-4: (a) Cross-Session Interference Curves; (b) M1 PyTorch Parity & VRAM Telemetry."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.0, 5.3))

    k_vals = [0, 1, 3, 7]
    pub_styles = [
        ("FrameSamp-Modul (2606.22338)", "#D55E00", "-o"),
        ("TokenDrop-Modul (2606.22338)", "#56B4E9", "-s"),
        ("GroundSG+QwenVL (2606.22338)", "#66A61E", "-^"),
        ("MemER (2606.22338)", "#999999", "-v"),
        ("Recurrent-TTT-Expert (2606.22338)", "#8C6BB1", ":d"),
    ]
    for name, col, style in pub_styles:
        ax_a.plot(k_vals, INTERFERENCE_PUBLISHED[name], style, color=col, linewidth=1.8, markersize=5.5, alpha=0.84, label=name)

    interf_runs = [
        (s, d) for s, d in runs.items() if int(d.get("config", {}).get("interference_k", 0)) > 1
    ]
    low_slots = [
        (0.10, 1.8),
        (3.55, 9.4),
        (3.55, 1.8),
        (0.10, 9.4),
    ]
    high_slots = [
        (3.18, 42.6),
        (3.18, 37.6),
        (3.18, 32.6),
    ]
    low_idx = 0
    high_idx = 0
    for s_k, d_k in interf_runs:
        k_sess = int(d_k["config"]["interference_k"])
        n_prior = k_sess - 1  # `--interference-k 3` = 2 prior un-reset distractor episodes + 1 target episode
        base_stem = s_k.replace(f"_interfk{k_sess}", "").replace(f"_k{k_sess}", "")
        if base_stem in runs:
            sr0 = float(runs[base_stem]["summary"]["overall"]["success_rate"]) * 100.0
            srk = float(d_k["summary"]["overall"]["success_rate"]) * 100.0
            split_k = d_k.get("config", {}).get("dataset", "val")
            lbl, col, _ = _classify_run_color_and_label(base_stem, runs[base_stem])
            if "framesamp" in base_stem.lower():
                short_tag = f"FrameSamp ({split_k})"
            elif any(k in base_stem.lower() for k in ("m12", "reactive")):
                short_tag = f"Reactive-Spec* ({split_k})"
            elif any(k in base_stem.lower() for k in ("m11", "continuous_vlm", "cont_vlm")):
                short_tag = f"Cont-VLM-Hyb* ({split_k})"
            elif any(k in base_stem.lower() for k in ("m10", "specialist")):
                short_tag = f"Specialist* ({split_k})"
            elif any(k in base_stem.lower() for k in ("dgvlm", "p2_m9", "m9_", "m9v", "hr16", "pyr", "m7c", "m7d", "m7e")):
                short_tag = f"Grounded-Hyb-VLM* ({split_k})"
            elif any(k in base_stem.lower() for k in ("vlmprefix", "p2_m8", "m8_")):
                short_tag = f"VLM-Hybrid* ({split_k})"
            elif any(k in base_stem.lower() for k in ("grounded_hybrid", "p2_m7", "m7_", "m7a", "m7b", "m7c", "m7d", "m7e")):
                short_tag = f"Grounded-Hybrid* ({split_k})"
            elif "hybrid" in base_stem.lower():
                short_tag = f"Hybrid* ({split_k})"
            elif "deltaground" in base_stem.lower():
                short_tag = f"DeltaGround* ({split_k})"
            else:
                short_tag = f"DeltaMem ({split_k})"
            ax_a.plot(
                [0, n_prior],
                [sr0, srk],
                "--*",
                color=col,
                linewidth=2.5,
                markersize=10.0,
                label=f"Live Un-Reset (k={k_sess}): {lbl}",
            )
            t_key = f"InterfDrop_{s_k}_vs_{base_stem}"
            fs_ref = (
                f"test_framesamp_modul_interfk{k_sess}"
                if split_k == "test"
                else f"m1_framesamp_modul_val80_interfk{k_sess}"
            )
            h2h_key = f"InterfHeadToHead_{s_k}_vs_{fs_ref}"
            if t_key in tests:
                t_info = tests[t_key]
                if srk >= 35.0:
                    bx, by = high_slots[high_idx % len(high_slots)]
                    high_idx += 1
                else:
                    bx, by = low_slots[low_idx % len(low_slots)]
                    low_idx += 1
                second_line = f"p={t_info['p_value_two_sided']:.4f} (Holm p={t_info['p_holm']:.4f})"
                if h2h_key in tests and srk >= 35.0:
                    h2h = tests[h2h_key]
                    second_line = f"vs FrameSamp k={k_sess}: {h2h['mean_diff']*100:+.1f} pp (Holm p={h2h['p_holm']:.4f})"
                ax_a.annotate(
                    f"{short_tag}: {sr0:.1f}%→{srk:.1f}% ({t_info['mean_diff']*100:+.1f} pp)\n{second_line}",
                    (n_prior, srk),
                    xytext=(bx, by),
                    fontsize=7.0,
                    fontweight="bold",
                    color=col,
                    arrowprops={"arrowstyle": "->", "color": col, "lw": 1.0},
                    bbox={"boxstyle": "round,pad=0.20", "facecolor": "#FFFDF9", "edgecolor": col, "alpha": 0.95},
                    zorder=6,
                )

    ax_a.axhline(17.3, color="#4D4D4D", linestyle="--", linewidth=1.0, label="π0.5 No-History Floor (17.3%, 2606.22338)")
    ax_a.set_xticks([0, 1, 2, 3, 7])
    ax_a.set_xticklabels(["0\n(Clean)", "1\n(+1 Prior)", "2\n(k=3 Sess)", "3\n(+3 Prior)", "7\n(+7 Prior)"])
    ax_a.set_ylim(0.0, 76.5)
    ax_a.set_xlabel("Prior Unrelated Distractor Episodes / Sessions")
    ax_a.set_ylabel("Cross-Session Success Rate (%)")
    ax_a.set_title("(a) Cross-Session Distractor Interference (arXiv:2606.22338 + Live Un-Reset)")
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.93, fontsize=6.5)

    # --- Panel (b): Milestone M1 Official Orbax vs. Converted PyTorch Bridge Verification ---
    categories = ["Overall\n(16 Tasks)", "Counting", "Permanence", "Reference", "Imitation"]
    base_pub = [17.93, 28.78, 17.00, 17.17, 8.78]
    fs_pub = [44.51, 65.22, 25.11, 36.33, 51.39]

    base_pt_run = runs.get("test_pi05_baseline") or runs.get("m1_pi05_baseline_val80")
    fs_pt_run = runs.get("test_framesamp_modul") or runs.get("m1_framesamp_modul_val80")

    x = np.arange(len(categories))
    w = 0.19
    ax_b.bar(x - 1.5 * w, base_pub, width=w, color="#B0B0B0", edgecolor="#222222", label="Yinpei/pi05_baseline (Pub. test)")
    if base_pt_run is not None:
        b_vals = [float(base_pt_run["summary"]["overall"]["success_rate"]) * 100.0] + [
            float(base_pt_run["summary"]["suites"][s]["success_rate"]) * 100.0 for s in SUITES
        ]
        b_split = base_pt_run.get("config", {}).get("dataset", "val")
        b_n = base_pt_run["summary"]["overall"]["num_episodes"]
        b_hatch = "//" if b_split == "val" else ""
        ax_b.bar(x - 0.5 * w, b_vals, width=w, color="#555555", edgecolor="#222222", hatch=b_hatch, label=f"pi05_baseline_pt ({b_split} N={b_n})")

    ax_b.bar(x + 0.5 * w, fs_pub, width=w, color="#F4A582", edgecolor="#222222", label="FrameSamp+Modul (Pub. test)")
    if fs_pt_run is not None:
        f_vals = [float(fs_pt_run["summary"]["overall"]["success_rate"]) * 100.0] + [
            float(fs_pt_run["summary"]["suites"][s]["success_rate"]) * 100.0 for s in SUITES
        ]
        f_split = fs_pt_run.get("config", {}).get("dataset", "val")
        f_n = fs_pt_run["summary"]["overall"]["num_episodes"]
        f_hatch = "//" if f_split == "val" else ""
        ax_b.bar(x + 1.5 * w, f_vals, width=w, color="#D55E00", edgecolor="#222222", hatch=f_hatch, label=f"framesamp_modul_pt ({f_split} N={f_n})")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(categories, fontweight="bold")
    ax_b.set_ylim(0.0, 96.0)
    ax_b.set_ylabel("Success Rate (%)")
    ax_b.set_title("(b) Milestone M1 Orbax→PyTorch Bridge Parity & Closed-Loop Validation")
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=7.1)

    if parity_summary is not None:
        b_vram = parity_summary.get("pi05_baseline_pt", {}).get("peak_eval_with_sapien_vram_mib", 6674.6) / 1024.0
        f_vram = parity_summary.get("perceptual_framesamp_modul_pt", {}).get("peak_eval_with_sapien_vram_mib", 6851.3) / 1024.0
        fp32_diff = parity_summary.get("pi05_baseline_pt", {}).get("parity_metrics", {}).get("float32_vs_flax_float32_vt_max_diff", 1.67e-6)
        ax_b.text(
            0.015,
            0.97,
            f"Orbax→PyTorch fp32 max diff = {fp32_diff:.2e}\n"
            f"Step-0 modul_residual diff = 0.0 (exact)\n"
            f"RTX 5080 VRAM: {b_vram:.2f} / {f_vram:.2f} GiB",
            transform=ax_b.transAxes,
            va="top",
            ha="left",
            fontsize=7.3,
            bbox={"boxstyle": "round,pad=0.26", "facecolor": "#F8F9FA", "edgecolor": "#BBBBBB", "alpha": 0.94},
        )

    fig.tight_layout()
    _save_both(fig, out_dir, "fig_p2_4_interference_and_parity")


def build_hypothesis_tests(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Construct all matched-episode paired hypothesis tests from completed Phase-2 runs."""
    tests: Dict[str, Dict[str, Any]] = {}
    if "m1_framesamp_modul_val80" in runs and "m1_pi05_baseline_val80" in runs:
        res = paired_test_episodes(runs["m1_framesamp_modul_val80"], runs["m1_pi05_baseline_val80"])
        if res is not None:
            tests["M1_FrameSamp_vs_Pi05Base_val80"] = res

    if "test_framesamp_modul" in runs and "test_pi05_baseline" in runs:
        res = paired_test_episodes(runs["test_framesamp_modul"], runs["test_pi05_baseline"])
        if res is not None:
            tests["Test_FrameSamp_vs_Pi05Base"] = res

    for stem, d in runs.items():
        s_low = stem.lower()
        split = d.get("config", {}).get("dataset", "val")
        interf_k = int(d.get("config", {}).get("interference_k", 0))
        if interf_k > 1:
            clean_stem = stem.replace(f"_interfk{interf_k}", "").replace(f"_k{interf_k}", "")
            if clean_stem in runs:
                res = paired_test_episodes(d, runs[clean_stem])
                if res is not None:
                    tests[f"InterfDrop_{stem}_vs_{clean_stem}"] = res
            if "framesamp" not in s_low:
                for fs_k in (f"m1_framesamp_modul_val80_interfk{interf_k}", f"test_framesamp_modul_interfk{interf_k}"):
                    if fs_k in runs and runs[fs_k].get("config", {}).get("dataset") == split:
                        res_k = paired_test_episodes(d, runs[fs_k])
                        if res_k is not None:
                            tests[f"InterfHeadToHead_{stem}_vs_{fs_k}"] = res_k
            continue
        if any(tok in s_low for tok in ("deltamem", "deltaground", "groundsg", "gdn2l", "p2_m", "p2_b1", "t2mem", "hybrid", "aligned", "m7", "m8", "m9", "m10", "m11", "m12", "hr16", "pyr", "reactive", "continuous_vlm", "specialist", "vlmprefix", "dgvlm")):
            for base_cand in ("m1_pi05_baseline_val80", "test_pi05_baseline", "m1_framesamp_modul_val80", "test_framesamp_modul"):
                if base_cand in runs and runs[base_cand].get("config", {}).get("dataset") == split:
                    res = paired_test_episodes(d, runs[base_cand])
                    if res is not None:
                        tests[f"{stem}_vs_{base_cand}"] = res

    if tests:
        apply_holm_bonferroni(tests)
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Phase-2 publication figures (CPU-only).")
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR, help="Directory with Phase-2 eval JSONs (read-only).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory for PNG/PDF figures.")
    parser.add_argument("--stats-out", type=Path, default=None, help="Optional output JSON for Holm-adjusted paired tests.")
    args = parser.parse_args()

    configure_style()
    runs, parity_summary = load_phase2_runs(args.eval_dir)
    tests = build_hypothesis_tests(runs)

    plot_fig1_leaderboard_and_suites(runs, tests, args.out_dir)
    plot_fig2_per_task_heatmap(runs, args.out_dir)
    plot_fig3_efficiency_pareto_and_state(runs, args.out_dir)
    plot_fig4_interference_and_parity(runs, parity_summary, tests, args.out_dir)

    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(json.dumps({"num_runs": len(runs), "runs": list(runs.keys()), "tests": tests}, indent=2) + "\n")

    print(f"[plot_phase2] Loaded {len(runs)} completed Phase-2 runs, computed {len(tests)} paired tests, rendered 4 PNG+PDF figures into {args.out_dir}")


if __name__ == "__main__":
    main()
