#!/usr/bin/env python3
"""Publication Figure Generator for Phase 3 (`DeltaMem-pi0.5 v3 + CORM`)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
})

REPO = Path("/home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla")
OUT_DIR = REPO / "docs/figures_phase3"
WEB_ASSETS = REPO / "docs/website/assets"
ART_DIR = Path("/home/kohei/.gemini/antigravity-cli/brain/b1ad88c2-334f-474b-bf14-0cb9ef0dc26d")

TASKS_16 = [
    "BinFill", "PickXtimes", "SwingXtimes", "StopCube",
    "VideoUnmask", "ButtonUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap",
    "PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder",
    "MoveCube", "InsertPeg", "PatternLock", "RouteStick",
]
SHORT_NAMES = [
    "BinFill", "PickX", "SwingX", "StopCube",
    "VidUmsk", "BtnUmsk", "VidSwap", "BtnSwap",
    "PickHL", "VidRepk", "VidPlcBtn", "VidPlcOrd",
    "MoveCube", "InsPeg", "PatLock", "RteStick",
]

FRAMESAMP_PUB_BY_TASK = {
    "BinFill": 39.56, "PickXtimes": 87.33, "SwingXtimes": 92.00, "StopCube": 42.00,
    "VideoUnmask": 32.67, "ButtonUnmask": 25.11, "VideoUnmaskSwap": 24.44, "ButtonUnmaskSwap": 18.22,
    "PickHighlight": 22.89, "VideoRepick": 30.44, "VideoPlaceButton": 60.00, "VideoPlaceOrder": 32.00,
    "MoveCube": 77.78, "InsertPeg": 7.56, "PatternLock": 53.56, "RouteStick": 66.67,
}


def save_both(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WEB_ASSETS.mkdir(parents=True, exist_ok=True)
    png_p = OUT_DIR / f"{stem}.png"
    pdf_p = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_p, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_p, bbox_inches="tight")
    shutil.copy2(png_p, WEB_ASSETS / f"{stem}.png")
    if ART_DIR.exists():
        shutil.copy2(png_p, ART_DIR / f"{stem}.png")
    plt.close(fig)
    print(f"Saved {png_p} and {pdf_p}")


def main() -> None:
    stats_path = REPO / "results/phase3/phase3_statistical_tests.json"
    st = json.loads(stats_path.read_text())
    r800 = st["all_episodes"]
    r560 = st["untouched_ep15_49"]
    r240 = st["ep0_14_subset"]
    rk3 = st.get("interfk3")

    # -------------------------------------------------------------------------
    # Figure 1: Leaderboard Overall + 4-Suite Breakdown
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.0, 5.3), gridspec_kw={"width_ratios": [1.14, 1.0], "wspace": 0.24}
    )

    methods = [
        "π0.5 Base\n(Published)",
        "MemER\n(Qwen3-VL-4B)",
        "FrameSamp+\nModulator",
        "Dual-Stream\n(Mask-Sup.)",
        "Ours Phase-2\n(test240)",
        "T²Mem\n(1,536 H200-h)",
        "Ours v3+CORM\n(Untouched N=560)",
        "Ours v3+CORM\n(Official N=800)",
    ]
    vals = [
        17.93, 42.38, 44.51, 49.22, 48.75, 56.83,
        r560["sr"] * 100.0, r800["sr"] * 100.0,
    ]
    colors = [
        "#94a3b8", "#64748b", "#3b82f6", "#0ea5e9", "#6366f1",
        "#d97706", "#059669", "#047857",
    ]
    y_pos = np.arange(len(methods))
    bars = ax1.barh(y_pos, vals, color=colors, height=0.66, edgecolor="#1e293b", linewidth=0.6)
    ax1.axvline(56.83, color="#d97706", linestyle="--", linewidth=1.4, alpha=0.85, label="T²Mem SOTA (56.83%)")

    hi_map = {}
    for idx_b, r_obj in [(6, r560), (7, r800)]:
        lo, hi = r_obj["clopper_pearson_ci95"][0] * 100.0, r_obj["clopper_pearson_ci95"][1] * 100.0
        c_v = r_obj["sr"] * 100.0
        hi_map[idx_b] = hi
        ax1.errorbar(c_v, idx_b, xerr=[[c_v - lo], [hi - c_v]], fmt="none", ecolor="#0f172a", elinewidth=1.6, capsize=4)

    for i, (b, v) in enumerate(zip(bars, vals)):
        fw = "bold" if i >= 5 else "normal"
        x_txt = (hi_map[i] + 1.1) if i in hi_map else (v + 1.1)
        ax1.text(x_txt, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", fontsize=9.4, fontweight=fw, color="#0f172a")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods, fontsize=9.2)
    ax1.set_xlim(0, 88)
    ax1.set_xlabel("16-Task Closed-Loop Success Rate (%)", fontsize=10.5, fontweight="bold")
    p_str = f"p = {r800['p_z_stratified_two_sample_two_sided']:.2e}"
    ax1.set_title(f"(a) Held-Out RoboMME test Leaderboard ({p_str} vs. T²Mem)", fontsize=10.8)
    ax1.legend(loc="lower right", frameon=True, fontsize=8.8)

    # Panel (b): Suite breakdown
    suites = ["Counting", "Permanence", "Reference", "Imitation"]
    fs_s = [65.22, 25.11, 36.33, 51.39]
    p2_s = [65.00, 31.67, 45.00, 53.33]
    t2_s = [78.17, 49.17, 38.67, 61.34]
    p3_s = [r800["by_suite"][s]["sr"] * 100.0 for s in suites]

    x = np.arange(len(suites))
    w = 0.20
    ax2.bar(x - 1.5 * w, fs_s, w, label="FrameSamp+Modul", color="#3b82f6", edgecolor="#1e293b", linewidth=0.5)
    ax2.bar(x - 0.5 * w, p2_s, w, label="Ours Phase-2 (Hop-2)", color="#6366f1", edgecolor="#1e293b", linewidth=0.5)
    ax2.bar(x + 0.5 * w, t2_s, w, label="T²Mem (Published)", color="#d97706", edgecolor="#1e293b", linewidth=0.5)
    rects_p3 = ax2.bar(x + 1.5 * w, p3_s, w, label="Ours v3 + CORM (test800)", color="#047857", edgecolor="#1e293b", linewidth=0.6)

    for rect in rects_p3:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2.0, h + 1.4, f"{h:.1f}", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#064e3b")

    ax2.set_xticks(x)
    ax2.set_xticklabels(suites, fontsize=10.0, fontweight="bold")
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Suite Success Rate (%)", fontsize=10.5, fontweight="bold")
    ax2.set_title("(b) Suite-Level Dissection on Official test Split (N = 200 / Suite)", fontsize=10.8)
    ax2.legend(loc="upper left", ncol=2, frameon=True, fontsize=8.3)

    save_both(fig, "fig_p3_1_leaderboard_and_suites")

    # -------------------------------------------------------------------------
    # Figure 2: 16-Task Head-to-Head Bar Chart
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14.2, 5.1))
    x16 = np.arange(len(TASKS_16))
    bw = 0.26
    fs_16 = [FRAMESAMP_PUB_BY_TASK[t] for t in TASKS_16]
    t2_16 = [r800["by_task"][t]["t2mem_sr"] * 100.0 for t in TASKS_16]
    p3_16 = [r800["by_task"][t]["sr"] * 100.0 for t in TASKS_16]

    ax.bar(x16 - bw, fs_16, bw, label="FrameSamp+Modul (44.51%)", color="#93c5fd", edgecolor="#1e293b", linewidth=0.45)
    ax.bar(x16, t2_16, bw, label="T²Mem SOTA (56.83%)", color="#f59e0b", edgecolor="#1e293b", linewidth=0.45)
    b_ours = ax.bar(x16 + bw, p3_16, bw, label=f"Ours: DeltaMem-π0.5 v3 + CORM ({r800['sr']*100:.2f}%)", color="#059669", edgecolor="#0f172a", linewidth=0.55)

    for i, rect in enumerate(b_ours):
        h = rect.get_height()
        diff = h - t2_16[i]
        col_txt = "#065f46" if diff >= 0 else "#475569"
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 1.3, f"{h:.0f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=col_txt)

    for sep in (3.5, 7.5, 11.5):
        ax.axvline(sep, color="#cbd5e1", linestyle="--", linewidth=1.1)

    ax.set_xticks(x16)
    ax.set_xticklabels(SHORT_NAMES, rotation=25, ha="right", fontsize=9.2, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Test Success Rate (%, N = 50 / Task)", fontsize=10.5, fontweight="bold")
    ax.set_title("Per-Task Official RoboMME test Performance (50 Episodes × 16 Tasks = 800 Rollouts, Zero Privileged Inputs)", fontsize=11.5)
    ax.legend(loc="upper right", ncol=3, frameon=True, fontsize=9.2)
    fig.tight_layout()
    save_both(fig, "fig_p3_2_per_task_comparison")

    # -------------------------------------------------------------------------
    # Figure 3: Un-Reset k=3 Interference + Fast-State Footprint Pareto
    # -------------------------------------------------------------------------
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.8, 4.9), gridspec_kw={"wspace": 0.25})

    labels_k = ["FrameSamp+Modul\n(PyTorch test240)", "Ours Phase-2\n(Hybrid Hop-2)", "Ours Phase-3\n(v3 + CORM)"]
    k1_vals = [45.83, 48.75, r240["sr"] * 100.0]
    k3_vals = [15.42, 41.67, (rk3["sr"] * 100.0 if rk3 else r240["sr"] * 100.0)]
    xk = np.arange(len(labels_k))
    wk = 0.32
    b_k1 = ax_a.bar(xk - wk / 2, k1_vals, wk, label="Clean Single-Episode (k = 1)", color="#3b82f6", edgecolor="#1e293b", linewidth=0.6)
    b_k3 = ax_a.bar(xk + wk / 2, k3_vals, wk, label="Un-Reset Multi-Session (k = 3, +2 Distractors)", color="#059669", edgecolor="#1e293b", linewidth=0.6)

    for r_bar in list(b_k1) + list(b_k3):
        hv = r_bar.get_height()
        ax_a.text(r_bar.get_x() + r_bar.get_width() / 2, hv + 1.3, f"{hv:.1f}%", ha="center", va="bottom", fontsize=9.0, fontweight="bold")

    ax_a.set_xticks(xk)
    ax_a.set_xticklabels(labels_k, fontsize=9.6, fontweight="bold")
    ax_a.set_ylim(0, 92)
    ax_a.set_ylabel("Held-Out test240 Success Rate (%)", fontsize=10.2, fontweight="bold")
    ax_a.set_title("(a) Un-Reset Cross-Session Interference (k = 1 vs. k = 3)", fontsize=10.6)
    ax_a.legend(loc="upper left", frameon=True, fontsize=8.8)

    # Panel (b): Fast-state size vs Official test SR Pareto
    pts = [
        ("π0.5 Base", 0.01, 17.93, "#94a3b8", (1.6, 17.0)),
        ("FrameSamp+Modul", 4.00, 44.51, "#3b82f6", (5.8, 40.5)),
        ("Ours Phase-2 (Hop-2)", 2.39, 48.75, "#6366f1", (1.2, 52.8)),
        ("T²Mem (36.35 MiB)", 36.35, 56.83, "#d97706", (25.5, 60.0)),
        ("Ours v3 + CORM", 3.04, r800["sr"] * 100.0, "#047857", (4.8, r800["sr"] * 100.0 + 1.5)),
    ]
    for name, mem_mb, sr_v, col, (tx, ty) in pts:
        sz = 200 if "v3 + CORM" in name else 115
        ax_b.scatter([mem_mb], [sr_v], s=sz, color=col, edgecolors="#0f172a", linewidths=1.1, zorder=4)
        ax_b.annotate(
            f"{name}\n({sr_v:.2f}%)",
            (mem_mb, sr_v),
            xytext=(tx, ty),
            fontsize=8.8,
            fontweight="bold" if "CORM" in name else "normal",
        )

    ax_b.axhline(56.83, color="#d97706", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_b.set_xlim(-2, 41)
    ax_b.set_ylim(10, 86)
    ax_b.set_xlabel("Per-Episode Fast Memory State Footprint (MiB, Lower is Better)", fontsize=10.0, fontweight="bold")
    ax_b.set_ylabel("Official test Success Rate (%)", fontsize=10.2, fontweight="bold")
    ax_b.set_title("(b) Test SR vs. Fast-State Footprint (12× Smaller than T²Mem)", fontsize=10.6)

    save_both(fig, "fig_p3_3_interference_and_corm")


if __name__ == "__main__":
    main()
