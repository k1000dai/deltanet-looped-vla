#!/usr/bin/env python3
"""Generate publication-quality Cycle 1 figures from saved 3-seed synthetic JSON results.

Strictly CPU-only (`CUDA_VISIBLE_DEVICES=""`). Outputs PNG (300 DPI) and vector PDF into
`results/synthetic/figures/`:
  1. `fig1_mechanistic_dissociation.{png,pdf}`
  2. `fig2_k_L_staircase_heatmap.{png,pdf}`
  3. `fig3_halting_and_qk_alignment.{png,pdf}`
  4. `fig4_task1b_multiscale.{png,pdf}`
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = REPO_ROOT / "results" / "synthetic"
FIG_DIR = SYNTH_DIR / "figures"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.0,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


ARCH_ORDER_1A: List[Tuple[str, str]] = [
    ("single_gdn_b1", "Single GDN\n(β∈(0,1), L=1)"),
    ("single_gdn_b2", "Single GDN\n(β∈(0,2), L=1)"),
    ("two_layer_gdn_b2", "2-Layer GDN\n(β∈(0,2), L=1)"),
    ("write_loop_gdn_b2", "Write-Loop GDN\n(DeltaProd+LT2)"),
    ("static_query_loop_b2", "Static-Q Loop\n(β∈(0,2), L=6)"),
    ("parallel_multiquery_b2", "Parallel MQ\n(K=6, L=1)"),
    ("untied_deep_readout_b2", "Untied Deep\n(L=6, 3.7× P)"),
    ("deltaloop_b1", "DeltaLoop (Ours)\n(β∈(0,1), L=6)"),
    ("deltaloop_b2", "DeltaLoop (Ours)\n(β∈(0,2), L=6)"),
    ("softmax_transformer", "Softmax Transf.\n(4L, 1-ep)"),
]


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _bounded_yerr(means: Sequence[float], stds: Sequence[float]) -> np.ndarray:
    """Clip symmetric std error bars to valid [0, 100]% accuracy bounds."""
    m = np.asarray(means, dtype=np.float64)
    s = np.asarray(stds, dtype=np.float64)
    low = np.minimum(s, np.maximum(0.0, m))
    high = np.minimum(s, np.maximum(0.0, 100.0 - m))
    return np.vstack([low, high])


def make_fig1_mechanistic_dissociation(d_online: Dict, d_retro: Dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.6, 9.0), sharex=False)

    keys = [k for k, _ in ARCH_ORDER_1A]
    labels = [lbl for _, lbl in ARCH_ORDER_1A]
    x = np.arange(len(keys))

    # --- Panel (a): Task 1A-Online ---
    on_id_m, on_id_s = [], []
    on_ood_m, on_ood_s = [], []
    for k in keys:
        runs = d_online["runs"][k]
        m1, s1 = _mean_std([r["id_acc"] for r in runs])
        m2, s2 = _mean_std([r["ood_hops_acc"] for r in runs])
        on_id_m.append(m1)
        on_id_s.append(s1)
        on_ood_m.append(m2)
        on_ood_s.append(s2)

    w1 = 0.36
    yerr_id = _bounded_yerr(on_id_m, on_id_s)
    yerr_ood = _bounded_yerr(on_ood_m, on_ood_s)

    ax1.bar(
        x - w1 / 2,
        on_id_m,
        w1,
        yerr=yerr_id,
        capsize=3,
        color="#2b6cb0",
        edgecolor="#1a365d",
        linewidth=0.7,
        label="ID Accuracy (k ∈ {1..4} swaps, T=40)",
        error_kw={"elinewidth": 1.0, "ecolor": "#1a202c"},
    )
    ax1.bar(
        x + w1 / 2,
        on_ood_m,
        w1,
        yerr=yerr_ood,
        capsize=3,
        color="#63b3ed",
        edgecolor="#2c5282",
        linewidth=0.7,
        label="Zero-Shot OOD Accuracy (k ∈ {5, 6} swaps, T=40)",
        error_kw={"elinewidth": 1.0, "ecolor": "#1a202c"},
    )
    ax1.axhline(12.5, color="#c53030", linestyle=":", linewidth=1.3, label="Random Chance (12.5%)")

    for idx_hl, col_hl in [(1, "#ebf8ff"), (7, "#fefcbf"), (8, "#fefcbf")]:
        ax1.axvspan(idx_hl - 0.46, idx_hl + 0.46, color=col_hl, alpha=0.55, zorder=0)

    for i, m_id in enumerate(on_id_m):
        if m_id >= 85.0 or i in (0, 7, 8, 9):
            top_bar = max(m_id + yerr_id[1, i], on_ood_m[i] + yerr_ood[1, i])
            txt = f"{m_id:.1f}%"
            x_txt = x[i] - w1 / 2
            if keys[i] == "deltaloop_b2":
                txt = f"{m_id:.1f}% (s0: 99.7%)"
                x_txt = x[i]
            elif m_id >= 99.0:
                x_txt = x[i]
            ax1.text(
                x_txt,
                top_bar + 1.8,
                txt,
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold" if m_id >= 99.0 else "normal",
                color="#1a202c",
            )

    ax1.set_ylim(0, 124)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.0, fontweight="bold")
    ax1.set_title(
        "(a) Task 1A-Online: Visible Container Swap Tracking (S₈ Group State, ID k≤4 & OOD k=5,6)",
        pad=8,
    )
    ax1.legend(loc="upper center", ncol=3, frameon=True, facecolor="white", framealpha=0.95)

    # --- Panel (b): Task 1A-Retro by Causal Arrival Order ---
    fw_m, fw_s = [], []
    mx_m, mx_s = [], []
    rv_m, rv_s = [], []
    for k in keys:
        runs = d_retro["runs"][k]
        mf, sf = _mean_std([r["id"]["order_breakdown_final_L"]["forward"] for r in runs])
        mm, sm = _mean_std([r["id"]["order_breakdown_final_L"]["mixed"] for r in runs])
        mr, sr = _mean_std([r["id"]["order_breakdown_final_L"]["reverse"] for r in runs])
        fw_m.append(mf)
        fw_s.append(sf)
        mx_m.append(mm)
        mx_s.append(sm)
        rv_m.append(mr)
        rv_s.append(sr)

    w2 = 0.26
    yerr_fw = _bounded_yerr(fw_m, fw_s)
    yerr_mx = _bounded_yerr(mx_m, mx_s)
    yerr_rv = _bounded_yerr(rv_m, rv_s)

    ax2.bar(
        x - w2,
        fw_m,
        w2,
        yerr=yerr_fw,
        capsize=2.5,
        color="#dd6b20",
        edgecolor="#7b341e",
        linewidth=0.7,
        label="Forward Causal Order (Post-Cue Occlusion: Obj→Box then Box→Zone)",
        error_kw={"elinewidth": 1.0, "ecolor": "#1a202c"},
    )
    ax2.bar(
        x,
        mx_m,
        w2,
        yerr=yerr_mx,
        capsize=2.5,
        color="#ed8936",
        edgecolor="#7b341e",
        linewidth=0.7,
        alpha=0.88,
        label="Mixed Causal Order (k ≥ 3 non-monotonic)",
        error_kw={"elinewidth": 1.0, "ecolor": "#1a202c"},
    )
    ax2.bar(
        x + w2,
        rv_m,
        w2,
        yerr=yerr_rv,
        capsize=2.5,
        color="#718096",
        edgecolor="#2d3748",
        linewidth=0.7,
        alpha=0.82,
        label="Reverse Order (Box→Zone before Obj→Box: write-time composable)",
        error_kw={"elinewidth": 1.0, "ecolor": "#1a202c"},
    )
    ax2.axhline(12.5, color="#c53030", linestyle=":", linewidth=1.3, label="Random Chance (12.5%)")

    for idx_hl in (7, 8):
        ax2.axvspan(idx_hl - 0.46, idx_hl + 0.46, color="#fefcbf", alpha=0.65, zorder=0)

    for i, mf in enumerate(fw_m):
        ax2.text(
            x[i] - w2,
            mf + yerr_fw[1, i] + 1.8,
            f"{mf:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.7,
            fontweight="bold" if mf >= 90.0 else "normal",
            color="#9c4221" if mf >= 90.0 else "#2d3748",
        )

    # Annotate exact Forward-only OOD k=5 vs k=6 breakdown on DeltaLoop bars
    b1_fk5 = float(np.mean([r["ood_hops"]["order_by_k_final_L"]["forward"]["k=5"] for r in d_retro["runs"]["deltaloop_b1"]]))
    b2_fk5 = float(np.mean([r["ood_hops"]["order_by_k_final_L"]["forward"]["k=5"] for r in d_retro["runs"]["deltaloop_b2"]]))
    b1_fk6 = float(np.mean([r["ood_hops"]["order_by_k_final_L"]["forward"]["k=6"] for r in d_retro["runs"]["deltaloop_b1"]]))
    b2_fk6 = float(np.mean([r["ood_hops"]["order_by_k_final_L"]["forward"]["k=6"] for r in d_retro["runs"]["deltaloop_b2"]]))
    ax2.text(
        7.5,
        109.5,
        f"OOD Fwd: k=5 {b2_fk5:.1f}–{b1_fk5:.1f}%, k=6 {b2_fk6:.1f}–{b1_fk6:.1f}%",
        ha="center",
        va="bottom",
        fontsize=7.2,
        fontweight="bold",
        color="#7b341e",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#fffaf0", "edgecolor": "#dd6b20", "alpha": 0.92},
    )

    ax2.set_ylim(0, 145)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.1, fontweight="bold")
    ax2.set_title(
        "(b) Task 1A-Retro: Multi-Hop Pointer Chaining (ID k<=4 & OOD k=5,6)",
        pad=8,
    )
    ax2.legend(loc="upper center", ncol=2, frameon=True, facecolor="white", framealpha=0.95)

    fig.tight_layout(h_pad=2.1)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig1_mechanistic_dissociation.{ext}", dpi=300)
    plt.close(fig)


def make_fig2_k_L_staircase(d_retro: Dict, out_dir: Path) -> None:
    panels = [
        ("static_query_loop_b2", "(a) Static-Query Loop (β∈(0,2))\n[Fixed q⁽⁰⁾, 436K Params]"),
        ("untied_deep_readout_b2", "(b) Untied Deep Readout (L=6)\n[Untied 1.63M (3.7×), 1-ep budget]"),
        ("deltaloop_b2", "(c) DeltaLoop (β∈(0,2), All-Order)\n[Weight-Tied q⁽ˡ⁾, 436K Params]"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), sharey=True)
    k_keys_id = ["k=1", "k=2", "k=3", "k=4"]
    k_keys_ood = ["k=5", "k=6"]
    row_labels = [
        "k = 1 (ID)",
        "k = 2 (ID)",
        "k = 3 (ID)",
        "k = 4 (ID)",
        "k = 5 (OOD)",
        "k = 6 (OOD)",
    ]
    col_labels = [f"L={l}" for l in range(1, 7)]

    im_last = None
    for idx, (arch_k, title) in enumerate(panels):
        ax = axes[idx]
        runs = d_retro["runs"][arch_k]
        mat = np.zeros((6, 6), dtype=np.float64)
        for r_i, kv in enumerate(k_keys_id):
            mat[r_i] = np.mean([r["id"]["matrix_k_L"][kv] for r in runs], axis=0)
        for r_i, kv in enumerate(k_keys_ood):
            mat[4 + r_i] = np.mean([r["ood_hops"]["matrix_k_L"][kv] for r in runs], axis=0)

        ax.grid(False)
        for sp in ("top", "right", "bottom", "left"):
            ax.spines[sp].set_visible(True)
            ax.spines[sp].set_linewidth(0.8)

        im = ax.imshow(mat, vmin=12.5, vmax=100.0, cmap="YlGnBu", aspect="auto")
        im_last = im

        for r_i in range(6):
            for c_i in range(6):
                val = mat[r_i, c_i]
                txt_color = "white" if val >= 62.0 else "#1a202c"
                is_diag = (idx == 2) and (r_i == c_i)
                ax.text(
                    c_i,
                    r_i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.3,
                    fontweight="bold" if (is_diag or val >= 80.0) else "normal",
                    color=txt_color,
                )
                if is_diag:
                    rect = mpatches.Rectangle(
                        (c_i - 0.48, r_i - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#f6ad55",
                        linewidth=2.0,
                    )
                    ax.add_patch(rect)

        ax.axhline(3.5, color="#e53e3e", linestyle="--", linewidth=1.8)
        if idx == 0:
            ax.text(
                -0.42,
                3.62,
                "Zero-Shot OOD (k > 4)",
                color="#c53030",
                fontsize=7.6,
                fontweight="bold",
                ha="left",
                va="top",
                bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "#feb2b2", "alpha": 0.9},
            )

        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(col_labels, fontweight="bold")
        ax.set_xlabel("Inference Readout Loop Depth L")
        ax.set_title(title, fontsize=10.2, pad=8)
        if idx == 0:
            ax.set_yticks(np.arange(6))
            ax.set_yticklabels(row_labels, fontweight="bold")
            ax.set_ylabel("Pointer Chain Hop Depth k")

    fig.subplots_adjust(right=0.90, wspace=0.12)
    cbar_ax = fig.add_axes([0.915, 0.16, 0.016, 0.72])
    cbar = fig.colorbar(im_last, cax=cbar_ax)
    cbar.set_label("All-Order Accuracy (%)", fontweight="bold")

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig2_k_L_staircase_heatmap.{ext}", dpi=300)
    plt.close(fig)


def make_fig3_halting_and_qk_alignment(d_retro: Dict, out_dir: Path) -> None:
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(14.6, 4.7), gridspec_kw={"width_ratios": [1.15, 0.98, 1.12]}
    )

    # --- Panel (a): Expected Halting Loops across k=1..6 ---
    ks = np.array([1, 2, 3, 4, 5, 6])
    for arch_k, label, color, marker in [
        ("deltaloop_b1", "DeltaLoop (β∈(0,1)) + 2-Stage Halt", "#2b6cb0", "o"),
        ("deltaloop_b2", "DeltaLoop (β∈(0,2)) + 2-Stage Halt", "#dd6b20", "s"),
    ]:
        runs = d_retro["runs"][arch_k]
        means, stds = [], []
        for kv in ("k=1", "k=2", "k=3", "k=4"):
            m, s = _mean_std([r["id"]["adaptive_halting"]["by_k"][kv]["mean_loops"] for r in runs])
            means.append(m)
            stds.append(s)
        for kv in ("k=5", "k=6"):
            m, s = _mean_std([r["ood_hops"]["adaptive_halting"]["by_k"][kv]["mean_loops"] for r in runs])
            means.append(m)
            stds.append(s)
        ax_a.errorbar(
            ks,
            means,
            yerr=stds,
            marker=marker,
            markersize=6,
            linewidth=2.1,
            capsize=3.5,
            color=color,
            label=label,
        )

    ax_a.plot(ks, ks, linestyle="--", color="#4a5568", linewidth=1.4, label="Ideal Oracle Depth (L = k)")
    ax_a.axvspan(4.5, 6.35, color="#fed7d7", alpha=0.35, label="Zero-Shot OOD Hops (k=5, 6)")
    ax_a.set_xlim(0.7, 6.3)
    ax_a.set_ylim(0.6, 6.4)
    ax_a.set_xticks(ks)
    ax_a.set_xticklabels(["1", "2", "3", "4", "5\n(OOD)", "6\n(OOD)"], fontweight="bold")
    ax_a.set_yticks([1, 2, 3, 4, 5, 6])
    ax_a.set_xlabel("True Relational Chain Depth k")
    ax_a.set_ylabel("Mean Executed Readout Loops E[Lₜ | k]")
    ax_a.set_title("(a) Adaptive Compute Scaling Across Hops")
    ax_a.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.94)

    # --- Panel (b): Stage-2 Halting vs RD-VLA Latent L2 Convergence (ID k=1..4) ---
    methods = ["Fixed L=6\n(Upper Bound)", "2-Stage Halt\n(Ours)", "Latent L2-Conv.\n(RD-VLA Style)"]
    xb = np.arange(len(methods))
    bw = 0.39

    b1_runs = d_retro["runs"]["deltaloop_b1"]
    b2_runs = d_retro["runs"]["deltaloop_b2"]

    b1_accs = [
        _mean_std([r["id_acc"] for r in b1_runs]),
        _mean_std([r["id"]["adaptive_halting"]["overall_acc"] for r in b1_runs]),
        _mean_std([r["id"]["latent_l2_convergence"]["overall_acc"] for r in b1_runs]),
    ]
    b2_accs = [
        _mean_std([r["id_acc"] for r in b2_runs]),
        _mean_std([r["id"]["adaptive_halting"]["overall_acc"] for r in b2_runs]),
        _mean_std([r["id"]["latent_l2_convergence"]["overall_acc"] for r in b2_runs]),
    ]
    b1_loops = [
        6.0,
        np.mean([r["id"]["adaptive_halting"]["mean_loops"] for r in b1_runs]),
        np.mean([r["id"]["latent_l2_convergence"]["mean_loops"] for r in b1_runs]),
    ]
    b2_loops = [
        6.0,
        np.mean([r["id"]["adaptive_halting"]["mean_loops"] for r in b2_runs]),
        np.mean([r["id"]["latent_l2_convergence"]["mean_loops"] for r in b2_runs]),
    ]

    b1_m = [m for m, _ in b1_accs]
    b1_yerr = _bounded_yerr(b1_m, [s for _, s in b1_accs])
    b2_m = [m for m, _ in b2_accs]
    b2_yerr = _bounded_yerr(b2_m, [s for _, s in b2_accs])

    ax_b.bar(
        xb - bw / 2,
        b1_m,
        bw,
        yerr=b1_yerr,
        capsize=3,
        color="#2b6cb0",
        edgecolor="#1a365d",
        label="DeltaLoop (β∈(0,1))",
    )
    ax_b.bar(
        xb + bw / 2,
        b2_m,
        bw,
        yerr=b2_yerr,
        capsize=3,
        color="#dd6b20",
        edgecolor="#7b341e",
        label="DeltaLoop (β∈(0,2))",
    )
    for i in range(3):
        ax_b.text(
            xb[i] - bw / 2,
            b1_m[i] + b1_yerr[1, i] + 1.8,
            f"{b1_m[i]:.1f}%\n({b1_loops[i]:.2f}L)",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold" if i == 1 else "normal",
            color="#1a365d",
        )
        ax_b.text(
            xb[i] + bw / 2,
            b2_m[i] + b2_yerr[1, i] + 1.8,
            f"{b2_m[i]:.1f}%\n({b2_loops[i]:.2f}L)",
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold" if i == 1 else "normal",
            color="#7b341e",
        )
    ax_b.axvspan(1 - 0.45, 1 + 0.45, color="#fefcbf", alpha=0.6, zorder=0)
    ax_b.set_ylim(0, 118)
    ax_b.set_yticks([0, 20, 40, 60, 80, 100])
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(methods, fontweight="bold", fontsize=8.3)
    ax_b.set_ylabel("ID Accuracy (%) & Mean Loops")
    ax_b.set_title("(b) Learned Halting (−56% Loops) vs. L2-Conv.")
    ax_b.legend(loc="lower left", fontsize=7.8, frameon=True, facecolor="white", framealpha=0.94)

    # --- Panel (c): 6 x 4 QK-Hop Top-1 Alignment Heatmap ---
    qmat = np.mean([r["id"]["qk_hop_top1_alignment_k4"] for r in b2_runs], axis=0)
    ax_c.grid(False)
    for sp in ("top", "right", "bottom", "left"):
        ax_c.spines[sp].set_visible(True)
        ax_c.spines[sp].set_linewidth(0.8)

    im_c = ax_c.imshow(qmat, vmin=0.0, vmax=88.0, cmap="magma_r", aspect="auto")
    for r_i in range(6):
        for c_i in range(4):
            val = qmat[r_i, c_i]
            ax_c.text(
                c_i,
                r_i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=8.4,
                fontweight="bold" if val >= 20.0 else "normal",
                color="white" if val >= 45.0 else "#1a202c",
            )
    ax_c.set_xticks(np.arange(4))
    ax_c.set_xticklabels(["Hop 1\n(x₀→x₁)", "Hop 2\n(x₁→x₂)", "Hop 3\n(x₂→x₃)", "Hop 4\n(x₃→Zone)"], fontweight="bold")
    ax_c.set_yticks(np.arange(6))
    ax_c.set_yticklabels([f"Loop l={l}" for l in range(1, 7)], fontweight="bold")
    ax_c.set_xlabel("True Historical Chain Edge (Random = 2.3%)")
    ax_c.set_title("(c) QK Top-1 Hop Alignment (k=4)")
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
    cbar_c.set_label("Top-1 Match (%)", fontweight="bold")

    fig.tight_layout(w_pad=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig3_halting_and_qk_alignment.{ext}", dpi=300)
    plt.close(fig)


def make_fig4_task1b_multiscale(d_1b: Dict, out_dir: Path) -> None:
    variants_order = [
        ("1_single_mh", "1. Unified 6H\n(Diverse fp32)"),
        ("1b_stacked_3layer", "1b. Stacked 3L\n(3×2H Resid.)"),
        ("2_3bank_same_fixed", "2. 3-Bank\n(Same Short τ)"),
        ("3_3bank_diverse_fixed", "3. 3-Bank\n(Div. Fixed τ)"),
        ("4_3bank_learnable", "4. 3-Bank\n(Learnable τ)"),
        ("5a_3bank_consolidation", "5a. 3-Bank +\nConsolidation"),
        ("5b_3bank_routing", "5b. 3-Bank +\nScale Router"),
        ("5c_3bank_full", "5c. 3-Bank\nFull (5a+5b)"),
    ]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.4, 5.1), gridspec_kw={"width_ratios": [2.05, 1.0]}
    )

    v_keys = [k for k, _ in variants_order]
    v_labels = [lbl for _, lbl in variants_order]
    x = np.arange(len(v_keys))
    w = 0.25

    metrics = [
        ("overall_acc", "Overall Accuracy (All 5 Subtypes)", "#2b6cb0"),
        ("mid_collide", "Mid-Lag Key Collision (200–400 steps)", "#38a169"),
        ("long_collide", "Long-Lag Key Collision (1500–2000 steps)", "#dd6b20"),
    ]

    for m_idx, (m_key, m_lbl, col) in enumerate(metrics):
        means, stds = [], []
        for vk in v_keys:
            m, s = _mean_std([r[m_key] for r in d_1b["runs"][vk]])
            means.append(m)
            stds.append(s)
        yerr = _bounded_yerr(means, stds)
        offset = (m_idx - 1) * w
        ax1.bar(
            x + offset,
            means,
            w,
            yerr=yerr,
            capsize=2.5,
            color=col,
            edgecolor="#1a202c",
            linewidth=0.6,
            label=m_lbl,
            error_kw={"elinewidth": 0.9, "ecolor": "#1a202c"},
        )
        if m_key == "overall_acc":
            for i, mv in enumerate(means):
                ax1.text(
                    x[i] + offset - (0.04 if i >= 3 else 0.0),
                    mv + yerr[1, i] + 1.6,
                    f"{mv:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold" if i in (0, 1) else "normal",
                    color="#1a365d",
                )

    ax1.axvspan(-0.46, 1.46, color="#e6fffa", alpha=0.65, zorder=0)
    ax1.axhline(12.5, color="#c53030", linestyle=":", linewidth=1.3, label="Chance / Latest-Obs Leak (~12.5%)")
    ax1.set_ylim(0, 125)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_xticks(x)
    ax1.set_xticklabels(v_labels, fontweight="bold", fontsize=8.0)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(
        "(a) Task 1B (T=2048, Strictly Iso-State = 1,536 Floats / 6,144 B Across All 8 Variants)",
        pad=8,
    )
    ax1.legend(loc="upper center", ncol=2, fontsize=7.9, frameon=True, facecolor="white", framealpha=0.95)

    # --- Panel (b): Softmax Scale-Router Collapse Heatmap (Variant 5c Mean Routing Weights π) ---
    q_scales = ["query_short", "query_mid", "query_long"]
    rmat_5c = np.zeros((3, 3), dtype=np.float64)
    for qi, qk in enumerate(q_scales):
        rmat_5c[qi] = np.mean([r["routing_matrix"][qk] for r in d_1b["runs"]["5c_3bank_full"]], axis=0) * 100.0

    ax2.grid(False)
    for sp in ("top", "right", "bottom", "left"):
        ax2.spines[sp].set_visible(True)
        ax2.spines[sp].set_linewidth(0.8)

    im2 = ax2.imshow(rmat_5c, vmin=0.0, vmax=100.0, cmap="OrRd", aspect="auto")
    for r_i in range(3):
        for c_i in range(3):
            val = rmat_5c[r_i, c_i]
            ax2.text(
                c_i,
                r_i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold" if (c_i == 2 or r_i == c_i) else "normal",
                color="white" if val >= 55.0 else "#1a202c",
            )

    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["Bank 0\n(Fast)", "Bank 1\n(Mid)", "Bank 2\n(Slow)"], fontweight="bold")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Short Query\n(10–30 lag)", "Mid Query\n(200–400 lag)", "Long Query\n(1500–2000 lag)"], fontweight="bold")
    ax2.set_xlabel("Routed Memory Bank Weight π (%) — Concentrates on Bank 2")
    ax2.set_title("(b) Diagnosis: Softmax Scale-Router Collapse (5c)", pad=8)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Router Weight π (%)", fontweight="bold")

    fig.tight_layout(w_pad=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig4_task1b_multiscale.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    configure_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    d_online = json.loads((SYNTH_DIR / "task1a_online.json").read_text())
    d_retro = json.loads((SYNTH_DIR / "task1a_retro.json").read_text())
    d_1b = json.loads((SYNTH_DIR / "task1b.json").read_text())

    make_fig1_mechanistic_dissociation(d_online, d_retro, FIG_DIR)
    make_fig2_k_L_staircase(d_retro, FIG_DIR)
    make_fig3_halting_and_qk_alignment(d_retro, FIG_DIR)
    make_fig4_task1b_multiscale(d_1b, FIG_DIR)

    generated = sorted(p.name for p in FIG_DIR.iterdir())
    print(f"Successfully generated {len(generated)} figure files in {FIG_DIR}:")
    for name in generated:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
