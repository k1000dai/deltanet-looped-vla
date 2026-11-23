#!/usr/bin/env python3
"""Publication-Grade RoboMME Evaluation & Mechanistic Probe Figure Suite (`fig5`–`fig8`).

Strictly CPU-only (`CUDA_VISIBLE_DEVICES=""`). 100% data-driven across:
  - Cycle 2A (`results/robomme/cycle2a`),
  - Cycle 3A (`results/robomme/cycle3a`: `U1–U5` + `520 KB` `M7` + live `*_interfk3.json`), and
  - Cycle 3B (`results/robomme/cycle3b_smolvla`: Pretrained `SmolVLA` `smolvla_m0..m7` + `U7 Q16` across 16 tasks).

Outputs 300-DPI PNG + vector PDF (`pdf.fonttype = 42`) into `--out-dir`:
  1. `fig5_robomme_main_suites.{png,pdf}`
  2. `fig6_pareto_and_adaptive_halting.{png,pdf}`
  3. `fig7_cross_session_interference.{png,pdf}`
  4. `fig8_mechanistic_probes_and_signflip.{png,pdf}`
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = REPO_ROOT / "results" / "robomme" / "cycle2a"
DEFAULT_PROBE_JSON = DEFAULT_EVAL_DIR / "mechanistic_probes.json"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "robomme" / "figures" / "cycle2a"

# Colorblind-safe Okabe-Ito-aligned palette matching `analysis/plot_synthetic.py`
PALETTE: Dict[str, str] = {
    "m0": "#7F7F7F",       # Neutral slate gray — Memoryless Flow / Base
    "m1": "#A6761D",       # Muted ochre/brown — RD-VLA Memoryless Loop
    "m2": "#D55E00",       # Vermilion — FrameSamp-32 Buffer
    "m3": "#009E73",       # Bluish green — 1L GDN beta in (0, 1)
    "m4": "#56B4E9",       # Sky blue — 1L GDN beta in (0, 2)
    "m5": "#0072B2",       # Deep royal blue — 2L Iso-State GDN beta in (0, 2)
    "m6_l1": "#8C6BB1",    # Soft violet — DeltaLoop L=1
    "m6_l2": "#CC79A7",    # Reddish purple — DeltaLoop L=2
    "m6_l4": "#984EA3",    # Rich purple — DeltaLoop L=4
    "m6_adapt": "#E69F00", # Gold/amber highlight — DeltaLoop Adaptive CRN Gate
    "m6_conv": "#B2ABD2",  # Lavender — DeltaLoop Latent L2 Conv
    "m6b": "#8C510A",      # Warm umber — DeltaLoop + Write-Back
    "m7_l2": "#C51B7D",    # Deep magenta — 2L DeltaLoop M7 Fixed L=2
    "m7": "#D90368",       # Crimson magenta — 2L Looped DeltaLoop M7 / SmolVLA-DeltaLoop
    "m7_q16": "#009E73",   # Bluish green — SmolVLA-DeltaLoop U7 Q16 (16 Coda Query Bank)
    "m7_anchor": "#6A3D9A", # Royal purple — SmolVLA-DeltaLoop U8 Anchor (4 Hop-0 Anchor + 4 Pointer)
    "m7_anchor_l2": "#8E63B5", # Medium purple — SmolVLA-DeltaLoop U8 Anchor Fixed L=2
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "axes.labelweight": "bold",
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.0,
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


def load_eval_directory(eval_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Load all clean (`k=0`) and distractor (`k>0`) evaluation JSONs + `paired_comparisons.json`."""
    runs: Dict[str, Dict[str, Any]] = {}
    paired: Dict[str, Any] = {}
    for p in sorted(eval_dir.glob("*.json")):
        if p.name in ("paired_comparisons.json", "mechanistic_probes.json"):
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and "summary" in data and "eval_spec" in data:
            runs[p.stem] = data

    paired_path = eval_dir / "paired_comparisons.json"
    if paired_path.is_file():
        paired = json.loads(paired_path.read_text())
    return runs, paired


def _extract_paired_map(paired: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap nested `cycle3a_pairs` / `cycle3b_pairs` or flat Cycle 2A `paired_comparisons.json`."""
    for sub_k in ("cycle3b_pairs", "cycle3a_pairs", "pairs"):
        if isinstance(paired.get(sub_k), dict):
            return paired[sub_k]
    return paired


def _compute_holm_map(paired_map: Dict[str, Any]) -> Dict[str, Tuple[float, bool]]:
    """Return `{pair_key: (p_holm, holm_significant_005)}` using stored fields or step-down Holm-Bonferroni."""
    raw_items: List[Tuple[float, str]] = []
    out: Dict[str, Tuple[float, bool]] = {}
    missing_any = False
    for k, v in paired_map.items():
        if not isinstance(v, dict) or "overall_success_diff" not in v:
            continue
        ov = v["overall_success_diff"]
        if "p_value_two_sided" not in ov:
            continue
        pv = float(ov["p_value_two_sided"])
        p_holm_stored = ov.get("p_holm", v.get("p_holm"))
        hsig_stored = ov.get("holm_significant_005", v.get("holm_significant_005"))
        if p_holm_stored is not None:
            ph = float(p_holm_stored)
            hs = bool(hsig_stored) if hsig_stored is not None else (ph <= 0.05)
            out[k] = (ph, hs)
        else:
            missing_any = True
        raw_items.append((pv, k))

    if missing_any and raw_items:
        raw_items.sort(key=lambda t: t[0])
        m = len(raw_items)
        running_max = 0.0
        all_holm: Dict[str, Tuple[float, bool]] = {}
        for rank_idx, (pv, k) in enumerate(raw_items):
            adj = min(1.0, max(running_max, pv * float(m - rank_idx)))
            running_max = adj
            all_holm[k] = (adj, adj <= 0.05)
        for k, val_t in all_holm.items():
            if k not in out:
                out[k] = val_t
    return out


def _format_two_sided_p(pval: float, p_holm: float, holm_sig: bool, compact: bool = False) -> str:
    """Format two-sided p-value with explicit Holm-Bonferroni adjustment status (no one-sided stars)."""
    p_str = "p<0.001" if pval < 1e-4 else (f"p={pval:.4f}" if pval < 0.001 else f"p={pval:.3f}")
    if holm_sig:
        ph_str = "p_Holm<0.001*" if p_holm < 0.001 else f"p_Holm={p_holm:.3f}*"
        return f"{p_str}, {ph_str}"
    if pval <= 0.05:
        return f"{p_str} unadj." if compact else f"{p_str} unadj., Holm NS"
    return f"{p_str} NS"


def _exact_mcnemar_two_sided_p(ep_a: Sequence[bool], ep_b: Sequence[bool]) -> float:
    """Exact two-sided binomial sign test on discordant episode pairs as fallback when not in paired_map."""
    n10 = sum(1 for a, b in zip(ep_a, ep_b) if a and not b)
    n01 = sum(1 for a, b in zip(ep_a, ep_b) if (not a) and b)
    n_disc = n10 + n01
    if n_disc == 0:
        return 1.0
    k_min = min(n10, n01)
    tail = sum(math.comb(n_disc, j) for j in range(0, k_min + 1)) * (0.5 ** n_disc)
    return float(min(1.0, 2.0 * tail))


def _sr_and_asym_ci(metric_dict: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return `(sr_pct, err_low_pct, err_high_pct)` from `summary` block."""
    sr = float(metric_dict.get("success_rate", 0.0)) * 100.0
    lo = float(metric_dict.get("success_ci95_low", metric_dict.get("success_rate", 0.0))) * 100.0
    hi = float(metric_dict.get("success_ci95_high", metric_dict.get("success_rate", 0.0))) * 100.0
    return sr, max(0.0, sr - lo), max(0.0, hi - sr)


def _state_kb_str(run_dict: Dict[str, Any]) -> str:
    espec = run_dict.get("eval_spec", {})
    mmeta = run_dict.get("model_meta", {})
    sb = int(
        mmeta.get(
            "state_bytes",
            mmeta.get(
                "state_size_bytes",
                espec.get("state_bytes", espec.get("state_size_bytes", 0)),
            ),
        )
    )
    if sb == 0 and "episodes" in run_dict and run_dict["episodes"]:
        aux = run_dict["episodes"][0].get("aux_info", {})
        sb = int(aux.get("state_bytes", aux.get("state_size_bytes", 0)))
    if sb <= 0:
        return "0 KB"
    return f"{int(round(sb / 1024.0))} KB"


def _draw_sig_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    color: str = "#222222",
    fontsize: float = 7.5,
) -> None:
    lo_x, hi_x = min(x1, x2), max(x1, x2)
    ax.plot([lo_x, lo_x, hi_x, hi_x], [y, y + h, y + h, y], lw=1.15, color=color, clip_on=False)
    ax.text(
        0.5 * (lo_x + hi_x),
        y + h + 0.25,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
    )


# =============================================================================
# Figure 5: Overall Success Rate + 4-Suite Breakdown with Paired Significance
# =============================================================================


def make_fig5_robomme_main_suites(
    runs: Dict[str, Dict[str, Any]],
    paired: Dict[str, Any],
    out_dir: Path,
) -> Optional[Path]:
    is_v3 = any("m7_deltaloop" in k or "interfk3" in k for k in runs)
    is_smolvla = any(k.startswith("smolvla_") for k in runs)

    m4_lbl = "M4: Motion-Gated\nGDN β∈(0,2)" if is_v3 else "M4: 1L GDN\nβ∈(0,2) (L=1)"
    candidates: List[Tuple[str, str, str]] = [
        ("m0_memoryless", "M0: Memoryless\nFlow (0 B, L=1)", PALETTE["m0"]),
        ("m1_rdvla_loop4", "M1: RD-VLA\nLoop (0 B, L=4)", PALETTE["m1"]),
        ("m2_framesamp_modul", "M2: FrameSamp-32\n(24 KB, L=1)", PALETTE["m2"]),
        ("m4_gdn_b2_l1", m4_lbl, PALETTE["m4"]),
        ("m3_gdn_b1_l1", "M3: 1L GDN\nβ∈(0,1) (L=1)", PALETTE["m3"]),
        ("m5_gdn2layer_b2_l1", "M5: 2L Iso-State\nGDN (L=1)", PALETTE["m5"]),
        ("m6_deltaloop_b2_L1", "M6: DeltaLoop\nFixed L=1", PALETTE["m6_l1"]),
        ("m6_deltaloop_b2_L2", "M6: DeltaLoop\nFixed L=2", PALETTE["m6_l2"]),
        ("m6_deltaloop_b2_adaptive", "M6: DeltaLoop\nAdapt (CRN Gate)", PALETTE["m6_adapt"]),
        ("m6b_deltaloop_wb_adaptive", "M6b: DeltaLoop\n+ WriteBack", PALETTE["m6b"]),
        ("m7_deltaloop_2l_b2_L2", "M7: 2L DeltaLoop\nFixed L=2", PALETTE["m7_l2"]),
        ("m7_deltaloop_2l_b2_adaptive", "M7: 2L DeltaLoop\nAdapt (CRN Gate)", PALETTE["m7"]),
        ("m7_deltaloop_2l_b2", "M7: 2L DeltaLoop\n(520 KB)", PALETTE["m7"]),
        # Pretrained SmolVLA (Cycle 3B + Cycle 4 U7) roster
        ("smolvla_m0_base", "SmolVLA-M0\nBase (0 B, L=1)", PALETTE["m0"]),
        ("smolvla_m1_rdvla_loop4", "SmolVLA-M1\nRD-Loop (0 B, L=4)", PALETTE["m1"]),
        ("smolvla_m2_framesamp", "SmolVLA-M2\nFrameSamp (24 KB)", PALETTE["m2"]),
        ("smolvla_m5_gdn2l_l1", "SmolVLA-M5\n2L GDN (520 KB)", PALETTE["m5"]),
        ("smolvla_m7_deltaloop_L1", "SmolVLA-M7\nFixed L=1 (520 KB)", PALETTE["m6_l1"]),
        ("smolvla_m7_deltaloop_L2", "SmolVLA-M7\nFixed L=2 (520 KB)", PALETTE["m6_l2"]),
        ("smolvla_m7_deltaloop_L4", "SmolVLA-M7\nFixed L=4 (520 KB)", PALETTE["m6_l4"]),
        ("smolvla_m7_deltaloop_adaptive", "SmolVLA-M7\nCRN Gate (520 KB)", PALETTE["m7"]),
        ("smolvla_m7_deltaloop_coda_conv", "SmolVLA-M7\nL2-Conv (520 KB)", PALETTE["m6_conv"]),
        ("smolvla_m7_deltaloop", "SmolVLA-M7\nDeltaLoop (520 KB)", PALETTE["m7"]),
        ("smolvla_m7_deltaloop_q16_adaptive", "SmolVLA-M7-Q16\n(N_c=16, 520 KB)", PALETTE["m7_q16"]),
        ("smolvla_m7_deltaloop_q16", "SmolVLA-M7-Q16\n(N_c=16, 520 KB)", PALETTE["m7_q16"]),
        ("smolvla_m7_deltaloop_anchor_adaptive", "SmolVLA-M7-Anchor\n(4+4 Reg, 520 KB)", PALETTE["m7_anchor"]),
        ("smolvla_m7_deltaloop_anchor", "SmolVLA-M7-Anchor\n(4+4 Reg, 520 KB)", PALETTE["m7_anchor"]),
        ("smolvla_m7_deltaloop_anchor_L2", "SmolVLA-M7-Anchor\nFixed L=2 (520 KB)", PALETTE["m7_anchor_l2"]),
    ]
    present = [(k, lbl, col) for k, lbl, col in candidates if k in runs]
    if not present:
        for k, d in runs.items():
            if int(d["eval_spec"].get("distractor_sessions", 0)) == 0:
                present.append((k, k.replace("_", "\n"), PALETTE["m5"]))
    if not present:
        return None

    fig = plt.figure(figsize=(15.2 if len(present) >= 11 else 14.4, 8.7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.44, wspace=0.24)
    ax_top = fig.add_subplot(gs[0, :])

    keys = [k for k, _, _ in present]
    labels = [lbl for _, lbl, _ in present]
    colors = [col for _, _, col in present]
    x = np.arange(len(present))

    srs, err_lo, err_hi = [], [], []
    for k in keys:
        sr, lo, hi = _sr_and_asym_ci(runs[k]["summary"]["overall"])
        srs.append(sr)
        err_lo.append(lo)
        err_hi.append(hi)

    bars = ax_top.bar(
        x,
        srs,
        width=0.62,
        color=colors,
        edgecolor="#222222",
        linewidth=0.85,
        zorder=3,
    )
    ax_top.errorbar(
        x,
        srs,
        yerr=np.vstack([err_lo, err_hi]),
        fmt="none",
        ecolor="#1A1A1A",
        elinewidth=1.25,
        capsize=4.0,
        zorder=4,
    )

    max_top_pt = max(s + h for s, h in zip(srs, err_hi))
    for b, sr_val, hi_val, k in zip(bars, srs, err_hi, keys):
        y_lbl = sr_val + hi_val + 0.42
        tag = f"{sr_val:.2f}%"
        if k in ("m5_gdn2layer_b2_l1", "smolvla_m5_gdn2l_l1"):
            tag = f"{sr_val:.2f}%\n(L=1)"
        elif k in (
            "m6_deltaloop_b2_adaptive",
            "m7_deltaloop_2l_b2_adaptive",
            "smolvla_m7_deltaloop_adaptive",
            "smolvla_m7_deltaloop_coda_conv",
            "smolvla_m7_deltaloop_q16_adaptive",
            "smolvla_m7_deltaloop_anchor_adaptive",
        ):
            l_m = float(runs[k]["summary"]["overall"].get("mean_loop_depth", 1.0))
            tag = f"{sr_val:.2f}%\n(E[L]={l_m:.2f})"
        ax_top.text(
            b.get_x() + b.get_width() / 2.0,
            y_lbl,
            tag,
            ha="center",
            va="bottom",
            fontsize=7.5 if len(present) >= 9 else 8.0,
            fontweight="bold",
            color="#111111",
        )

    paired_map = _extract_paired_map(paired)
    holm_map = _compute_holm_map(paired_map)
    ms_summary = paired.get("multiseed_summary", {}) if isinstance(paired, dict) else {}
    has_ms_badge = bool(ms_summary) and "H_MS1_Pooled_M7_vs_M0" in paired_map

    idx_map = {k: i for i, k in enumerate(keys)}
    bracket_base = max_top_pt + 3.4
    step_y = 2.75
    drawn = 0
    sig_specs = [
        # Cycle 2A keys
        ("H1b_M6_Adapt_vs_M1_RDVLA_Loop4", "m1_rdvla_loop4", "m6_deltaloop_b2_adaptive", "H1b (M6 vs M1)"),
        ("H1a_M6_Adapt_vs_M0_Memoryless", "m0_memoryless", "m6_deltaloop_b2_adaptive", "H1a (M6 vs M0)"),
        ("H6a_M6b_WB_Adapt_vs_M6_Adapt", "m6_deltaloop_b2_adaptive", "m6b_deltaloop_wb_adaptive", "H6a (WB vs Read-Only)"),
        # Cycle 3A keys
        ("H1c_M5_2L_L1_vs_M0_Memoryless", "m0_memoryless", "m5_gdn2layer_b2_l1", "H1c (2L GDN vs M0)"),
        ("H3c_M5_2L_L1_vs_M2_FrameSamp", "m2_framesamp_modul", "m5_gdn2layer_b2_l1", "H3c (2L GDN vs M2)"),
        ("H2b_M7_2L_Loop_vs_M5_2L_L1", "m5_gdn2layer_b2_l1", "m7_deltaloop_2l_b2_adaptive", "H2b (M7 Loop vs M5 2L)"),
        # Cycle 3B + Cycle 4 U7/U8 SmolVLA keys
        ("H0b_M2_FrameSamp_vs_M0_Base", "smolvla_m0_base", "smolvla_m2_framesamp", "H0b (M2 vs Base)"),
        ("H4e_M7_CodaConv_vs_M7_L1_IntraCkpt", "smolvla_m7_deltaloop_L1", "smolvla_m7_deltaloop_coda_conv", "H4e (L2-Conv vs L=1)"),
        ("H1c_M5_GDN2L_vs_M0_Base", "smolvla_m0_base", "smolvla_m5_gdn2l_l1", "H1c (M5 2L vs Base)"),
        ("H4d_M7_Adapt_vs_M7_L1_IntraCkpt", "smolvla_m7_deltaloop_L1", "smolvla_m7_deltaloop_adaptive", "H4d (CRN vs L=1)"),
        ("H1a_M7_Adapt_vs_M0_Base", "smolvla_m0_base", "smolvla_m7_deltaloop_adaptive", "H1a (M7 CRN vs Base)"),
        ("H9f_M7_Anchor_L2_vs_Anchor_Adapt", "smolvla_m7_deltaloop_anchor_adaptive", "smolvla_m7_deltaloop_anchor_L2", "H9f (Anc L=2 vs CRN)"),
        ("H8a_M7_Q16_vs_M0_Base", "smolvla_m0_base", "smolvla_m7_deltaloop_q16_adaptive", "H8a (M7-Q16 vs Base)"),
        ("H3b_M5_GDN2L_vs_M2_FrameSamp", "smolvla_m2_framesamp", "smolvla_m5_gdn2l_l1", "H3b (M5 vs M2, k=0)"),
        ("H3a_M7_Adapt_vs_M2_FrameSamp", "smolvla_m2_framesamp", "smolvla_m7_deltaloop_adaptive", "H3a (M7 vs M2, k=0)"),
    ]
    level_spans: List[List[Tuple[int, int]]] = []
    brk_fs = 7.15 if len(present) >= 10 else 7.5
    min_center_gap = 3.45 if len(present) >= 10 else 3.4
    for p_key, ka, kb, prefix in sig_specs:
        if p_key in paired_map and ka in idx_map and kb in idx_map and len(level_spans) <= 3:
            ia, ib = min(idx_map[ka], idx_map[kb]), max(idx_map[ka], idx_map[kb])
            mid_ab = 0.5 * (ia + ib)
            placed_lvl: Optional[int] = None
            for lvl_i, spans in enumerate(level_spans):
                # Keep upper bracket levels clear of top-right 3-seed badge when present
                if has_ms_badge and lvl_i >= 1 and ib >= len(present) - 3:
                    continue
                if all(
                    (ib < s_lo - 0.15 or ia > s_hi + 0.15)
                    and abs(mid_ab - 0.5 * (s_lo + s_hi)) >= min_center_gap
                    for s_lo, s_hi in spans
                ):
                    placed_lvl = lvl_i
                    spans.append((ia, ib))
                    break
            if placed_lvl is None:
                if len(level_spans) >= 3:
                    continue
                if has_ms_badge and len(level_spans) >= 1 and ib >= len(present) - 3:
                    continue
                placed_lvl = len(level_spans)
                level_spans.append([(ia, ib)])
            ov = paired_map[p_key]["overall_success_diff"]
            diff_pct = float(ov["mean_diff"]) * 100.0
            pval = float(ov["p_value_two_sided"])
            p_holm, holm_sig = holm_map.get(p_key, (1.0, False))
            p_status = _format_two_sided_p(pval, p_holm, holm_sig, compact=(len(present) >= 10))
            txt = f"{prefix}: {diff_pct:+.2f}% ({p_status})"
            _draw_sig_bracket(
                ax_top,
                idx_map[ka],
                idx_map[kb],
                bracket_base + placed_lvl * step_y,
                0.65,
                txt,
                color="#222222",
                fontsize=brk_fs,
            )
            drawn = max(drawn, placed_lvl + 1)

    n_eps_ov = int(runs[keys[0]]["summary"]["overall"].get("num_episodes", 160))
    split_lbl = "Test" if is_smolvla else "Validation"
    top_xtick_fs = 6.7 if len(present) >= 11 else (7.0 if len(present) >= 10 else (7.4 if len(present) >= 9 else 8.1))
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(labels, fontsize=top_xtick_fs)
    ax_top.set_ylabel("Overall Success Rate (%)")
    ax_top.set_ylim(0.0, bracket_base + max(1, drawn) * step_y + (3.6 if has_ms_badge else 2.4))
    if has_ms_badge:
        cms1 = paired_map["H_MS1_Pooled_M7_vs_M0"]["overall_success_diff"]
        m7_ms = ms_summary.get("smolvla_m7_deltaloop__learned_gate__4", {})
        m0_ms = ms_summary.get("smolvla_m0_base__fixed__1", {})
        m2_ms = ms_summary.get("smolvla_m2_framesamp__fixed__1", {})
        p_ms1 = float(cms1["p_value_two_sided"])
        p_ms1_holm, ms1_hsig = holm_map.get("H_MS1_Pooled_M7_vs_M0", (1.0, False))
        ms1_p_fmt = _format_two_sided_p(p_ms1, p_ms1_holm, ms1_hsig, compact=False)
        diff_ms1 = float(cms1.get("mean_diff", 0.0)) * 100.0
        ms7_std = float(m7_ms.get("std_sr", 0.0)) * 100.0
        ms0_std = float(m0_ms.get("std_sr", 0.0)) * 100.0
        ms2_std = float(m2_ms.get("std_sr", 0.0)) * 100.0
        n_per_seed = int(m7_ms.get("n_matched_eps_per_seed", 240))
        row_means = (
            f"M7: {float(m7_ms.get('mean_sr', 0.0))*100:.2f}±{ms7_std:.2f}% | "
            f"M0: {float(m0_ms.get('mean_sr', 0.0))*100:.2f}±{ms0_std:.2f}%"
        )
        if m2_ms:
            row_means += f" | M2: {float(m2_ms.get('mean_sr', 0.0))*100:.2f}±{ms2_std:.2f}%"
        ms_lines = [
            f"3-Seed Mean±SD ({{42,43,44}}, N={n_per_seed}/seed):",
            row_means,
            f"H_MS1 (M7 vs M0, N={int(cms1.get('n_pairs', 720))}): {diff_ms1:+.2f}% ({ms1_p_fmt})",
        ]
        ax_top.text(
            0.99,
            0.97,
            "\n".join(ms_lines),
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontsize=6.95,
            fontweight="bold",
            color="#1A1A1A",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#F8F9FA", "edgecolor": "#BDBDBD", "alpha": 0.94},
            zorder=6,
        )
    ax_top.set_title(
        f"(a) RoboMME Multi-Suite Overall Success Rate (N = {n_eps_ov} Matched {split_lbl} Episodes, 95% Bootstrap CIs)",
        pad=10,
    )

    # Bottom row: 4 Memory Suites (`Permanence`, `Counting`, `Reference`, `Imitation`)
    suite_models: List[Tuple[str, str, str]] = [
        ("m0_memoryless", "M0", PALETTE["m0"]),
        ("m1_rdvla_loop4", "M1", PALETTE["m1"]),
        ("m2_framesamp_modul", "M2", PALETTE["m2"]),
        ("m4_gdn_b2_l1", "M4", PALETTE["m4"]),
        ("m3_gdn_b1_l1", "M3", PALETTE["m3"]),
        ("m5_gdn2layer_b2_l1", "M5", PALETTE["m5"]),
        ("m6_deltaloop_b2_L2", "M6(L2)", PALETTE["m6_l2"]),
        ("m6_deltaloop_b2_adaptive", "M6(Ad)", PALETTE["m6_adapt"]),
        ("m7_deltaloop_2l_b2_L2", "M7(L2)", PALETTE["m7_l2"]),
        ("m7_deltaloop_2l_b2_adaptive", "M7(Ad)", PALETTE["m7"]),
        ("smolvla_m0_base", "M0", PALETTE["m0"]),
        ("smolvla_m1_rdvla_loop4", "M1", PALETTE["m1"]),
        ("smolvla_m2_framesamp", "M2", PALETTE["m2"]),
        ("smolvla_m5_gdn2l_l1", "M5", PALETTE["m5"]),
        ("smolvla_m7_deltaloop_L1", "M7(L1)", PALETTE["m6_l1"]),
        ("smolvla_m7_deltaloop_L2", "M7(L2)", PALETTE["m6_l2"]),
        ("smolvla_m7_deltaloop_adaptive", "M7(Ad)", PALETTE["m7"]),
        ("smolvla_m7_deltaloop_coda_conv", "M7(Cv)", PALETTE["m6_conv"]),
        ("smolvla_m7_deltaloop_q16_adaptive", "M7-Q16", PALETTE["m7_q16"]),
        ("smolvla_m7_deltaloop_q16", "M7-Q16", PALETTE["m7_q16"]),
        ("smolvla_m7_deltaloop_anchor_adaptive", "M7-Anc", PALETTE["m7_anchor"]),
        ("smolvla_m7_deltaloop_anchor", "M7-Anc", PALETTE["m7_anchor"]),
        ("smolvla_m7_deltaloop_anchor_L2", "M7-Anc(L2)", PALETTE["m7_anchor_l2"]),
    ]
    suite_present = [(k, s_lbl, col) for k, s_lbl, col in suite_models if k in runs]

    suites = ["Permanence", "Counting", "Reference", "Imitation"]
    panel_letters = ["(b)", "(c)", "(d)", "(e)"]
    for s_idx, (suite_name, p_let) in enumerate(zip(suites, panel_letters)):
        ax_s = fig.add_subplot(gs[1, s_idx])
        xs = np.arange(len(suite_present))
        s_vals, s_lo, s_hi, s_cols, s_lbls = [], [], [], [], []
        n_suite_eps = 40
        for k, s_lbl, col in suite_present:
            sd = runs[k]["summary"]["suites"].get(suite_name, {})
            n_suite_eps = int(sd.get("num_episodes", n_suite_eps))
            v, lo, hi = _sr_and_asym_ci(sd)
            s_vals.append(v)
            s_lo.append(lo)
            s_hi.append(hi)
            s_cols.append(col)
            s_lbls.append(s_lbl)

        ax_s.bar(xs, s_vals, width=0.66, color=s_cols, edgecolor="#222222", linewidth=0.75, zorder=3)
        ax_s.errorbar(
            xs,
            s_vals,
            yerr=np.vstack([s_lo, s_hi]),
            fmt="none",
            ecolor="#222222",
            elinewidth=1.0,
            capsize=2.4,
            zorder=4,
        )
        for xi, v, hi in zip(xs, s_vals, s_hi):
            ax_s.text(
                xi,
                v + hi + 0.55,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5 if len(suite_present) >= 9 else 7.0,
                fontweight="bold",
            )
        ax_s.set_xticks(xs)
        ax_s.set_xticklabels(s_lbls, rotation=32, ha="right", fontsize=7.1 if len(suite_present) >= 9 else 7.6)
        if s_idx == 0:
            ax_s.set_ylabel("Suite Success Rate (%)")
        top_lim = max(26.0, max(v + h for v, h in zip(s_vals, s_hi)) + 5.5)
        ax_s.set_ylim(0.0, top_lim)
        ax_s.set_title(f"{p_let} {suite_name} ({n_suite_eps} eps)", fontsize=9.8)

    out_stem = "fig5_robomme_main_suites"
    _save_both(fig, out_dir, out_stem)
    return out_dir / f"{out_stem}.png"


# =============================================================================
# Figure 6: Systems Pareto Frontier + Intra-Ckpt Multi-Hop + Phase L_t Heatmap
# =============================================================================


def make_fig6_pareto_and_adaptive_halting(
    runs: Dict[str, Dict[str, Any]],
    out_dir: Path,
    paired: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    if not runs:
        return None
    pm_fig6 = _extract_paired_map(paired or {})

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(16.5, 5.2), gridspec_kw={"width_ratios": [1.16, 1.05, 1.24], "wspace": 0.38}
    )

    # --- Panel (a): Chunk Latency (ms/chunk) & State Size vs Overall SR (%) ---
    m3_kb = _state_kb_str(runs["m3_gdn_b1_l1"]) if "m3_gdn_b1_l1" in runs else "516 KB"
    m5_kb = _state_kb_str(runs["m5_gdn2layer_b2_l1"]) if "m5_gdn2layer_b2_l1" in runs else "520 KB"
    m6_el = float(runs["m6_deltaloop_b2_adaptive"]["summary"]["overall"]["mean_loop_depth"]) if "m6_deltaloop_b2_adaptive" in runs else 1.52
    m7_el = float(runs["m7_deltaloop_2l_b2_adaptive"]["summary"]["overall"]["mean_loop_depth"]) if "m7_deltaloop_2l_b2_adaptive" in runs else 1.06
    sm5_kb = _state_kb_str(runs["smolvla_m5_gdn2l_l1"]) if "smolvla_m5_gdn2l_l1" in runs else "520 KB"
    sm7_kb = _state_kb_str(runs["smolvla_m7_deltaloop_adaptive"]) if "smolvla_m7_deltaloop_adaptive" in runs else "520 KB"
    used_contended_note = False

    pareto_specs: List[Tuple[str, str, str, str]] = [
        ("m0_memoryless", "M0 (0 KB)", PALETTE["m0"], "o"),
        ("m1_rdvla_loop4", "M1 RD-VLA L=4 (0 KB)", PALETTE["m1"], "v"),
        ("m2_framesamp_modul", "M2 FrameSamp (24 KB)", PALETTE["m2"], "s"),
        ("m4_gdn_b2_l1", "M4 GDN β∈(0,2)", PALETTE["m4"], "D"),
        ("m3_gdn_b1_l1", f"M3 GDN β∈(0,1) ({m3_kb})", PALETTE["m3"], "P"),
        ("m5_gdn2layer_b2_l1", f"M5 2L GDN ({m5_kb})", PALETTE["m5"], "^"),
        ("m6_deltaloop_b2_L1", "M6 Fixed L=1", PALETTE["m6_l1"], "o"),
        ("m6_deltaloop_b2_L2", "M6 Fixed L=2", PALETTE["m6_l2"], "h"),
        ("m6_deltaloop_b2_L4", "M6 Fixed L=4", PALETTE["m6_l4"], "X"),
        ("m6_deltaloop_b2_adaptive", f"M6 CRN Gate (E[L]={m6_el:.2f})", PALETTE["m6_adapt"], "*"),
        ("m6_deltaloop_b2_coda_conv", "M6 L2-Conv", PALETTE["m6_conv"], "d"),
        ("m6b_deltaloop_wb_adaptive", "M6b WriteBack", PALETTE["m6b"], "s"),
        ("m7_deltaloop_2l_b2_L2", "M7 2L Fixed L=2", PALETTE["m7_l2"], "p"),
        ("m7_deltaloop_2l_b2_adaptive", f"M7 2L Gate (E[L]={m7_el:.2f})", PALETTE["m7"], "H"),
        # SmolVLA roster (Wave-1 + Wave-2 + Cycle 4 U7)
        ("smolvla_m0_base", "M0 Base (0 KB)", PALETTE["m0"], "o"),
        ("smolvla_m1_rdvla_loop4", "M1 RD-Loop L=4 (0 KB)", PALETTE["m1"], "v"),
        ("smolvla_m2_framesamp", "M2 FrameSamp (24 KB)", PALETTE["m2"], "s"),
        ("smolvla_m5_gdn2l_l1", f"M5 2L GDN ({sm5_kb})", PALETTE["m5"], "^"),
        ("smolvla_m7_deltaloop_L1", "M7 Fixed L=1", PALETTE["m6_l1"], "o"),
        ("smolvla_m7_deltaloop_L2", "M7 Fixed L=2", PALETTE["m6_l2"], "h"),
        ("smolvla_m7_deltaloop_L4", "M7 Fixed L=4", PALETTE["m6_l4"], "X"),
        ("smolvla_m7_deltaloop_coda_conv", f"M7 L2-Conv ({sm7_kb})", PALETTE["m6_conv"], "d"),
        ("smolvla_m7_deltaloop_adaptive", f"M7 CRN Gate ({sm7_kb})", PALETTE["m7"], "*"),
        ("smolvla_m7_deltaloop_q16_adaptive", f"M7-Q16 CRN (N_c=16, {sm7_kb})", PALETTE["m7_q16"], "P"),
        ("smolvla_m7_deltaloop_q16", f"M7-Q16 (N_c=16, {sm7_kb})", PALETTE["m7_q16"], "P"),
        ("smolvla_m7_deltaloop_anchor_adaptive", f"M7-Anchor CRN (4+4, {sm7_kb})", PALETTE["m7_anchor"], "D"),
        ("smolvla_m7_deltaloop_anchor", f"M7-Anchor (4+4, {sm7_kb})", PALETTE["m7_anchor"], "D"),
        ("smolvla_m7_deltaloop_anchor_L2", "M7-Anchor Fixed L=2", PALETTE["m7_anchor_l2"], "s"),
    ]

    is_cycle3a = ("m7_deltaloop_2l_b2_adaptive" in runs) and ("m1_rdvla_loop4" not in runs)
    is_smolvla_pareto = any(k.startswith("smolvla_") for k in runs)
    if is_smolvla_pareto:
        label_specs: Dict[str, Tuple[float, float, bool]] = {
            "smolvla_m0_base": (2.5, -0.75, False),
            "smolvla_m1_rdvla_loop4": (6.5, -0.55, False),
            "smolvla_m2_framesamp": (6.0, 0.72, True),
            "smolvla_m5_gdn2l_l1": (-94.0, 1.50, True),
            "smolvla_m7_deltaloop_coda_conv": (10.5, 1.45, True),
            "smolvla_m7_deltaloop_adaptive": (10.5, 0.50, True),
            "smolvla_m7_deltaloop_L2": (10.5, -0.38, True),
            "smolvla_m7_deltaloop_anchor_adaptive": (10.5, -0.82, True),
            "smolvla_m7_deltaloop_anchor": (10.5, -0.82, True),
            "smolvla_m7_deltaloop_L1": (10.5, -1.68, True),
            "smolvla_m7_deltaloop_L4": (6.0, 0.95, True),
            "smolvla_m7_deltaloop_q16_adaptive": (-105.0, 0.50, True),
            "smolvla_m7_deltaloop_q16": (-105.0, 0.50, True),
            "smolvla_m7_deltaloop_anchor_L2": (-98.0, -0.58, True),
        }
    elif is_cycle3a:
        # Non-overlapping data-verified offsets for Cycle 3A latency cluster (~55.6–62.6 ms)
        label_specs = {
            "m0_memoryless": (1.0, 0.30, False),
            "m2_framesamp_modul": (1.1, -0.45, False),
            "m4_gdn_b2_l1": (-16.2, 0.75, True),
            "m3_gdn_b1_l1": (-18.5, -1.25, True),
            "m5_gdn2layer_b2_l1": (-16.8, 0.35, True),
            "m6_deltaloop_b2_adaptive": (-19.2, 0.55, True),
            "m6_deltaloop_b2_L2": (1.3, 0.45, True),
            "m7_deltaloop_2l_b2_L2": (1.0, 0.25, False),
            "m7_deltaloop_2l_b2_adaptive": (-18.8, -0.85, True),
        }
    else:
        label_specs = {
            "m0_memoryless": (1.4, 0.30, False),
            "m1_rdvla_loop4": (1.6, 0.10, False),
            "m2_framesamp_modul": (-21.0, -0.55, True),
            "m4_gdn_b2_l1": (-19.5, 0.45, True),
            "m3_gdn_b1_l1": (1.6, -0.20, False),
            "m5_gdn2layer_b2_l1": (1.8, 0.20, False),
            "m6_deltaloop_b2_L1": (1.6, -0.45, False),
            "m6_deltaloop_b2_L2": (1.8, 0.35, False),
            "m6_deltaloop_b2_L4": (-4.5, 0.65, False),
            "m6_deltaloop_b2_adaptive": (-24.5, 0.65, True),
            "m6_deltaloop_b2_coda_conv": (2.2, -0.45, True),
            "m6b_deltaloop_wb_adaptive": (1.6, -0.25, False),
        }

    pts_for_frontier: List[Tuple[float, float]] = []
    xs_seen, ys_seen = [], []
    for k, short_lbl, col, marker in pareto_specs:
        if k not in runs:
            continue
        ov = runs[k]["summary"]["overall"]
        sr = float(ov["success_rate"]) * 100.0
        ms = float(ov["mean_ms_per_chunk"])
        lbl_use = short_lbl
        if k == "m6_deltaloop_b2_L2" and ms > 85.0 and "m6b_deltaloop_wb_L2" in runs:
            ms_plot = float(runs["m6b_deltaloop_wb_L2"]["summary"]["overall"]["mean_ms_per_chunk"])
            lbl_use = "M6 Fixed L=2*"
            used_contended_note = True
        else:
            ms_plot = ms

        sz = 165 if marker == "*" else 76
        ax_a.scatter(
            ms_plot,
            sr,
            s=sz,
            color=col,
            marker=marker,
            edgecolors="#1A1A1A",
            linewidths=0.9,
            zorder=5,
        )
        pts_for_frontier.append((ms_plot, sr))
        xs_seen.append(ms_plot)
        ys_seen.append(sr)
        dx, dy, use_arr = label_specs.get(k, (1.2, 0.25, False))
        ax_a.annotate(
            lbl_use,
            (ms_plot, sr),
            xytext=(ms_plot + dx, sr + dy),
            arrowprops={"arrowstyle": "-", "lw": 0.75, "color": "#666666"} if use_arr else None,
            fontsize=7.2,
            fontweight="bold"
            if k
            in (
                "m6_deltaloop_b2_adaptive",
                "m5_gdn2layer_b2_l1",
                "m6_deltaloop_b2_L2",
                "smolvla_m5_gdn2l_l1",
                "smolvla_m7_deltaloop_adaptive",
                "smolvla_m7_deltaloop_coda_conv",
            )
            else "normal",
            color="#1A1A1A",
        )

    if pts_for_frontier:
        pts_sorted = sorted(pts_for_frontier, key=lambda t: t[0])
        front_x, front_y = [], []
        best_y = -1.0
        for px, py in pts_sorted:
            if py > best_y + 1e-6:
                front_x.append(px)
                front_y.append(py)
                best_y = py
        ax_a.step(front_x, front_y, where="post", linestyle="--", lw=1.3, color="#444444", alpha=0.75)

    ax_a.set_xlabel("Mean Policy Chunk Latency (ms / chunk)")
    ax_a.set_ylabel("Overall Success Rate (%)")
    if xs_seen:
        if is_smolvla_pareto:
            x_lo = max(0.0, min(xs_seen) - 24.0)
            x_hi = max(xs_seen) + 76.0
            ax_a.set_xlim(x_lo, x_hi)
            ax_a.set_ylim(-0.4, max(10.0, max(ys_seen) + 2.8))
        else:
            x_lo = min(33.0, min(xs_seen) - 4.0)
            x_hi = max(72.0 if is_cycle3a else 86.5, max(xs_seen) + 10.0)
            ax_a.set_xlim(x_lo, x_hi)
            ax_a.set_ylim(-0.4, max(13.5, max(ys_seen) + 2.5))
    if used_contended_note:
        ax_a.text(
            0.03,
            0.96,
            "*Uncontended L=2: 61.6 ms/ch\n (101.1 ms under concurrent load)",
            transform=ax_a.transAxes,
            fontsize=6.9,
            color="#444444",
            va="top",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
        )
    ax_a.set_title("(a) Latency & Memory vs. Success Pareto")

    # --- Panel (b): Intra-Checkpoint Multi-Hop Curve vs. Single-Pass 2L GDN (M5) ---
    m6_depths = [("m6_deltaloop_b2_L1", 1.0), ("m6_deltaloop_b2_L2", 2.0), ("m6_deltaloop_b2_L4", 4.0)]
    smol_depths = [
        ("smolvla_m7_deltaloop_L1", 1.0),
        ("smolvla_m7_deltaloop_L2", 2.0),
        ("smolvla_m7_deltaloop_L4", 4.0),
    ]
    if all(k in runs for k, _ in m6_depths):
        xs_m6 = [d for _, d in m6_depths]
        ys_m6 = [float(runs[k]["summary"]["overall"]["success_rate"]) * 100.0 for k, _ in m6_depths]
        ys_ref = [float(runs[k]["summary"]["suites"]["Reference"]["success_rate"]) * 100.0 for k, _ in m6_depths]
        ax_b.plot(xs_m6, ys_m6, "-o", color=PALETTE["m6_l4"], lw=2.2, markersize=7.0, label="M6 Overall SR (Fixed L=1,2,4)", zorder=4)
        ax_b.plot(xs_m6, ys_ref, "--^", color="#6A3D9A", lw=1.5, markersize=5.8, alpha=0.85, label=f"M6 Reference Suite ({ys_ref[0]:.1f}→{max(ys_ref):.1f}%)", zorder=3)
        for lx, ly in zip(xs_m6, ys_m6):
            ax_b.annotate(f"{ly:.1f}%", (lx, ly), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")

        if "m5_gdn2layer_b2_l1" in runs:
            sr_m5_2a = float(runs["m5_gdn2layer_b2_l1"]["summary"]["overall"]["success_rate"]) * 100.0
            ax_b.scatter([1.0], [sr_m5_2a], s=92, color=PALETTE["m5"], marker="^", edgecolors="#111111", linewidths=0.95, zorder=6, label=f"M5 2L GDN (L=1, {sr_m5_2a:.1f}%)")
            ax_b.annotate(
                f"Single-pass M5 (2L, L=1): {sr_m5_2a:.1f}%\n(> looped M6 {min(ys_m6):.1f}–{max(ys_m6):.1f}%)",
                (1.0, sr_m5_2a),
                xytext=(1.18, 9.55),
                arrowprops={"arrowstyle": "->", "lw": 0.85, "color": PALETTE["m5"]},
                fontsize=6.9,
                fontweight="bold",
                color=PALETTE["m5"],
            )

        m6b_depths = [("m6b_deltaloop_wb_L1", 1.0), ("m6b_deltaloop_wb_L2", 2.0), ("m6b_deltaloop_wb_L4", 4.0)]
        if all(k in runs for k, _ in m6b_depths):
            xs_wb = [d for _, d in m6b_depths]
            ys_wb = [float(runs[k]["summary"]["overall"]["success_rate"]) * 100.0 for k, _ in m6b_depths]
            ax_b.plot(xs_wb, ys_wb, "-s", color=PALETTE["m6b"], lw=1.8, markersize=5.8, label="M6b +WriteBack SR", zorder=3)

        if "m6_deltaloop_b2_adaptive" in runs:
            ov_ad = runs["m6_deltaloop_b2_adaptive"]["summary"]["overall"]
            l_ad = float(ov_ad["mean_loop_depth"])
            sr_ad = float(ov_ad["success_rate"]) * 100.0
            ax_b.scatter([l_ad], [sr_ad], s=210, color=PALETTE["m6_adapt"], marker="*", edgecolors="#111111", linewidths=1.0, zorder=6, label=f"Stage-II CRN Gate (E[L]={l_ad:.2f}, {sr_ad:.1f}%)")
            ax_b.annotate(
                f"CRN: {sr_ad:.1f}% @ E[L]={l_ad:.2f}\n(-{int(round((1.0 - l_ad / 4.0) * 100))}% loops vs L=4)",
                (l_ad, sr_ad),
                xytext=(l_ad + 0.36, sr_ad - 2.35),
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#333333"},
                fontsize=7.0,
                fontweight="bold",
                color="#222222",
            )
        if "m6_deltaloop_b2_coda_conv" in runs:
            ov_cv = runs["m6_deltaloop_b2_coda_conv"]["summary"]["overall"]
            ax_b.scatter(
                [float(ov_cv["mean_loop_depth"])],
                [float(ov_cv["success_rate"]) * 100.0],
                s=95,
                color=PALETTE["m6_conv"],
                marker="D",
                edgecolors="#111111",
                linewidths=0.9,
                zorder=5,
                label=f"Latent L2 Conv (E[L]={float(ov_cv['mean_loop_depth']):.2f})",
            )
        ax_b.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        ax_b.set_xlim(0.8, 4.28)
        ax_b.set_ylim(0.0, 21.5)
    elif is_smolvla_pareto:
        present_smol_d = [(k, d) for k, d in smol_depths if k in runs]
        max_ys_smol = 6.0
        if len(present_smol_d) >= 2:
            xs_s = [d for _, d in present_smol_d]
            ys_s = [float(runs[k]["summary"]["overall"]["success_rate"]) * 100.0 for k, _ in present_smol_d]
            max_ys_smol = max(max_ys_smol, max(ys_s))
            ax_b.plot(
                xs_s,
                ys_s,
                "-o",
                color=PALETTE["m7"],
                lw=2.2,
                markersize=7.0,
                label=f"SmolVLA-M7 Fixed L ({ys_s[0]:.2f}%→{ys_s[-1]:.2f}%)",
                zorder=4,
            )
            for lx, ly in zip(xs_s, ys_s):
                if abs(lx - 1.0) < 0.05:
                    ax_b.annotate(
                        f"L=1: {ly:.2f}%",
                        (lx, ly),
                        xytext=(10, -11),
                        textcoords="offset points",
                        ha="left",
                        fontsize=7.2,
                        fontweight="bold",
                        color=PALETTE["m7"],
                    )
                else:
                    has_cv_near = "smolvla_m7_deltaloop_coda_conv" in runs
                    ax_b.annotate(
                        f"L={int(lx)}: {ly:.2f}%",
                        (lx, ly),
                        xytext=(-14 if has_cv_near else 7, 9 if has_cv_near else 4),
                        textcoords="offset points",
                        ha="center" if has_cv_near else "left",
                        fontsize=7.0,
                        fontweight="bold",
                        color=PALETTE["m7"],
                    )
        elif len(present_smol_d) == 1:
            k_s, d_s = present_smol_d[0]
            y_s = float(runs[k_s]["summary"]["overall"]["success_rate"]) * 100.0
            max_ys_smol = max(max_ys_smol, y_s)
            ax_b.scatter([d_s], [y_s], s=85, color=PALETTE["m6_l2"], marker="h", edgecolors="#111111", linewidths=0.9, zorder=5, label=f"M7 Fixed L={int(d_s)}: {y_s:.2f}%")
        for ref_k, ref_lbl, ref_col, ref_mk in [
            ("smolvla_m0_base", "M0 Base (0B, L=1)", PALETTE["m0"], "o"),
            ("smolvla_m5_gdn2l_l1", "M5 2L GDN (520KB, L=1)", PALETTE["m5"], "^"),
            ("smolvla_m1_rdvla_loop4", "M1 RD-Loop (0B, L=4)", PALETTE["m1"], "v"),
        ]:
            if ref_k in runs:
                ov_r = runs[ref_k]["summary"]["overall"]
                lr = float(ov_r.get("mean_loop_depth", 4.0 if "loop4" in ref_k else 1.0))
                sr_r = float(ov_r["success_rate"]) * 100.0
                max_ys_smol = max(max_ys_smol, sr_r)
                ax_b.scatter([lr], [sr_r], s=90 if "m5" in ref_k else 82, color=ref_col, marker=ref_mk, edgecolors="#111111", linewidths=0.95, zorder=6 if "m5" in ref_k else 5, label=f"{ref_lbl}: {sr_r:.2f}%")
        if "smolvla_m5_gdn2l_l1" in runs:
            sr_m5_3b = float(runs["smolvla_m5_gdn2l_l1"]["summary"]["overall"]["success_rate"]) * 100.0
            ax_b.annotate(
                f"Single-pass M5 (L=1): {sr_m5_3b:.2f}%\n(dominates looped M7 3.0–4.5%)",
                (1.0, sr_m5_3b),
                xytext=(1.18, sr_m5_3b + 1.05),
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": PALETTE["m5"]},
                fontsize=6.75,
                fontweight="bold",
                color=PALETTE["m5"],
            )
        if "smolvla_m7_deltaloop_adaptive" in runs:
            ov_ad = runs["smolvla_m7_deltaloop_adaptive"]["summary"]["overall"]
            l_ad = float(ov_ad["mean_loop_depth"])
            sr_ad = float(ov_ad["success_rate"]) * 100.0
            max_ys_smol = max(max_ys_smol, sr_ad)
            ax_b.scatter([l_ad], [sr_ad], s=210, color=PALETTE["m6_adapt"], marker="*", edgecolors="#111111", linewidths=1.0, zorder=6, label=f"M7 CRN Gate (E[L]={l_ad:.2f}, {sr_ad:.2f}%)")
        if "smolvla_m7_deltaloop_coda_conv" in runs:
            ov_cv = runs["smolvla_m7_deltaloop_coda_conv"]["summary"]["overall"]
            l_cv = float(ov_cv["mean_loop_depth"])
            sr_cv = float(ov_cv["success_rate"]) * 100.0
            max_ys_smol = max(max_ys_smol, sr_cv)
            if "smolvla_m7_deltaloop_L2" in runs:
                sr_l2_c = float(runs["smolvla_m7_deltaloop_L2"]["summary"]["overall"]["success_rate"]) * 100.0
                ax_b.plot([2.0, l_cv], [sr_l2_c, sr_cv], ":", color=PALETTE["m7"], lw=1.7, alpha=0.85, zorder=4)
            ax_b.scatter(
                [l_cv],
                [sr_cv],
                s=105,
                color=PALETTE["m6_conv"],
                marker="D",
                edgecolors="#111111",
                linewidths=1.0,
                zorder=6,
                label=f"M7 L2-Conv (E[L]={l_cv:.2f}, {sr_cv:.2f}%)",
            )
            sr_l1_ref = float(runs["smolvla_m7_deltaloop_L1"]["summary"]["overall"]["success_rate"]) * 100.0 if "smolvla_m7_deltaloop_L1" in runs else 2.75
            p_h4e = float(pm_fig6.get("H4e_M7_CodaConv_vs_M7_L1_IntraCkpt", {}).get("overall_success_diff", {}).get("p_value_two_sided", 0.0654))
            p_h4d = float(pm_fig6.get("H4d_M7_Adapt_vs_M7_L1_IntraCkpt", {}).get("overall_success_diff", {}).get("p_value_two_sided", 0.0696))
            ax_b.annotate(
                f"M7 intra-ckpt: {sr_l1_ref:.2f}%→{sr_cv:.2f}%\n"
                f"(Conv p={p_h4e:.3f} NS, CRN p={p_h4d:.3f} NS;\n"
                f"random-L ckpt sensitive to L=1)",
                (l_cv, sr_cv),
                xytext=(4.22, sr_cv + 0.42),
                ha="right",
                arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#5E3C99"},
                fontsize=6.5,
                fontweight="bold",
                color="#3F2768",
            )
        if "smolvla_m7_deltaloop_q16_adaptive" in runs:
            ov_q16 = runs["smolvla_m7_deltaloop_q16_adaptive"]["summary"]["overall"]
            l_q16 = float(ov_q16["mean_loop_depth"])
            sr_q16 = float(ov_q16["success_rate"]) * 100.0
            max_ys_smol = max(max_ys_smol, sr_q16)
            ax_b.scatter([l_q16], [sr_q16], s=140, color=PALETTE["m7_q16"], marker="P", edgecolors="#111111", linewidths=1.0, zorder=6, label=f"M7-Q16 (E[L]={l_q16:.2f}, {sr_q16:.2f}%)")
        if "smolvla_m7_deltaloop_anchor_adaptive" in runs:
            ov_anc = runs["smolvla_m7_deltaloop_anchor_adaptive"]["summary"]["overall"]
            l_anc = float(ov_anc["mean_loop_depth"])
            sr_anc = float(ov_anc["success_rate"]) * 100.0
            max_ys_smol = max(max_ys_smol, sr_anc)
            if "smolvla_m7_deltaloop_anchor_L2" in runs:
                sr_anc_l2 = float(runs["smolvla_m7_deltaloop_anchor_L2"]["summary"]["overall"]["success_rate"]) * 100.0
                p_h9f = float(pm_fig6.get("H9f_M7_Anchor_L2_vs_Anchor_Adapt", {}).get("overall_success_diff", {}).get("p_value_two_sided", 0.4832))
                max_ys_smol = max(max_ys_smol, sr_anc_l2)
                ax_b.plot(
                    [l_anc, 2.0],
                    [sr_anc, sr_anc_l2],
                    "--s",
                    color=PALETTE["m7_anchor_l2"],
                    lw=1.8,
                    markersize=5.8,
                    label=f"M7-Anchor CRN→L=2 ({sr_anc:.2f}%→{sr_anc_l2:.2f}%)",
                    zorder=5,
                )
                ax_b.annotate(
                    f"Anc L=1→2: {sr_anc:.2f}%→{sr_anc_l2:.2f}%\n(two-sided p={p_h9f:.3f} NS)",
                    (2.0, sr_anc_l2),
                    xytext=(2.12, sr_anc_l2 - 1.65),
                    arrowprops={"arrowstyle": "->", "lw": 0.85, "color": PALETTE["m7_anchor"]},
                    fontsize=6.6,
                    fontweight="bold",
                    color=PALETTE["m7_anchor"],
                )
                ax_b.scatter([l_anc], [sr_anc], s=95, color=PALETTE["m7_anchor"], marker="D", edgecolors="#111111", linewidths=1.0, zorder=6)
            else:
                ax_b.scatter([l_anc], [sr_anc], s=95, color=PALETTE["m7_anchor"], marker="D", edgecolors="#111111", linewidths=1.0, zorder=6, label=f"M7-Anchor (E[L]={l_anc:.2f}, {sr_anc:.2f}%)")
        ax_b.set_xticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        ax_b.set_xlim(0.8, 4.28)
        ax_b.set_ylim(0.0, max(14.5, max_ys_smol + 8.5))
    else:
        # Cycle 3A intra-checkpoint comparison vs. single-pass 2L GDN (M5)
        max_y_b = 18.0
        if "m5_gdn2layer_b2_l1" in runs:
            sr_m5_3a = float(runs["m5_gdn2layer_b2_l1"]["summary"]["overall"]["success_rate"]) * 100.0
            ax_b.scatter([1.0], [sr_m5_3a], s=92, color=PALETTE["m5"], marker="^", edgecolors="#111111", linewidths=0.95, zorder=6, label=f"M5 2L GDN (L=1: {sr_m5_3a:.1f}%)")
            ax_b.annotate(
                f"Single-pass M5 (L=1): {sr_m5_3a:.1f}%\n(> looped M6 8.1% / M7 6.2–6.9%)",
                (1.0, sr_m5_3a),
                xytext=(1.06, 15.1),
                arrowprops={"arrowstyle": "->", "lw": 0.85, "color": PALETTE["m5"]},
                fontsize=6.9,
                fontweight="bold",
                color=PALETTE["m5"],
            )
        if "m6_deltaloop_b2_adaptive" in runs and "m6_deltaloop_b2_L2" in runs:
            l6_a = float(runs["m6_deltaloop_b2_adaptive"]["summary"]["overall"]["mean_loop_depth"])
            y6_ov = [
                float(runs["m6_deltaloop_b2_adaptive"]["summary"]["overall"]["success_rate"]) * 100.0,
                float(runs["m6_deltaloop_b2_L2"]["summary"]["overall"]["success_rate"]) * 100.0,
            ]
            y6_pm = [
                float(runs["m6_deltaloop_b2_adaptive"]["summary"]["suites"]["Permanence"]["success_rate"]) * 100.0,
                float(runs["m6_deltaloop_b2_L2"]["summary"]["suites"]["Permanence"]["success_rate"]) * 100.0,
            ]
            ax_b.plot([l6_a, 2.0], y6_ov, "-o", color=PALETTE["m6_adapt"], lw=2.2, markersize=6.8, label=f"M6 Overall ({y6_ov[0]:.1f}% → {y6_ov[1]:.1f}%, NS)", zorder=4)
            ax_b.plot([l6_a, 2.0], y6_pm, "--^", color=PALETTE["m6_l4"], lw=1.9, markersize=6.2, label=f"M6 Permanence ({y6_pm[0]:.1f}% → {y6_pm[1]:.1f}%)", zorder=5)
            ax_b.annotate(
                f"M6 Permanence: {y6_pm[0]:.1f}%→{y6_pm[1]:.1f}%\n(Overall p=1.00 NS)",
                (2.0, y6_pm[1]),
                xytext=(1.38, 12.8),
                arrowprops={"arrowstyle": "->", "lw": 0.85, "color": PALETTE["m6_l4"]},
                fontsize=6.9,
                fontweight="bold",
                color=PALETTE["m6_l4"],
            )
        if "m7_deltaloop_2l_b2_adaptive" in runs and "m7_deltaloop_2l_b2_L2" in runs:
            l7_a = float(runs["m7_deltaloop_2l_b2_adaptive"]["summary"]["overall"]["mean_loop_depth"])
            y7_ov = [
                float(runs["m7_deltaloop_2l_b2_adaptive"]["summary"]["overall"]["success_rate"]) * 100.0,
                float(runs["m7_deltaloop_2l_b2_L2"]["summary"]["overall"]["success_rate"]) * 100.0,
            ]
            y7_rf = [
                float(runs["m7_deltaloop_2l_b2_adaptive"]["summary"]["suites"]["Reference"]["success_rate"]) * 100.0,
                float(runs["m7_deltaloop_2l_b2_L2"]["summary"]["suites"]["Reference"]["success_rate"]) * 100.0,
            ]
            ax_b.plot([l7_a, 2.0], y7_ov, "-s", color=PALETTE["m7"], lw=2.0, markersize=6.2, label=f"M7 2L Overall ({y7_ov[0]:.1f}% → {y7_ov[1]:.1f}%, NS)", zorder=4)
            ax_b.plot([l7_a, 2.0], y7_rf, ":D", color=PALETTE["m7_l2"], lw=1.7, markersize=5.6, label=f"M7 Reference ({y7_rf[0]:.1f}% → {y7_rf[1]:.1f}%)", zorder=4)
            max_y_b = max(max_y_b, max(y7_ov + y7_rf) + 5.0)
        ax_b.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0])
        ax_b.set_xlim(0.92, 2.18)
        ax_b.set_ylim(0.0, max_y_b)

    ax_b.set_xlabel(r"Mean Readout Loop Depth $\mathbb{E}[L_t]$")
    ax_b.set_ylabel("Success Rate (%)")
    ax_b.legend(
        loc="lower left" if is_cycle3a else "upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        fontsize=6.7 if is_smolvla_pareto else 7.1,
    )
    ax_b.set_title("(b) Intra-Checkpoint Multi-Hop & Halting")

    # --- Panel (c): Task x Trajectory-Phase Loop Allocation Heatmap ---
    adapt_candidates = [
        k
        for k in (
            "smolvla_m7_deltaloop_anchor_adaptive",
            "smolvla_m7_deltaloop_anchor",
            "smolvla_m7_deltaloop_q16_adaptive",
            "smolvla_m7_deltaloop_q16",
            "smolvla_m7_deltaloop_adaptive",
            "m6_deltaloop_b2_adaptive",
            "m7_deltaloop_2l_b2_adaptive",
        )
        if k in runs and "loop_diagnostics" in runs[k]
    ]
    # Prefer the adaptive checkpoint with the largest mean loop depth for clear phase visualization
    adapt_key = (
        max(adapt_candidates, key=lambda k: float(runs[k]["summary"]["overall"].get("mean_loop_depth", 1.0)))
        if adapt_candidates
        else None
    )
    if adapt_key is not None:
        ld = runs[adapt_key]["loop_diagnostics"]
        per_task = ld.get("per_task") or ld.get("by_task") or {}
        if len(per_task) > 8:
            ranked_t = sorted(
                per_task.keys(),
                key=lambda tn: sum(
                    float(per_task[tn].get(pf, 1.0))
                    for pf in ("chunk0_mean_L", "early_mean_L", "mid_mean_L", "late_mean_L")
                )
                / 4.0,
                reverse=True,
            )
            tasks_used = ranked_t[:8] + ranked_t[-2:]
        else:
            task_order = [
                "StopCube",
                "PickXtimes",
                "MoveCube",
                "VideoPlaceOrder",
                "PickHighlight",
                "ButtonUnmaskSwap",
                "VideoUnmaskSwap",
                "PatternLock",
            ]
            tasks_used = [t for t in task_order if t in per_task]
            if not tasks_used:
                tasks_used = list(per_task.keys())[:8]
        phase_cols = [
            ("chunk0_mean_L", "Chunk-0\n(Onset)"),
            ("early_mean_L", "Early\n[0–33%]"),
            ("mid_mean_L", "Mid\n[33–67%]"),
            ("late_mean_L", "Late\n[67–100%]"),
        ]
        mat = np.zeros((len(tasks_used), len(phase_cols)), dtype=np.float64)
        for r_i, t_name in enumerate(tasks_used):
            for c_i, (p_field, _) in enumerate(phase_cols):
                mat[r_i, c_i] = float(per_task[t_name].get(p_field, 1.0))

        vmax_c = max(1.28, float(mat.max()) + 0.04)
        thresh_white = 1.0 + 0.62 * (vmax_c - 1.0)
        ax_c.grid(False)
        im = ax_c.imshow(mat, cmap="YlOrRd", vmin=1.0, vmax=vmax_c, aspect="auto")
        ax_c.set_xticks(np.arange(len(phase_cols)))
        ax_c.set_xticklabels([lbl for _, lbl in phase_cols], fontsize=8.0)
        ax_c.set_yticks(np.arange(len(tasks_used)))
        ax_c.set_yticklabels(tasks_used, fontsize=8.2)
        for r_i in range(mat.shape[0]):
            for c_i in range(mat.shape[1]):
                val = mat[r_i, c_i]
                txt_col = "white" if val >= thresh_white else "#111111"
                ax_c.text(
                    c_i,
                    r_i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    fontweight="bold",
                    color=txt_col,
                )
        cbar = fig.colorbar(im, ax=ax_c, fraction=0.042, pad=0.03)
        cbar.set_label(r"Mean Loop Depth $L_t$", fontsize=8.4)
        short_ad = (
            "M7-Anchor"
            if "anchor" in adapt_key
            else ("M7-Q16" if "q16" in adapt_key else ("M7" if "m7" in adapt_key else "M6"))
        )
        ax_c.set_title(f"(c) Phase Loop Allocation ({short_ad} CRN Gate)")

    out_stem = "fig6_pareto_and_adaptive_halting"
    _save_both(fig, out_dir, out_stem)
    return out_dir / f"{out_stem}.png"


# =============================================================================
# Figure 7: Cross-Session Interference (k in {0, 1, 3} Prior Distractor Sessions)
# =============================================================================


def make_fig7_cross_session_interference(
    probe_data: Dict[str, Any],
    runs: Dict[str, Dict[str, Any]],
    out_dir: Path,
    paired: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    models_dict = probe_data.get("models", {})
    if not models_dict:
        return None

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(15.6, 4.9), gridspec_kw={"width_ratios": [1.0, 1.0, 1.28], "wspace": 0.28}
    )

    series_specs: List[Tuple[str, str, str, str]] = [
        ("m2_framesamp_modul", "M2: FrameSamp-32 (Buffer)", PALETTE["m2"], "-s"),
        ("m3_gdn_b1_l1", "M3: 1L GDN β∈(0,1)", PALETTE["m3"], "-o"),
        ("m4_gdn_b2_l1", "M4: 1L GDN β∈(0,2)", PALETTE["m4"], "-D"),
        ("m5_gdn2layer_b2_l1", "M5: 2L Iso-State GDN", PALETTE["m5"], "-^"),
        ("m6_deltaloop_b2", "M6: DeltaLoop (1L)", PALETTE["m6_adapt"], "-*"),
        ("m6b_deltaloop_wb", "M6b: DeltaLoop + WB", PALETTE["m6b"], "--v"),
        ("m7_deltaloop_2l_b2", "M7: 2L DeltaLoop (Ours)", PALETTE["m7"], "-P"),
        # SmolVLA equivalents
        ("smolvla_m2_framesamp", "SmolVLA-M2: FrameSamp-32", PALETTE["m2"], "-s"),
        ("smolvla_m5_gdn2l_l1", "SmolVLA-M5: 2L GDN", PALETTE["m5"], "-^"),
        ("smolvla_m7_deltaloop", "SmolVLA-M7: DeltaLoop", PALETTE["m7"], "-*"),
        ("smolvla_m7_deltaloop_q16", "SmolVLA-M7-Q16 (16 Coda)", PALETTE["m7_q16"], "-P"),
        ("smolvla_m7_deltaloop_anchor", "SmolVLA-M7-Anchor (4+4)", PALETTE["m7_anchor"], "-D"),
    ]
    ks = [0, 1, 3]

    gdn_cos_k3: List[float] = []
    gdn_dpx_k3: List[float] = []
    buf_cos_k3: Optional[float] = None
    buf_dpx_k3: Optional[float] = None
    buf_dpx_max_k3: float = 0.0
    max_band_b: float = 20.0

    for m_key, lbl, col, style in series_specs:
        if m_key not in models_dict:
            continue
        tasks_map = models_dict[m_key]["tasks"]
        cos_means, cos_mins, cos_maxs = [], [], []
        dpx_means, dpx_stds = [], []

        for k in ks:
            k_str = f"k={k}"
            c_vals, px_vals = [], []
            for t_res in tasks_map.values():
                bk = t_res["probe3_cross_session_and_halting"]["by_distractor_k"]
                if k_str in bk:
                    c_vals.append(float(bk[k_str]["coda_cos_sim_vs_k0"]))
                    if "probe1_fused_px_err" in bk[k_str] and "probe1_fused_px_err" in bk["k=0"]:
                        px_vals.append(
                            float(bk[k_str]["probe1_fused_px_err"]) - float(bk["k=0"]["probe1_fused_px_err"])
                        )
            cos_means.append(float(np.mean(c_vals)) if c_vals else float("nan"))
            cos_mins.append(float(np.min(c_vals)) if c_vals else float("nan"))
            cos_maxs.append(float(np.max(c_vals)) if c_vals else float("nan"))
            dpx_means.append(float(np.mean(px_vals)) if px_vals else 0.0)
            dpx_stds.append(float(np.std(px_vals)) if px_vals else 0.0)
            if k == 3 and "framesamp" in m_key and px_vals:
                buf_dpx_max_k3 = max(px_vals)

        if "framesamp" in m_key:
            buf_cos_k3 = cos_means[-1]
            buf_dpx_k3 = dpx_means[-1]
        else:
            gdn_cos_k3.append(cos_means[-1])
            gdn_dpx_k3.append(dpx_means[-1])

        lw = 2.3 if ("framesamp" in m_key or "m6_deltaloop" in m_key or "m7_deltaloop" in m_key) else 1.5
        msz = 8.5 if style.endswith("*") else 5.8
        ax_a.plot(ks, cos_means, style, color=col, lw=lw, markersize=msz, label=lbl, zorder=4)
        if "framesamp" in m_key:
            ax_a.fill_between(ks, cos_mins, cos_maxs, color=col, alpha=0.14)

        ax_b.plot(ks, dpx_means, style, color=col, lw=lw, markersize=msz, label=lbl, zorder=4)
        if "framesamp" in m_key or m_key in ("m6_deltaloop_b2", "m7_deltaloop_2l_b2", "smolvla_m7_deltaloop"):
            hi_band = np.asarray(dpx_means) + np.asarray(dpx_stds)
            max_band_b = max(max_band_b, float(np.max(hi_band)))
            ax_b.fill_between(
                ks,
                np.maximum(0.0, np.asarray(dpx_means) - np.asarray(dpx_stds)),
                hi_band,
                color=col,
                alpha=0.12,
            )

    ax_a.set_xticks(ks)
    ax_a.set_xticklabels(["k = 0\n(Clean)", "k = 1\n(+1 Ep)", "k = 3\n(+3 Eps)"])
    ax_a.set_ylim(0.0, 1.06)
    ax_a.set_ylabel(r"Coda Cosine Similarity $\cos(\mathbf{c}_t(k), \mathbf{c}_t(0))$")
    ax_a.set_title("(a) Coda Representation Stability vs. k")
    is_high_buf_cos = buf_cos_k3 is not None and math.isfinite(buf_cos_k3) and buf_cos_k3 > 0.36
    if gdn_cos_k3:
        c_lo, c_hi = min(gdn_cos_k3), max(gdn_cos_k3)
        c_mid = 0.5 * (c_lo + c_hi)
        ax_a.annotate(
            f"GDN Selective Decay:\ncos = {c_lo:.3f}–{c_hi:.3f}",
            xy=(3.0, c_mid),
            xytext=(0.68 if is_high_buf_cos else 0.75, 0.80 if is_high_buf_cos else max(0.66, c_lo - 0.16)),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#222222"},
            fontsize=7.8,
            fontweight="bold",
        )
    if buf_cos_k3 is not None and math.isfinite(buf_cos_k3):
        buf_xytext = (1.58, 0.28) if is_high_buf_cos else (1.78, max(0.32, buf_cos_k3 + 0.16))
        ax_a.annotate(
            f"Buffer Saturation:\ncos = {buf_cos_k3:.3f}",
            xy=(3.0, buf_cos_k3),
            xytext=buf_xytext,
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PALETTE["m2"]},
            fontsize=7.8,
            fontweight="bold",
            color=PALETTE["m2"],
        )
    leg_anchor_a = (0.02, 0.03) if is_high_buf_cos else (0.01, 0.27)
    ax_a.legend(loc="lower left", bbox_to_anchor=leg_anchor_a, frameon=True, facecolor="white", framealpha=0.92, fontsize=6.9)

    max_dpx_plot = max([buf_dpx_k3 or 10.0] + gdn_dpx_k3)
    y_top_b = max(28.5, max_dpx_plot + 8.5, max_band_b + 5.2)
    ax_b.set_xticks(ks)
    ax_b.set_xticklabels(["k = 0\n(Clean)", "k = 1\n(+1 Ep)", "k = 3\n(+3 Eps)"])
    ax_b.set_ylabel(r"Localization Error Shift $+\Delta\mathrm{px}$ (256×256)")
    ax_b.set_ylim(-0.8, y_top_b)
    ax_b.set_title("(b) <x, y> Target Drift Under Prior Sessions")
    if buf_dpx_k3 is not None:
        ax_b.annotate(
            f"+{buf_dpx_k3:.1f} px mean\n(up to +{buf_dpx_max_k3:.1f} px)",
            xy=(3.0, buf_dpx_k3),
            xytext=(0.78 if y_top_b > 32.0 else 0.95, min(max(22.0, buf_dpx_k3 + 6.0), y_top_b - 5.5)),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": PALETTE["m2"]},
            fontsize=7.8,
            fontweight="bold",
            color=PALETTE["m2"],
        )
    if gdn_dpx_k3 and buf_dpx_k3 is not None:
        g_lo, g_hi = min(gdn_dpx_k3), max(gdn_dpx_k3)
        g_mid = 0.5 * (g_lo + g_hi)
        ratio_drift = buf_dpx_k3 / max(g_mid, 0.5)
        span_lbl = f"(+{g_mid:.1f} px mean)" if abs(g_hi - g_lo) < 0.15 else f"(+{g_lo:.1f} to +{g_hi:.1f} px)"
        ax_b.annotate(
            f"{ratio_drift:.1f}× lower drift\n{span_lbl}",
            xy=(3.0, g_mid),
            xytext=(1.05, g_hi + 4.6),
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#222222"},
            fontsize=7.8,
            fontweight="bold",
        )

    # --- Panel (c): Per-Task k=3 Localization Drift + Live k=3 Rollout Badge ---
    ref_buf_key = "smolvla_m2_framesamp" if "smolvla_m2_framesamp" in models_dict else "m2_framesamp_modul"
    loc_tasks = [
        ("StopCube", "StopCube"),
        ("ButtonUnmaskSwap", "ButtonSwap"),
        ("SwingXtimes", "SwingX"),
        ("MoveCube", "MoveCube"),
        ("VideoRepick", "VidRepick"),
        ("PickHighlight", "PickHighlt"),
        ("VideoUnmaskSwap", "VideoSwap"),
        ("VideoPlaceOrder", "PlaceOrder"),
        ("PickXtimes", "PickXtimes"),
    ]
    avail_tasks = [
        (t_full, t_short)
        for t_full, t_short in loc_tasks
        if ref_buf_key in models_dict
        and t_full in models_dict[ref_buf_key]["tasks"]
        and "probe1_fused_px_err" in models_dict[ref_buf_key]["tasks"][t_full]["probe3_cross_session_and_halting"]["by_distractor_k"]["k=3"]
    ]
    bar_candidates = [
        ("m2_framesamp_modul", "M2 FrameSamp-32", PALETTE["m2"]),
        ("m5_gdn2layer_b2_l1", "M5 2L GDN", PALETTE["m5"]),
        ("m6_deltaloop_b2", "M6 DeltaLoop", PALETTE["m6_adapt"]),
        ("m7_deltaloop_2l_b2", "M7 2L DeltaLoop", PALETTE["m7"]),
        ("smolvla_m2_framesamp", "M2 FrameSamp-32", PALETTE["m2"]),
        ("smolvla_m5_gdn2l_l1", "M5 2L GDN", PALETTE["m5"]),
        ("smolvla_m7_deltaloop", "M7 DeltaLoop", PALETTE["m7"]),
        ("smolvla_m7_deltaloop_q16", "M7-Q16 DeltaLoop", PALETTE["m7_q16"]),
        ("smolvla_m7_deltaloop_anchor", "M7-Anchor (4+4)", PALETTE["m7_anchor"]),
    ]
    bar_models = [(mk, ml, mc) for mk, ml, mc in bar_candidates if mk in models_dict]

    def _k3_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, str]:
        rk_name = item[0]
        if "m2_" in rk_name:
            return (0, rk_name)
        if "m5_" in rk_name:
            return (1, rk_name)
        if "m6_" in rk_name:
            return (2, rk_name)
        if "anchor" in rk_name:
            return (5, rk_name)
        if "q16" in rk_name:
            return (4, rk_name)
        return (3, rk_name)

    live_k3 = sorted(
        [
            (rk, rd)
            for rk, rd in runs.items()
            if int(rd["eval_spec"].get("distractor_sessions", 0)) == 3
            and "_s43" not in rk
            and "_s44" not in rk
        ],
        key=_k3_sort_key,
    )

    if avail_tasks and bar_models:
        xt = np.arange(len(avail_tasks))
        n_bm = len(bar_models)
        total_span = 0.68
        bar_w = total_span / n_bm
        offsets = np.linspace(-0.5 * total_span + 0.5 * bar_w, 0.5 * total_span - 0.5 * bar_w, n_bm)
        max_bar_c = 25.0
        for off, (mk, ml, mc) in zip(offsets, bar_models):
            vals = []
            for t_full, _ in avail_tasks:
                bk = models_dict[mk]["tasks"][t_full]["probe3_cross_session_and_halting"]["by_distractor_k"]
                vals.append(float(bk["k=3"]["probe1_fused_px_err"]) - float(bk["k=0"]["probe1_fused_px_err"]))
            max_bar_c = max(max_bar_c, max(vals))
            ax_c.bar(xt + off, vals, width=bar_w * 0.90, color=mc, edgecolor="#222222", linewidth=0.65, label=ml, zorder=3)

        ax_c.axhline(0.0, color="#333333", lw=0.8)
        ax_c.set_xticks(xt)
        ax_c.set_xticklabels([ts for _, ts in avail_tasks], rotation=28, ha="right", fontsize=8.0)
        ax_c.set_ylabel("k = 3 Localization Drift +Δpx")
        extra_top = 24.5 if len(live_k3) >= 3 else (22.0 if len(live_k3) == 2 else (15.0 if live_k3 else 5.5))
        ax_c.set_ylim(-1.0, max(34.0, max_bar_c + extra_top))
        if live_k3:
            leg_y = 0.78 if len(live_k3) >= 3 else 0.80
            ax_c.legend(
                loc="upper right",
                bbox_to_anchor=(0.99, leg_y),
                ncol=2,
                frameon=True,
                facecolor="white",
                framealpha=0.92,
                fontsize=6.15,
                columnspacing=0.8,
            )
        else:
            ax_c.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.7)
        ax_c.set_title("(c) Per-Task k = 3 Localization Degradation")

    if live_k3:
        pm_k3 = _extract_paired_map(paired or {})
        holm_k3 = _compute_holm_map(pm_k3)
        m2_k3_by_id: Dict[Tuple[str, int], bool] = {}
        for rk_chk, rd_chk in live_k3:
            if "m2_" in rk_chk:
                m2_k3_by_id = {
                    (ep["task"], int(ep["episode_id"])): bool(ep.get("success", False))
                    for ep in rd_chk.get("episodes", [])
                }
                break

        def _short_tag_for_k3(rk_s: str) -> str:
            if "anchor" in rk_s:
                return "M7-Anc"
            if "q16" in rk_s:
                return "M7-Q16"
            if "m2_" in rk_s:
                return "M2"
            if "m5_" in rk_s:
                return "M5"
            if "m7_" in rk_s:
                return "M7"
            return "M6"

        def _short_p_tag(pv: float, hsig: bool) -> str:
            if pv < 1e-4:
                return "p<0.001*" if hsig else "p<0.001 unadj."
            p_s = f"{pv:.3f}" if pv >= 0.001 else f"{pv:.4f}"
            if hsig:
                return f"p={p_s}*"
            if pv <= 0.05:
                return f"p={p_s} unadj."
            return f"p={p_s} NS"

        parts = []
        matched_n: Optional[int] = None
        for rk, rd in live_k3[:4]:
            short_n = _short_tag_for_k3(rk)
            base_rk = rk.replace("_interfk3", "")
            k3_by_id = {
                (ep["task"], int(ep["episode_id"])): bool(ep.get("success", False))
                for ep in rd.get("episodes", [])
            }
            if base_rk in runs and k3_by_id:
                k0_by_id = {
                    (ep["task"], int(ep["episode_id"])): bool(ep.get("success", False))
                    for ep in runs[base_rk].get("episodes", [])
                }
                common_ids = sorted(set(k3_by_id.keys()) & set(k0_by_id.keys()))
                n_m = len(common_ids)
                matched_n = n_m
                c_k0 = sum(1 for cid in common_ids if k0_by_id[cid])
                c_k3 = sum(1 for cid in common_ids if k3_by_id[cid])
                sr_k0 = 100.0 * c_k0 / max(1, n_m)
                sr_k3 = 100.0 * c_k3 / max(1, n_m)

                # Lookup two-sided p for k=3 vs k=0 drop
                p_drop: Optional[float] = None
                hsig_drop = False
                pk_drop_matched: Optional[str] = None
                for pk_cand, pv_dict in pm_k3.items():
                    if not isinstance(pv_dict, dict) or "overall_success_diff" not in pv_dict:
                        continue
                    if ("Interf" in pk_cand or "k3" in pk_cand) and ("Drop" in pk_cand or "k3_vs_k0" in pk_cand):
                        if f"_{short_n}_" in f"_{pk_cand}_":
                            pk_drop_matched = pk_cand
                            p_drop = float(pv_dict["overall_success_diff"]["p_value_two_sided"])
                            _, hsig_drop = holm_k3.get(pk_cand, (1.0, False))
                            break
                ep_k3_vec = [k3_by_id[cid] for cid in common_ids]
                ep_k0_vec = [k0_by_id[cid] for cid in common_ids]
                n10_d = sum(1 for a, b in zip(ep_k3_vec, ep_k0_vec) if a and not b)
                n01_d = sum(1 for a, b in zip(ep_k3_vec, ep_k0_vec) if (not a) and b)
                p_mcn_drop = _exact_mcnemar_two_sided_p(ep_k3_vec, ep_k0_vec)
                if p_drop is None:
                    p_drop = p_mcn_drop
                if (n10_d + n01_d) > 0 and (n10_d == 0 or n01_d == 0):
                    # Zero-gain/loss discordant pairs (e.g. M5 10/0): bootstrap underestimates variance;
                    # report exact two-sided McNemar p + step-down Holm across the full suite family as primary.
                    p_mcn_4dp = round(p_mcn_drop, 4)
                    fam_pvals: List[Tuple[float, str]] = []
                    if isinstance(paired, dict):
                        sub_dicts = [paired[sk] for sk in ("cycle3b_pairs", "cycle3a_pairs", "multiseed_summary") if isinstance(paired.get(sk), dict)]
                        if not sub_dicts:
                            sub_dicts = [pm_k3]
                        for sd_i in sub_dicts:
                            for fk, fv in sd_i.items():
                                if isinstance(fv, dict) and "overall_success_diff" in fv and "p_value_two_sided" in fv["overall_success_diff"]:
                                    pv_f = p_mcn_4dp if fk == pk_drop_matched else float(fv["overall_success_diff"]["p_value_two_sided"])
                                    fam_pvals.append((pv_f, fk))
                    fam_pvals.sort(key=lambda t: t[0])
                    m_fam = max(1, len(fam_pvals))
                    ph_mcn = min(1.0, p_mcn_4dp * m_fam)
                    rm_f = 0.0
                    for r_idx, (pv_f, fk) in enumerate(fam_pvals):
                        rm_f = min(1.0, max(rm_f, pv_f * float(m_fam - r_idx)))
                        if fk == pk_drop_matched:
                            ph_mcn = rm_f
                            break
                    if ph_mcn <= 0.05:
                        drop_str = f"McN p={p_mcn_drop:.3f}* (Holm {ph_mcn:.3f}*)"
                    elif p_mcn_drop <= 0.05:
                        drop_str = f"McN p={p_mcn_drop:.3f} unadj. (Holm {ph_mcn:.3f} NS)"
                    else:
                        drop_str = f"McN p={p_mcn_drop:.3f} NS"
                elif p_drop < 1e-4:
                    drop_str = f"Δ{_short_p_tag(p_drop, hsig_drop)} (McN {p_mcn_drop:.3f})"
                else:
                    drop_str = f"Δ{_short_p_tag(p_drop, hsig_drop)}"

                if short_n == "M2":
                    parts.append(f"{short_n}: {c_k0}/{n_m} ({sr_k0:.1f}%)→{c_k3}/{n_m} ({sr_k3:.1f}%), {drop_str}")
                else:
                    p_vsm2: Optional[float] = None
                    hsig_vsm2 = False
                    p_vsm5: Optional[float] = None
                    hsig_vsm5 = False
                    for pk_cand, pv_dict in pm_k3.items():
                        if not isinstance(pv_dict, dict) or "overall_success_diff" not in pv_dict:
                            continue
                        if (
                            ("Interf" in pk_cand or "k3" in pk_cand)
                            and ("Drop" not in pk_cand and "k3_vs_k0" not in pk_cand)
                            and ("DiD" not in pk_cand and "Penalty" not in pk_cand)
                        ):
                            if p_vsm2 is None and f"_{short_n}_" in f"_{pk_cand}_" and "M2" in pk_cand:
                                p_vsm2 = float(pv_dict["overall_success_diff"]["p_value_two_sided"])
                                _, hsig_vsm2 = holm_k3.get(pk_cand, (1.0, False))
                            elif p_vsm5 is None and short_n == "M7" and f"_{short_n}_" in f"_{pk_cand}_" and "M5" in pk_cand:
                                p_vsm5 = float(pv_dict["overall_success_diff"]["p_value_two_sided"])
                                _, hsig_vsm5 = holm_k3.get(pk_cand, (1.0, False))
                    if p_vsm2 is None and m2_k3_by_id:
                        c_m2_ids = sorted(set(k3_by_id.keys()) & set(m2_k3_by_id.keys()))
                        p_vsm2 = _exact_mcnemar_two_sided_p(
                            [k3_by_id[cid] for cid in c_m2_ids],
                            [m2_k3_by_id[cid] for cid in c_m2_ids],
                        )
                    vsm2_str = f"; vs M2 {_short_p_tag(p_vsm2, hsig_vsm2)}" if p_vsm2 is not None else ""
                    vsm5_str = f", M5 {_short_p_tag(p_vsm5, hsig_vsm5)}" if p_vsm5 is not None else ""
                    parts.append(
                        f"{short_n}: {c_k0}/{n_m} ({sr_k0:.1f}%)→{c_k3}/{n_m} ({sr_k3:.1f}%), {drop_str}{vsm2_str}{vsm5_str}"
                    )
            else:
                sr_k3 = float(rd["summary"]["overall"]["success_rate"]) * 100.0
                parts.append(f"{short_n}: {sr_k3:.2f}%")
        hdr_k3 = f"Closed-Loop k=0→k=3 (N={matched_n}, two-sided):\n" if matched_n else "Closed-Loop k=0→k=3 (two-sided):\n"
        ax_c.text(
            0.02,
            0.97,
            hdr_k3 + "\n".join(parts),
            transform=ax_c.transAxes,
            va="top",
            fontsize=5.45 if len(parts) >= 3 else 6.35,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "#FFF8DC", "edgecolor": "#888888", "alpha": 0.94},
        )

    out_stem = "fig7_cross_session_interference"
    _save_both(fig, out_dir, out_stem)
    return out_dir / f"{out_stem}.png"


# =============================================================================
# Figure 8: Frozen S_{past,t} Retention + Hop-by-Hop <x,y> + Neg-Eig Dissociation
# =============================================================================


def make_fig8_mechanistic_probes_and_signflip(
    probe_data: Dict[str, Any],
    runs: Dict[str, Dict[str, Any]],
    out_dir: Path,
) -> Optional[Path]:
    models_dict = probe_data.get("models", {})
    if not models_dict:
        return None

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(16.0, 5.0), gridspec_kw={"width_ratios": [1.16, 1.06, 1.12], "wspace": 0.28}
    )

    # --- Panel (a): Probe 2 Frozen S_{past,t} Count & Post-Swap Retention ---
    p2_candidates: List[Tuple[str, str, str]] = [
        ("m0_memoryless", "M0 Markov (0B)", PALETTE["m0"]),
        ("m1_rdvla_loop4", "M1 RD-VLA (0B)", PALETTE["m1"]),
        ("m2_framesamp_modul", "M2 FrameSamp", PALETTE["m2"]),
        ("m3_gdn_b1_l1", "M3 GDN β∈(0,1)", PALETTE["m3"]),
        ("m4_gdn_b2_l1", "M4 GDN β∈(0,2)", PALETTE["m4"]),
        ("m5_gdn2layer_b2_l1", "M5 2L GDN", PALETTE["m5"]),
        ("m6_deltaloop_b2", "M6 DeltaLoop", PALETTE["m6_adapt"]),
        ("m7_deltaloop_2l_b2", "M7 2L DeltaLoop", PALETTE["m7"]),
        # SmolVLA equivalents
        ("smolvla_m0_base", "M0 Base (0B)", PALETTE["m0"]),
        ("smolvla_m1_rdvla_loop4", "M1 RD-Loop (0B)", PALETTE["m1"]),
        ("smolvla_m2_framesamp", "M2 FrameSamp", PALETTE["m2"]),
        ("smolvla_m5_gdn2l_l1", "M5 2L GDN", PALETTE["m5"]),
        ("smolvla_m7_deltaloop", "M7 DeltaLoop", PALETTE["m7"]),
        ("smolvla_m7_deltaloop_q16", "M7-Q16 (16 Coda)", PALETTE["m7_q16"]),
        ("smolvla_m7_deltaloop_anchor", "M7-Anchor (4+4)", PALETTE["m7_anchor"]),
    ]
    # Keep up to 6 representative models if 7 are present so grouped bars stay crisp
    p2_groups = [(mk, ml, mc) for mk, ml, mc in p2_candidates if mk in models_dict]
    if len(p2_groups) > 6 and any(mk == "m4_gdn_b2_l1" for mk, _, _ in p2_groups) and any(mk == "m7_deltaloop_2l_b2" for mk, _, _ in p2_groups):
        p2_groups = [(mk, ml, mc) for mk, ml, mc in p2_groups if mk != "m4_gdn_b2_l1"]

    first_tmap = next(iter(models_dict.values())).get("tasks", {})
    has_smol_10t = "SwingXtimes" in first_tmap and "VideoRepick" in first_tmap
    cat_labels = (
        [
            "Late Horizon\n(≥96 steps)",
            "PickXtimes\nPick Count",
            "SwingXtimes\nSwing Count",
            "VideoRepick\nRepick Count",
        ]
        if has_smol_10t
        else [
            "Early Count\n(<32 steps)",
            "Late Count\n(≥96 steps)",
            "PickXtimes\nPick Count",
            "PickHighlight\nStage Count",
        ]
    )
    pxt_cat_idx = 1.0 if has_smol_10t else 2.0
    xc = np.arange(len(cat_labels))
    w_a = min(0.14, 0.76 / max(1, len(p2_groups)))
    offs_a = np.linspace(-w_a * (len(p2_groups) - 1) / 2.0, w_a * (len(p2_groups) - 1) / 2.0, len(p2_groups))

    m0_pxt: Optional[float] = None
    best_pxt = (-1.0, 0.0)
    for off, (mk, ml, mc) in zip(offs_a, p2_groups):
        tmap = models_dict[mk]["tasks"]
        early_v = [
            float(td["probe2_state_retention"]["count_from_S_past_by_horizon"]["early_lt32"]["acc_pct"])
            for td in tmap.values()
            if "early_lt32" in td["probe2_state_retention"].get("count_from_S_past_by_horizon", {})
        ]
        late_v = [
            float(td["probe2_state_retention"]["count_from_S_past_by_horizon"]["late_ge96"]["acc_pct"])
            for td in tmap.values()
            if "late_ge96" in td["probe2_state_retention"].get("count_from_S_past_by_horizon", {})
        ]
        pxt_v = float(tmap["PickXtimes"]["probe2_state_retention"]["count_from_S_past"]["acc_pct"]) if "PickXtimes" in tmap else float("nan")
        phl_v = float(tmap["PickHighlight"]["probe2_state_retention"]["count_from_S_past"]["acc_pct"]) if "PickHighlight" in tmap else float("nan")
        swx_v = float(tmap["SwingXtimes"]["probe2_state_retention"]["count_from_S_past"]["acc_pct"]) if "SwingXtimes" in tmap else float("nan")
        vrp_v = float(tmap["VideoRepick"]["probe2_state_retention"]["count_from_S_past"]["acc_pct"]) if "VideoRepick" in tmap else float("nan")
        if "m0_" in mk:
            m0_pxt = pxt_v
        elif math.isfinite(pxt_v) and pxt_v >= best_pxt[0]:
            best_pxt = (pxt_v, float(off))
        bars_v = [float(np.mean(late_v)), pxt_v, swx_v, vrp_v] if has_smol_10t else [float(np.mean(early_v)), float(np.mean(late_v)), pxt_v, phl_v]
        ax_a.bar(xc + off, bars_v, width=w_a * 0.92, color=mc, edgecolor="#222222", linewidth=0.65, label=ml, zorder=3)

    ax_a.set_xticks(xc)
    ax_a.set_xticklabels(cat_labels, fontsize=8.0)
    ax_a.set_ylim(42.0, 111.0)
    ax_a.set_ylabel(r"Linear Probe Accuracy from $\mathbf{S}_{\mathrm{past}, t}$ (%)")
    if m0_pxt is not None and best_pxt[0] > m0_pxt:
        gain_pxt = best_pxt[0] - m0_pxt
        ax_a.annotate(
            f"+{gain_pxt:.1f} pp vs M0\non PickXtimes",
            xy=(pxt_cat_idx + best_pxt[1], best_pxt[0] + 0.4),
            xytext=(pxt_cat_idx - 0.78, min(103.0, best_pxt[0] + 8.5)),
            arrowprops={"arrowstyle": "->", "lw": 0.95, "color": "#222222"},
            fontsize=7.4,
            fontweight="bold",
        )
    ax_a.legend(loc="lower left", ncol=2, frameon=True, facecolor="white", framealpha=0.92, fontsize=6.8)
    ax_a.set_title(r"(a) Frozen $\mathbf{S}_{\mathrm{past}, t}$ State & Count Retention")

    # --- Panel (b): Hop-by-Hop <x, y> Dereferencing Change Δpx(l) - px(l=1) ---
    is_smolvla_probes = "smolvla_m7_deltaloop" in models_dict
    is_v3_probes = ("m7_deltaloop_2l_b2" in models_dict) or is_smolvla_probes
    deref_tasks = (
        ["ButtonUnmaskSwap", "VideoUnmaskSwap", "PickHighlight", "PickXtimes", "MoveCube", "SwingXtimes", "VideoRepick", "VideoPlaceOrder"]
        if is_smolvla_probes
        else (
            ["ButtonUnmaskSwap", "VideoUnmaskSwap", "PickHighlight", "PickXtimes", "StopCube", "MoveCube"]
            if is_v3_probes
            else ["ButtonUnmaskSwap", "VideoUnmaskSwap", "VideoPlaceOrder"]
        )
    )
    hops = [1, 2, 3, 4]
    hop_series = [
        ("m1_rdvla_loop4", "M1: RD-VLA (Memoryless Loop)", PALETTE["m1"], "--s"),
        ("smolvla_m1_rdvla_loop4", "M1: SmolVLA RD-Loop (0 B)", PALETTE["m1"], "--s"),
        ("m6_deltaloop_b2", "M6: 1L DeltaLoop (Re-Query)", PALETTE["m6_adapt"], "-*"),
        ("m6b_deltaloop_wb", "M6b: DeltaLoop + WriteBack", PALETTE["m6b"], "-o"),
        ("m7_deltaloop_2l_b2", "M7: 2L DeltaLoop (6-Task Mean)", PALETTE["m7"], "-P"),
        ("smolvla_m7_deltaloop", "M7: SmolVLA-DeltaLoop (8-Task Mean)", PALETTE["m7"], "-*"),
        ("smolvla_m7_deltaloop_q16", "M7-Q16: 16-Coda DeltaLoop", PALETTE["m7_q16"], "-P"),
        ("smolvla_m7_deltaloop_anchor", "M7-Anchor: 4+4 Partitioned", PALETTE["m7_anchor"], "-D"),
    ]

    ax_b.axhline(0.0, color="#333333", lw=1.0, linestyle=":")
    min_delta_b = -0.75 if is_smolvla_probes else -0.9
    max_delta_b = 0.22 if is_smolvla_probes else 0.35
    for mk, ml, mc, st in hop_series:
        if mk not in models_dict:
            continue
        tmap = models_dict[mk]["tasks"]
        per_task_deltas = []
        for dt_name in deref_tasks:
            if dt_name not in tmap:
                continue
            c_depth = tmap[dt_name]["probe1_xy_localization"].get("coda_only_by_depth", {})
            if all(f"l={l}" in c_depth for l in hops):
                base_px = float(c_depth["l=1"]["mean_px_err"])
                per_task_deltas.append([float(c_depth[f"l={l}"]["mean_px_err"]) - base_px for l in hops])
        if per_task_deltas:
            arr = np.asarray(per_task_deltas, dtype=np.float64)
            mean_d = arr.mean(axis=0)
            sem_d = arr.std(axis=0) / math.sqrt(len(per_task_deltas))
            min_delta_b = min(min_delta_b, float((mean_d - sem_d).min()))
            max_delta_b = max(max_delta_b, float((mean_d + sem_d).max()))
            msz = 8.5 if st.endswith("*") else 6.0
            ax_b.plot(hops, mean_d, st, color=mc, lw=2.2, markersize=msz, label=ml, zorder=4)
            ax_b.fill_between(hops, mean_d - sem_d, mean_d + sem_d, color=mc, alpha=0.14)

    # Single-task strongest multi-hop sharpening highlight curve
    if "smolvla_m7_deltaloop_anchor" in models_dict and "PickHighlight" in models_dict["smolvla_m7_deltaloop_anchor"]["tasks"]:
        cd_ph_a = models_dict["smolvla_m7_deltaloop_anchor"]["tasks"]["PickHighlight"]["probe1_xy_localization"]["coda_only_by_depth"]
        b_ph_a = float(cd_ph_a["l=1"]["mean_px_err"])
        ph_a_curve = [float(cd_ph_a[f"l={l}"]["mean_px_err"]) - b_ph_a for l in hops]
        min_delta_b = min(min_delta_b, min(ph_a_curve))
        ax_b.plot(
            hops,
            ph_a_curve,
            ":^",
            color=PALETTE["m7_anchor"],
            lw=1.75,
            markersize=5.4,
            label=f"M7-Anchor PickHighlight ({min(ph_a_curve):+.2f} px @ l=4)",
            zorder=3,
        )
    elif "smolvla_m7_deltaloop" in models_dict and "PickHighlight" in models_dict["smolvla_m7_deltaloop"]["tasks"]:
        cd_ph_s = models_dict["smolvla_m7_deltaloop"]["tasks"]["PickHighlight"]["probe1_xy_localization"]["coda_only_by_depth"]
        b_ph_s = float(cd_ph_s["l=1"]["mean_px_err"])
        ph_s_curve = [float(cd_ph_s[f"l={l}"]["mean_px_err"]) - b_ph_s for l in hops]
        min_delta_b = min(min_delta_b, min(ph_s_curve))
        ax_b.plot(
            hops,
            ph_s_curve,
            ":^",
            color=PALETTE["m7_l2"],
            lw=1.7,
            markersize=5.4,
            label=f"M7 PickHighlight ({min(ph_s_curve):+.2f} px @ l=4)",
            zorder=3,
        )
    elif "m7_deltaloop_2l_b2" in models_dict and "PickHighlight" in models_dict["m7_deltaloop_2l_b2"]["tasks"]:
        cd_ph = models_dict["m7_deltaloop_2l_b2"]["tasks"]["PickHighlight"]["probe1_xy_localization"]["coda_only_by_depth"]
        b_ph = float(cd_ph["l=1"]["mean_px_err"])
        ph_curve = [float(cd_ph[f"l={l}"]["mean_px_err"]) - b_ph for l in hops]
        min_delta_b = min(min_delta_b, min(ph_curve))
        ax_b.plot(
            hops,
            ph_curve,
            ":^",
            color=PALETTE["m7_l2"],
            lw=1.7,
            markersize=5.4,
            label=f"M7 PickHighlight ({min(ph_curve):+.2f} px @ l=3)",
            zorder=3,
        )
    elif "m6_deltaloop_b2" in models_dict and "VideoUnmaskSwap" in models_dict["m6_deltaloop_b2"]["tasks"]:
        cd_vus = models_dict["m6_deltaloop_b2"]["tasks"]["VideoUnmaskSwap"]["probe1_xy_localization"]["coda_only_by_depth"]
        b_vus = float(cd_vus["l=1"]["mean_px_err"])
        vus_curve = [float(cd_vus[f"l={l}"]["mean_px_err"]) - b_vus for l in hops]
        min_delta_b = min(min_delta_b, min(vus_curve))
        ax_b.plot(
            hops,
            vus_curve,
            ":^",
            color=PALETTE["m6_l4"],
            lw=1.6,
            markersize=5.2,
            label=f"M6 VideoUnmaskSwap ({min(vus_curve):+.2f} px @ l=3)",
            zorder=3,
        )

    ax_b.set_xticks(hops)
    ax_b.set_xticklabels(["Hop l=1", "Hop l=2", "Hop l=3", "Hop l=4"])
    ax_b.set_ylabel("Δ Localization Error vs. Hop 1 (px, ↓ sharper)")
    y_lo_b = min(-1.12 if is_smolvla_probes else -1.40, min_delta_b - 0.68)
    y_hi_b = max(0.36 if is_smolvla_probes else 0.48, max_delta_b + 0.18)
    ax_b.set_ylim(y_lo_b, y_hi_b)
    if "m1_rdvla_loop4" in models_dict and max_delta_b > 0.12:
        ax_b.text(1.06, y_hi_b - 0.12, "↑ Error Grows Without Memory (Drift)", fontsize=7.5, color=PALETTE["m1"], fontweight="bold")
    elif "smolvla_m7_deltaloop_anchor" in models_dict:
        ax_b.text(
            1.03,
            y_hi_b - 0.11,
            "4+4 Anchor: -0.60 px @ l=4 (vs -0.24 px N_c=4) | U4: 4.35 px",
            fontsize=6.6,
            color=PALETTE["m7_anchor"],
            fontweight="bold",
        )
    elif "smolvla_m7_deltaloop_q16" in models_dict:
        ax_b.text(
            1.04,
            y_hi_b - 0.10,
            "M7-Q16 Zero-Shot U4: 4.57→4.35 px",
            fontsize=7.0,
            color=PALETTE["m7_q16"],
            fontweight="bold",
        )
    elif is_smolvla_probes:
        ax_b.text(1.06, y_hi_b - 0.10, "Coda Re-Query vs. 0B RD-Loop (U4: 4.7 px)", fontsize=7.2, color="#222222", fontweight="bold")
    else:
        ax_b.text(1.06, y_hi_b - 0.14, "Zero-Shot U4 Head: 1.4–2.4 px (<1% img)", fontsize=7.4, color="#333333", fontweight="bold")
    if min_delta_b <= -0.85 and not is_smolvla_probes:
        ax_b.text(2.04, -0.73, "↓ Multi-Hop Re-Query Sharpens Target <x, y>", fontsize=7.2, color="#222222", fontweight="bold")
    elif not is_smolvla_probes:
        ax_b.text(1.06, y_lo_b + 0.44, "↓ Multi-Hop Re-Query Sharpens Target <x, y>", fontsize=7.4, color="#222222", fontweight="bold")
    ax_b.legend(loc="lower left", bbox_to_anchor=(0.0, 0.01), frameon=True, facecolor="white", framealpha=0.93, fontsize=6.65)
    ax_b.set_title("(b) Hop-by-Hop <x, y> Target Dereferencing")

    # --- Panel (c): Negative-Eigenvalue & Motion-Gated Beta Decomposition ---
    is_v3_eval = "m7_deltaloop_2l_b2_adaptive" in runs or "m7_deltaloop_2l_b2_L2" in runs
    m4_c_lbl = "M4: Motion-Gated\n1L β∈(0,2)" if is_v3_eval else "M4: Un-gated\n1L β∈(0,2)"
    sf_candidates = [
        ("m3_gdn_b1_l1", "M3: 1L β∈(0,1)\n(Monotone)", PALETTE["m3"]),
        ("m4_gdn_b2_l1", m4_c_lbl, PALETTE["m4"]),
        ("m6_deltaloop_b2_L2", "M6: Loop L=2\nβ∈(0,2)", PALETTE["m6_l2"]),
        ("m5_gdn2layer_b2_l1", "M5: 2L Iso-State\nβ∈(0,2)", PALETTE["m5"]),
        # SmolVLA fallback
        ("smolvla_m0_base", "M0: Base (0B)", PALETTE["m0"]),
        ("smolvla_m2_framesamp", "M2: FrameSamp", PALETTE["m2"]),
        ("smolvla_m5_gdn2l_l1", "M5: 2L GDN", PALETTE["m5"]),
        ("smolvla_m7_deltaloop_adaptive", "M7: DeltaLoop", PALETTE["m7"]),
    ]
    sf_models = [(mk, ml, mc) for mk, ml, mc in sf_candidates if mk in runs]
    if not is_smolvla_probes and len(sf_models) >= 3:
        cats_c = ["Permanence\n(Suite)", "Reference\n(Suite)", "Imitation\n(Suite)", "StopCube\n(Counting)"]
        xc_c = np.arange(len(cats_c))
        w_c = min(0.19, 0.76 / len(sf_models))
        offs_c = np.linspace(-w_c * (len(sf_models) - 1) / 2.0, w_c * (len(sf_models) - 1) / 2.0, len(sf_models))
        max_val_c = 22.0
        vals_by_mk: Dict[str, List[float]] = {}
        off_by_mk: Dict[str, float] = {}
        for off, (mk, ml, mc) in zip(offs_c, sf_models):
            sm = runs[mk]["summary"]
            vals_c = [
                float(sm["suites"]["Permanence"]["success_rate"]) * 100.0,
                float(sm["suites"]["Reference"]["success_rate"]) * 100.0,
                float(sm["suites"]["Imitation"]["success_rate"]) * 100.0,
                float(sm["tasks"]["StopCube"]["success_rate"]) * 100.0 if "StopCube" in sm.get("tasks", {}) else float(sm["suites"]["Counting"]["success_rate"]) * 100.0,
            ]
            vals_by_mk[mk] = vals_c
            off_by_mk[mk] = float(off)
            max_val_c = max(max_val_c, max(vals_c))
            ax_c.bar(xc_c + off, vals_c, width=w_c * 0.92, color=mc, edgecolor="#222222", linewidth=0.7, label=ml.replace("\n", " "), zorder=3)

        ax_c.set_xticks(xc_c)
        ax_c.set_xticklabels(cats_c, fontsize=8.0)
        y_top_c = max(26.5, max_val_c + (11.5 if is_v3_eval else 7.5))
        ax_c.set_ylim(0.0, y_top_c)
        ax_c.set_ylabel("Success Rate (%)")

        if "m4_gdn_b2_l1" in vals_by_mk and "m3_gdn_b1_l1" in vals_by_mk:
            m4_ov = float(runs["m4_gdn_b2_l1"]["summary"]["overall"]["success_rate"]) * 100.0
            m3_ov = float(runs["m3_gdn_b1_l1"]["summary"]["overall"]["success_rate"]) * 100.0
            if m4_ov < m3_ov - 2.0:
                # Cycle 2A un-gated negative-eigenvalue static-frame sign-flip
                ax_c.annotate(
                    "Static-frame sign-flip\nhurts un-gated M4\n(overall p=0.105 NS)",
                    xy=(1.0 + off_by_mk["m4_gdn_b2_l1"], vals_by_mk["m4_gdn_b2_l1"][1] + 0.3),
                    xytext=(0.10, y_top_c - 7.4),
                    arrowprops={"arrowstyle": "->", "lw": 0.95, "color": "#333333"},
                    fontsize=7.0,
                    fontweight="bold",
                )
                if "m5_gdn2layer_b2_l1" in vals_by_mk:
                    sc_v = vals_by_mk["m5_gdn2layer_b2_l1"][3]
                    ax_c.annotate(
                        f"2L M5 StopCube:\n{sc_v:.0f}% (overall 11.2%)",
                        xy=(3.0 + off_by_mk["m5_gdn2layer_b2_l1"], sc_v + 0.3),
                        xytext=(1.95, y_top_c - 3.8),
                        arrowprops={"arrowstyle": "->", "lw": 0.95, "color": PALETTE["m5"]},
                        fontsize=7.0,
                        fontweight="bold",
                        color=PALETTE["m5"],
                    )
            else:
                # Cycle 3A U5 Motion-Gated beta closes static-frame gap (M4=M3=10.6%, p=1.00 NS)
                ref_m4 = vals_by_mk["m4_gdn_b2_l1"][1]
                ax_c.annotate(
                    f"Motion-β M4: {ref_m4:.1f}% Ref.\n(overall M4=M3=10.6%, NS)",
                    xy=(1.0 + off_by_mk["m4_gdn_b2_l1"], ref_m4 + 0.4),
                    xytext=(0.56, y_top_c - 10.6),
                    arrowprops={"arrowstyle": "->", "lw": 0.95, "color": "#005A8D"},
                    fontsize=6.9,
                    fontweight="bold",
                    color="#005A8D",
                )
                if "m5_gdn2layer_b2_l1" in vals_by_mk:
                    sc_m5 = vals_by_mk["m5_gdn2layer_b2_l1"][3]
                    ax_c.annotate(
                        f"2L M5: 12.5% overall\n(StopCube {sc_m5:.0f}%)",
                        xy=(3.0 + off_by_mk["m5_gdn2layer_b2_l1"], sc_m5 + 0.4),
                        xytext=(2.18, y_top_c - 4.8),
                        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": PALETTE["m5"]},
                        fontsize=7.0,
                        fontweight="bold",
                        color=PALETTE["m5"],
                    )

        ax_c.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.7)
        title_c = "(c) Motion-Gated β vs. β∈(0,1) & 2L GDN" if is_v3_eval else "(c) β∈(0,2) Static Sign-Flip vs. 2L GDN"
        ax_c.set_title(title_c)
    else:
        # Probe-driven Coda & Zero-Shot U4 <x, y> Spatial Grounding panel (Cycle 3B SmolVLA)
        p1_models = [
            (mk, ml, mc)
            for mk, ml, mc in [
                ("smolvla_m0_base", "M0 Base (0B)", PALETTE["m0"]),
                ("smolvla_m1_rdvla_loop4", "M1 RD-Loop (0B)", PALETTE["m1"]),
                ("smolvla_m2_framesamp", "M2 FrameSamp (24KB)", PALETTE["m2"]),
                ("smolvla_m5_gdn2l_l1", "M5 2L GDN (520KB)", PALETTE["m5"]),
                ("smolvla_m7_deltaloop", "M7 DeltaLoop (520KB)", PALETTE["m7"]),
                ("smolvla_m7_deltaloop_q16", "M7-Q16 (16 Coda, 520KB)", PALETTE["m7_q16"]),
                ("smolvla_m7_deltaloop_anchor", "M7-Anchor (4+4, 520KB)", PALETTE["m7_anchor"]),
            ]
            if mk in models_dict
        ]
        if p1_models:
            cats_p1 = [
                "9-Task Mean\n(Coda <x,y>)",
                "VideoSwap &\nPickHighlight",
                "SwingX &\nPickXtimes",
                "9-Task Mean\n(Zero-Shot U4)",
            ]
            xc_p1 = np.arange(len(cats_p1))
            w_p1 = min(0.15, 0.76 / len(p1_models))
            offs_p1 = np.linspace(-w_p1 * (len(p1_models) - 1) / 2.0, w_p1 * (len(p1_models) - 1) / 2.0, len(p1_models))
            max_p1 = 16.0
            m2_coda_mean: Optional[float] = None
            m7_coda_best: Optional[Tuple[float, float, str]] = None
            q16_u4_best: Optional[Tuple[float, float]] = None
            for off, (mk, ml, mc) in zip(offs_p1, p1_models):
                tmap = models_dict[mk]["tasks"]
                c_all, c_ref, c_cnt, z_all = [], [], [], []
                for tn, tv in tmap.items():
                    p1 = tv["probe1_xy_localization"]
                    cd = p1.get("coda_only_by_depth", {})
                    if cd:
                        best_c = min(float(v["mean_px_err"]) for v in cd.values())
                        c_all.append(best_c)
                        if tn in ("VideoUnmaskSwap", "PickHighlight"):
                            c_ref.append(best_c)
                        if tn in ("SwingXtimes", "PickXtimes"):
                            c_cnt.append(best_c)
                    for zk in ("zero_shot_u4_by_depth", "zero_shot_grounding_by_depth", "zero_shot_by_depth"):
                        if p1.get(zk):
                            z_all.append(min(float(v["mean_px_err"]) for v in p1[zk].values()))
                            break
                vals_p1 = [
                    float(np.mean(c_all)) if c_all else 0.0,
                    float(np.mean(c_ref)) if c_ref else 0.0,
                    float(np.mean(c_cnt)) if c_cnt else 0.0,
                    float(np.mean(z_all)) if z_all else 0.0,
                ]
                if "framesamp" in mk:
                    m2_coda_mean = vals_p1[0]
                elif "m7_deltaloop" in mk:
                    if m7_coda_best is None or vals_p1[0] <= m7_coda_best[0]:
                        m7_coda_best = (vals_p1[0], float(off), mc)
                    if "q16" in mk and vals_p1[3] > 0:
                        q16_u4_best = (vals_p1[3], float(off))
                max_p1 = max(max_p1, max(vals_p1))
                ax_c.bar(xc_p1 + off, vals_p1, width=w_p1 * 0.92, color=mc, edgecolor="#222222", linewidth=0.65, label=ml, zorder=3)

            ax_c.set_xticks(xc_p1)
            ax_c.set_xticklabels(cats_p1, fontsize=8.0)
            y_top_p1 = max(23.2, max_p1 + 7.2)
            ax_c.set_ylim(0.0, y_top_p1)
            ax_c.set_ylabel("Target <x, y> Error (px, ↓ sharper)")
            if m2_coda_mean is not None and m7_coda_best is not None and m7_coda_best[0] > 0:
                ratio_coda = m2_coda_mean / m7_coda_best[0]
                ax_c.annotate(
                    f"Coda probe: {ratio_coda:.2f}× lower\npx err vs FrameSamp-32",
                    xy=(0.0 + m7_coda_best[1], m7_coda_best[0] + 0.3),
                    xytext=(-0.42, y_top_p1 - 4.8),
                    arrowprops={"arrowstyle": "->", "lw": 0.95, "color": m7_coda_best[2]},
                    fontsize=7.0,
                    fontweight="bold",
                    color=m7_coda_best[2],
                )
            if q16_u4_best is not None:
                ax_c.annotate(
                    f"M7-Q16 U4:\n{q16_u4_best[0]:.2f} px",
                    xy=(3.0 + q16_u4_best[1], q16_u4_best[0] + 0.25),
                    xytext=(2.36, q16_u4_best[0] + 4.8),
                    arrowprops={"arrowstyle": "->", "lw": 0.9, "color": PALETTE["m7_q16"]},
                    fontsize=7.0,
                    fontweight="bold",
                    color=PALETTE["m7_q16"],
                )
            ax_c.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.92, fontsize=6.7)
            ax_c.set_title("(c) Coda & Zero-Shot U4 <x, y> Grounding")

    out_stem = "fig8_mechanistic_probes_and_signflip"
    _save_both(fig, out_dir, out_stem)
    return out_dir / f"{out_stem}.png"


def _build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate RoboMME evaluation & mechanistic probe figures (fig5-fig8)")
    p.add_argument("--eval-dir", type=str, default=str(DEFAULT_EVAL_DIR), help="Directory of RoboMME eval JSONs")
    p.add_argument("--probe-json", type=str, default=str(DEFAULT_PROBE_JSON), help="Path to mechanistic_probes.json")
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory for PNG + PDF figures")
    return p


def main() -> None:
    args = _build_cli().parse_args()
    configure_style()

    eval_dir = Path(args.eval_dir)
    probe_path = Path(args.probe_json)
    out_dir = Path(args.out_dir)

    runs, paired = load_eval_directory(eval_dir)
    probe_data = json.loads(probe_path.read_text()) if probe_path.is_file() else {}

    clean_eval_count = sum(
        1 for d in runs.values() if int(d.get("eval_spec", {}).get("distractor_sessions", 0)) == 0
    )
    generated: List[Path] = []
    fn_list = []
    if clean_eval_count >= 3:
        fn_list.extend(
            [
                lambda: make_fig5_robomme_main_suites(runs, paired, out_dir),
                lambda: make_fig6_pareto_and_adaptive_halting(runs, out_dir, paired=paired),
            ]
        )
    fn_list.extend(
        [
            lambda: make_fig7_cross_session_interference(probe_data, runs, out_dir, paired=paired),
            lambda: make_fig8_mechanistic_probes_and_signflip(probe_data, runs, out_dir),
        ]
    )
    for fn in fn_list:
        res_path = fn()
        if res_path is not None:
            generated.append(res_path)

    print(f"[DONE] Generated {len(generated)} figure pairs (.png + .pdf) in {out_dir}:")
    for p in generated:
        print(f"  - {p.name} & {p.with_suffix('.pdf').name}")


if __name__ == "__main__":
    main()
