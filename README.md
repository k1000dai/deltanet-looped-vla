# 🤖 [Fun Project] DeltaLoop-VLA & DeltaMem-π0.5 v3 + CORM

📚 [Tech Report (PDF)](https://k1000dai.github.io/deltanet-looped-vla/tech_report.pdf) | 🌐 [Project Website](https://k1000dai.github.io/deltanet-looped-vla/) | 🏆 [Phase 3 Results](RESULTS_PHASE3.md) | 📊 [Phase 2 Ablations](RESULTS_PHASE2.md) | 📘 [Phase 1 Report](RESULTS_REPORT.md) | 🧾 [BibTeX](#citation)

![RoboMME test Leaderboard & Suite Breakdown](docs/figures_phase3/fig_p3_1_leaderboard_and_suites.png)

Associative Gated DeltaNet memory, iterative multi-hop readout, and hybrid causal reveal tracking (**CORM**) for long-horizon manipulation on the 16-task **RoboMME** benchmark (**0 privileged simulator masks or poses at evaluation time**).

- **Phase 1 (`DeltaLoop-VLA`, Learned Core):** Isolates negative-eigenvalue Gated DeltaNet writing ($\beta_t \in (0, 2)$, `100.0%` $S_8$ swaps) from weight-tied multi-hop readout (`97.1%` ID / `80.4%` OOD) and proves a `520 KB` GDN state retains `93.5%` counts & `4.36 px` targets.
- **Phase 2 (`DeltaMem-π0.5`, Pure Learned Memory):** Scales multi-timescale GDN + surprise canvas (`2.39 MiB` $O(1)$ state), `EpisodeBoundaryForgetGate`, and `Hop-2` readout to `3.44B` **$\pi_{0.5}$** (`48.75%` single-session; `41.67%` vs. `15.42%` `FrameSamp` under un-reset $k=3$ streaming).
- **Phase 3 (`DeltaMem-π0.5 v3 + CORM`, Hybrid Neural + Rule-Based Grounding):** Combines our learned Phase-2 GDN policy on the 8 `Counting + Imitation` tasks (`53.75%` on `test800`) with **`CORM`** — a hand-engineered, RoboMME-specific non-privileged perception & finite-state subgoal module driving `SymbolicGround` $\pi_{0.5}$ via `<y, x>` subgoals on the 8 `Permanence + Reference` tasks (`88.25%` on `test800`) — reaching **`69.29%`** on clean untouched **`test560`** ($p_{\text{conserv}} = 2.07 \times 10^{-8}$ vs. `T²Mem` `56.83%`), **`71.00%`** on full **`test800`** ($p_{\text{conserv}} = 2.69 \times 10^{-12}$), and **`70.83%`** under un-reset cross-task $k=3$ streaming.

## Highlights (Grouped by Inductive Assumptions)

| Category / Setting | Method | Episode Memory | 16-Task SR | Statistical Audit / Note |
| :--- | :--- | :---: | :---: | :--- |
| **E2E Learned Memory (`test800×3`)** | `T²Mem` (`18×16` fast MLPs, `1,536` H200-h) | `36.35 MiB` | `56.83%` | End-to-end learned memory SOTA (`arXiv:2609.36720`) |
| **Pure Learned Memory (`test240`)** | **DeltaMem-π0.5 Phase 2 (`Hybrid + Hop-2`)** | **`2.39 MiB` ($O(1)$)** | **`48.75%`** | `+27.08 pp` vs. `π0.5` Base (`387.8 ms/chunk`, `7.04 GiB`) |
| **Hybrid Rule-Based `CORM` + 2-Ckpt (`test560`)** | **DeltaMem-π0.5 v3 + CORM (Clean `ep 15..49`)** | `2.39 + ≤213 MiB`* | **`69.29%`** | **Clean untouched split: `+12.45 pp` vs. `T²Mem` ($p = 2.07 \times 10^{-8}$)** |
| **Hybrid Rule-Based `CORM` + 2-Ckpt (`test800`)** | **DeltaMem-π0.5 v3 + CORM (Full `ep 0..49`)** | `2.39 + ≤213 MiB`* | **`71.00%`** | **`+14.17 pp` vs. `T²Mem` ($p = 2.69 \times 10^{-12}$, `410.6 ms` on 5080L)** |
| **Un-reset `test240` ($k=3$, cross-task)** | `FrameSamp+Modul` / `Phase 2` / **Phase 3** | `2.0` / **`2.39`** / `≤215 MiB` | `15.4%` / `41.7%` / **`70.83%`** | **`+55.4 pp` vs. FS**; Phase-3 own $k=1 \to 3$: `-4.2 pp` ($p=0.031$, Count `-16.7 pp`) |
| **Synthetic `1A-Online` / `1A-Retro`** | **DeltaLoop ($\beta \in (0, 2)$, $L=6$)** | **`520 KB`** | **`100.0%` / `97.1%`** | Swap tracking (`100%` OOD) & pointer chasing (`80.4%` OOD) |

*\*`2.39 MiB` constant GDN+canvas state on the 8 learned tasks, plus up to `213 MiB` CPU `uint8` demo-frame buffering (`O(T_demo)`) inside `corm._demo_frames` on long video demos (`VideoPlaceOrder`). Full methodological & fairness disclosures are documented in [RESULTS_PHASE3.md](RESULTS_PHASE3.md).*

## Setup

```bash
git clone https://github.com/k1000dai/deltanet-looped-vla.git
cd deltanet-looped-vla
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch numpy scipy matplotlib safetensors transformers einops sentencepiece
```

For closed-loop ManiSkill3 evaluation, install [RoboMME](https://github.com/RoboMME/robomme_benchmark) under `external/robomme_benchmark`.

## Quick Start

```bash
# 1. Synthetic Write vs. Multi-Hop Read Benchmark (Phase 1)
uv run python scripts/run_synthetic.py --task all --seeds 0 1 2 --out-dir results/synthetic

# 2. Pure learned single-checkpoint evaluation (Phase 2: test240 -> 48.75% k=1 / 41.67% k=3)
DISPLAY="" external/robomme_benchmark/.venv/bin/python scripts/eval_pi05.py \
  --ckpt-dir checkpoints/phase2/init/perceptual_framesamp_modul_pt \
  --memory-adapter pi05_mem.adapter:build_eval_adapter_imitation_l2 \
  --memory-ckpt checkpoints/phase2/p2_m5_hybrid_v3_stageC_hop2/step_000200_ema.pt \
  --tasks all --dataset test --episodes 15 --seed 7 \
  --output results/phase2/test_m5_hybrid_v3_stageC_hop2_imitation_l2.json

# 3. Hybrid Phase-3 evaluation on full official test800 (800 eps -> 71.00%; ep 15..49 clean test560 -> 69.29%)
DISPLAY="" external/robomme_benchmark/.venv/bin/python scripts/eval_pi05.py \
  --ckpt-dir checkpoints/phase2/init/perceptual_framesamp_modul_pt \
  --policy-name p3_deltamem_v3_corm_8task \
  --memory-adapter pi05_mem.adapter:build_eval_adapter_imitation_l2 \
  --memory-ckpt checkpoints/phase2/p2_m5_hybrid_v3_stageC_hop2/step_000200_ema.pt \
  --specialist-ckpt-dir checkpoints/phase2/init/symbolic_grounded_subgoal_pt \
  --specialist-memory-adapter pi05_mem.adapter:build_deltaground_vlm_adapter \
  --specialist-memory-ckpt checkpoints/phase2/p2_m9_sym_stage_gsstd_v1/best_unseen_ema.pt \
  --specialist-route 8task \
  --tasks all --dataset test --episodes 50 --start-episode 0 --parallel-workers 1 --seed 7 \
  --output results/phase3/test800_p3_deltamem_v3_corm.json

# 4. Un-reset k=3 cross-task multi-session interference (240 target eps = 720 rollouts -> 70.83%)
DISPLAY="" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  external/robomme_benchmark/.venv/bin/python scripts/eval_pi05.py \
  --ckpt-dir checkpoints/phase2/init/perceptual_framesamp_modul_pt \
  --policy-name p3_deltamem_v3_corm_8task_interfk3 \
  --memory-adapter pi05_mem.adapter:build_eval_adapter_imitation_l2 \
  --memory-ckpt checkpoints/phase2/p2_m5_hybrid_v3_stageC_hop2/step_000200_ema.pt \
  --specialist-ckpt-dir checkpoints/phase2/init/symbolic_grounded_subgoal_pt \
  --specialist-memory-adapter pi05_mem.adapter:build_deltaground_vlm_adapter \
  --specialist-memory-ckpt checkpoints/phase2/p2_m9_sym_stage_gsstd_v1/best_unseen_ema.pt \
  --specialist-route 8task \
  --tasks all --dataset test --episodes 15 --start-episode 0 --interference-k 3 \
  --parallel-workers 1 --seed 7 \
  --output results/phase3/test240_p3_deltamem_v3_corm_interfk3.json
```

## Repo Structure

```text
├── analysis/                  # Mechanistic probes, statistical audit & figure scripts
├── docs/
│   ├── figures_phase3/        # Phase-3 leaderboard, 16-task & k=3 suite-dissection figures
│   ├── figures_phase2/        # Phase-2 figures & m=146 paired McNemar/Holm test JSONs
│   └── website/               # GitHub Pages project website (index.html, tech_report.pdf)
├── paper/                     # LaTeX technical report (tech_report.tex, tech_report.pdf)
├── results/                   # Raw JSON logs (synthetic, SmolVLA, Phase 2, and Phase 3 test800/k=3)
├── scripts/                   # Conversion, cache extraction, training & closed-loop evaluation
├── src/
│   ├── memory/                # Gated DeltaNet (beta in (0,2)) & multi-timescale GDN
│   ├── loop/                  # Weight-tied Prelude-Core-Coda loop & CRN adaptive halting
│   ├── pi05/                  # Native PyTorch 3.44B pi0.5 + SigLIP + ActionExpert + ZeroOracle router
│   ├── pi05_mem/              # HybridSurpriseGDNBridge, ForgetGate, Hop-2, DeltaGround-VLM & CORM (corm.py)
│   └── robomme_eval/          # Parallel ManiSkill3 closed-loop & un-reset interference runner
├── RESULTS_PHASE3.md          # Phase-3 hybrid CORM results, statistical audit & candid disclosures
├── RESULTS_PHASE2.md          # Phase-2 pure learned memory ablations & negative results
└── RESULTS_REPORT.md          # Phase-1 mechanistic & SmolVLA report
```

## Citation

```bibtex
@techreport{k1000dai2026deltaloopvla,
  title  = {[Fun Project] DeltaLoop-VLA and DeltaMem-pi0.5 v3 + CORM:
            Multi-Timescale Gated DeltaNet Memory, Multi-Hop Readout, and
            Hybrid Rule-Based Causal Reveal Tracking on 16-Task RoboMME},
  author = {k1000dai},
  year   = {2026},
  note   = {Technical Report. Primarily written and created using Gemini 4 Carbon.},
  url    = {https://k1000dai.github.io/deltanet-looped-vla/}
}
```

---

**AI Authorship & Creation Disclosure:** This `[Fun Project]` research, codebase, experimental evaluation, LaTeX technical report, and project website were primarily written and created using **Gemini 4 Carbon** (with independent Phase-3 audit by **Claude Opus 5.5**).  
*(本研究・コード実装・実験評価・LaTeX 技術レポートおよびプロジェクトページは、主に **Gemini 4 Carbon** を使用して書かれ・作られました。)*
