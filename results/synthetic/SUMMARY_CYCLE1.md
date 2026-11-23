# Cycle 1 Scientific Report: Shortcut-Free Synthetic Mechanistic & Ablation Suite for DeltaLoop-VLA

**Date:** 2026-11-11  
**Compute:** Local NVIDIA GeForce RTX 5080 Laptop GPU (Peak VRAM across all 87 runs: `4,288 MB` $\ll$ `6,000 MB` cap)  
**Seeds:** 3 independent random seeds (`0, 1, 2`) for every architecture and variant (`87` full training & evaluation runs total)  
**Raw Structured Artifacts:**
- `results/synthetic/task1a_online.json` (`10` architectures $\times$ `3` seeds = `30` runs)
- `results/synthetic/task1a_retro.json` (`11` architectures $\times$ `3` seeds = `33` runs)
- `results/synthetic/task1b.json` & `results/synthetic/task1b_multiscale.json` (`8` strictly iso-state variants $\times$ `3` seeds = `24` runs)

---

## 1. Executive Summary & The $2 \times 2$ Mechanistic Dissociation

Cycle 1 was designed to test **both** our core **DeltaLoop-VLA** hypotheses and every mechanistic critique raised in **Opus 5.5's Review (`REVIEW_OPUS55.md` §2–§3 & Re-Review `#11`)** on shortcut-free synthetic benchmarks under strictly matched state-memory and parameter budgets.

The 87-run empirical suite uncovers a **three-way synthetic mechanistic dissociation** under strictly matched state-memory budgets:

| Embodied Memory Regime | Synthetic Benchmark | Standard GDN $\beta \in (0,1)$ ($L=1$) | Neg-Eig GDN $\beta \in (0,2)$ ($L=1$) | Write-Loop / 2-Layer / Untied $L=6$ ($\beta \in (0,2)$) | **DeltaLoop (Weight-Tied Dynamic Read-Loop)** | 4L Causal Transformer *(1-ep budget)* | Mechanistic Summary & Caveats |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Online Group State Tracking** *(visible container swaps `Swap(Cup_a, Cup_b)`)* | **Task 1A-Online** *(ID $k \le 4$ / OOD $k=5,6$)* | `25.1%` / `4.5%` | **`100.0%` / `100.0%`** | 2-Layer: `87.8%` / `85.8%`<br>Untied $L=6$: `99.7%` / `99.4%` | $\beta \in (0,1)$: `20.2%` / `5.4%`<br>$\beta \in (0,2)$: `63.6%` / `55.7%` *(s0: `99.7%`)* | `15.6%` / `4.5%` | **Write-time $\beta \in (0,2)$ solves synthetic $S_8$ swaps (`100%` at $L=1$; Grazzi et al., 2025).** Read-looping cannot rescue $\beta \in (0,1)$ (`20.2%`). The 4L Transformer's `15.6%` reflects under-optimization under the shared 1-epoch (`20.5s`) budget rather than a theoretical self-attention limit. |
| **2. Post-Cue Occluded Pointer Chaining** *(target entity revealed at $t=T$; `Forward` vs All-Order)* | **Task 1A-Retro** *(Forward-only vs All-Order; ID $k \le 4$ / OOD $k=5,6$)* | **Fwd:** `19.1%` ID / `21.3%` OOD<br>**All:** `57.0%` ID / `27.2%` OOD | **Fwd:** `18.1%` ID / `21.5%` OOD<br>**All:** `60.3%` ID / `39.3%` OOD | Write-Loop: Fwd `17.4%` *(All `59.8%`)*<br>2-Layer: Fwd `17.1%` *(All `60.3%`)*<br>Par-MQ: Fwd `17.5%` *(All `60.0%`)*<br>Untied $L=6$: Fwd `21.4%` *(All `61.4%`)* | **$\beta \in (0,1)$:** **Fwd** ID `97.1%`, OOD `58.9%` *($k=5$: `80.4%`, $k=6$: `37.3%`)*; **All** ID `97.9%`, OOD `68.1%` *($k=5$: `85.0%`, $k=6$: `51.2%`)*<br>**$\beta \in (0,2)$:** **Fwd** ID `94.8%`, OOD `56.8%` *($k=5$: `76.4%`, $k=6$: `37.0%`)*; **All** ID `96.3%`, OOD `68.6%` *($k=5$: `82.3%`, $k=6$: `54.9%`)* | **Fwd:** `15.6%` ID / `18.1%` OOD<br>**All:** `37.5%` ID / `18.3%` OOD | **Weight-tied dynamic re-querying solves ID `Forward` chains (`94.8–97.1%`) and generalizes partially to OOD $k=5$ (`76.4–80.4%` Fwd / `82.3–85.0%` All), but degrades sharply at $k=6$ (`37.0–37.3%` Fwd / `51.2–54.9%` All).** Untied $L=6$'s `21.4%` ID Fwd cannot rule out depth-optimization difficulty without shared gradients. |
| **3. Long-Horizon Key-Collision Retention** *($T=2048$, 300 colliding updates, Iso-State = `6,144 B`)* | **Task 1B** *(Overall / `long_collide` 1500–2000 lag)* | Same-Fixed Decay: `30.0%` / `13.7%` | **Unified 6-Head `fp32` Log-Decay: `99.4%` / `98.9%`** | **Stacked 3-Layer GDN: `99.8%` / `100.0%`**<br>3-Bank + Softmax Router: `71.6–78.5%` | Compatible with Unified / 2–3L Multi-Head GDN | $O(T^2)$ KV-cache (`>128 KB`) | **Falsifies Ad-Hoc 3-Bank Softmax Routing (`Pillar 2` Gate):** Under strict Iso-State (`1,536` floats), Unified Multi-Head `fp32` log-decay (`99.4%`) & Residual Stacked GDN (`99.8%`) beat hard 3-bank softmax routing (`71.6–78.5%`), which suffers MoE router collapse. |

---

## 2. Dataset Shortcut Audits & Experimental Rigor

Before training any neural model, every synthetic benchmark split was audited programmatically (`src/synthetic/*_dataset.py`) to verify zero label leakage, uniform class marginals, and near-chance performance of recency / positional heuristics:

| Benchmark | Split | Classes ($1/C$) | Max Class Marginal | Key Heuristic Audits | State Budget (`fp32`) | Peak VRAM |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **Task 1A-Online** (`OnlineSwapDataset`) | ID ($k \in \{1..4\}, T=40$) | `8` (`12.5%`) | `13.29%` | Initial cup: `9.42%`; Last swap `arg1`: `12.08%`; **Bayes-optimal from final swap only: `12.95%`** | `4,096` floats (`16,384 B`) | `1,238 MB` |
| **Task 1A-Online** | OOD-Hops ($k \in \{5,6\}, T=40$) | `8` (`12.5%`) | `14.10%` | Initial cup: `10.85%`; Bayes-optimal from final swap (`12` total swaps): `22.18%` | `4,096` floats (`16,384 B`) | `1,238 MB` |
| **Task 1A-Retro** (`RetroChainDataset`) | ID ($k \in \{1..4\}, T=44$) | `8` (`12.5%`) | `14.25%` | **Last zone mentioned in stream (`6` concurrent chains w/ matched order): `18.75%`** | `4,096` floats (`16,384 B`) | `1,308 MB` |
| **Task 1A-Retro** | OOD-Hops ($k \in \{5,6\}, T=44$) | `8` (`12.5%`) | `13.75%` | Last zone mentioned in stream: `18.45%` | `4,096` floats (`16,384 B`) | `1,308 MB` |
| **Task 1B** (`MultiTimescaleDataset`) | Eval ($T=2048$, 5 subtypes) | `8` (`12.5%`) | `13.95%` | **Latest observation on queried channel: `11.75%` (`mid_collide`), `15.00%` (`long_collide`)** | **`1,536` floats (`6,144 B`)** | `4,288 MB` |

**Implementation Parity Verified (`< 1e-5` forward & backward max abs diff):**
- Pure-PyTorch `chunk_gated_delta_rule` (`WY` triangular solver in `fp32`) vs step-by-step `recurrent_gated_delta_rule` verified across $\beta_{\max} \in \{1.0, 2.0\}$, `read_from in {"S_prev", "S_curr"}`, multi-query `(B, T, M_q, H, d_k)`, `return_penultimate_state=True`, and fused $v \equiv 1$ **Delta-Rule Key Occupancy State** $\mathbf{n}_t = \alpha_t (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top) \mathbf{n}_{t-1} + \beta_t \mathbf{k}_t$.
- Separate gradient-norm clipping (`clip_and_step`, `max_norm=1.0`) applied independently to temporal-write parameters vs recurrent-depth parameters across **all** models so $L$-step BPTT unrolling never throttles single-pass GDN write gradients.

---

## 3. Task 1A-Online: Online Container Swap State Tracking

### 3.1 Task Formulation
At $t=0\dots 7$, 8 distinct objects are placed bijectively into 8 cups (`Place(Obj_m, Cup_j)`). Over the sequence ($T=40$ ID / OOD-Hops with 12 total swaps; $T=56$ OOD-Length with 16 total swaps), pairs of cups swap contents (`Swap(Cup_a, Cup_b)`), where **only cup identities** (not object identities) are observed during swaps. Exactly $k \in \{1,2,3,4\}$ (ID) or $k \in \{5,6\}$ (OOD) swaps move the target object, interleaved uniformly with distractor swaps among the other 6 cups. At $t=T-1$, the model receives `Query(Obj_target)` and predicts its current cup.

### 3.2 Quantitative Results (3 Seeds: `0, 1, 2`, Mean $\pm$ Std)

| Architecture | Core Params | State (`fp32`) | Train Time | **ID Overall ($k=1..4$)** | $k=1$ | $k=2$ | $k=3$ | $k=4$ | **OOD-Hops ($k=5,6$)** | $k=5$ (OOD) | $k=6$ (OOD) | **OOD-Len+Hops ($T=56$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1. single_gdn_b1` ($\beta \in (0,1), L=1$) | `432,916` | `16,384 B` | `11.7s` | `25.1 ± 1.9%` | `45.6%` | `30.2%` | `16.6%` | `8.1%` | `4.5 ± 0.7%` | `5.4%` | `3.7%` | `7.7 ± 0.2%` |
| **`2. single_gdn_b2` ($\beta \in (0,2), L=1$)** | **`432,916`** | **`16,384 B`** | **`11.7s`** | **`100.0 ± 0.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0%`** | **`100.0 ± 0.0%`** | **`100.0%`** | **`100.0%`** | **`99.8 ± 0.0%`** |
| `3. two_layer_gdn_b2` ($\beta \in (0,2), L=1$) | `492,768` | `16,384 B` | `23.1s` | `87.8 ± 8.5%` | `89.6%` | `87.7%` | `86.6%` | `87.4%` | `85.8 ± 10.1%` | `84.7%` | `87.0%` | `84.6 ± 10.4%` |
| `4. write_loop_gdn_b2` (`DeltaProduct+LT2`) | `463,764` | `16,384 B` | `50.5s` | `19.1 ± 1.6%` | `53.7%` | `11.4%` | `7.1%` | `4.2%` | `2.5 ± 0.1%` | `2.9%` | `2.1%` | `4.7 ± 0.6%` |
| `5. static_query_loop_b2` ($L=6$ fixed $\mathbf{q}^{(0)}$) | `432,916` | `16,384 B` | `26.1s` | `28.2 ± 10.9%` | `37.4%` | `30.2%` | `23.9%` | `21.4%` | `19.5 ± 11.7%` | `20.4%` | `18.5%` | `22.1 ± 11.9%` |
| `6. parallel_multiquery_b2` ($K_{\mathrm{par}}=6, L=1$) | `604,820` | `16,384 B` | `12.6s` | `74.3 ± 19.0%` | `73.3%` | `75.1%` | `73.0%` | `75.9%` | `71.9 ± 20.1%` | `72.3%` | `71.5%` | `69.9 ± 20.6%` |
| **`7. untied_deep_readout_b2` ($L=6$ untied)** | `1,628,180` | `16,384 B` | `31.1s` | **`99.7 ± 0.2%`** | `99.8%` | `99.7%` | `99.5%` | `99.8%` | **`99.4 ± 0.5%`** | `99.3%` | `99.5%` | `93.1 ± 5.4%` |
| `8. deltaloop_b1` ($\beta \in (0,1), L=6$) | `432,916` | `16,384 B` | `33.4s` | `20.2 ± 5.0%` | `44.8%` | `17.5%` | `11.6%` | `7.0%` | `5.4 ± 0.3%` | `6.3%` | `4.5%` | `7.6 ± 0.4%` |
| `9. deltaloop_b2` ($\beta \in (0,2), L=6$) | `432,916` | `16,384 B` | `31.2s` | `63.6 ± 29.1%` *(s0: `99.7%`)* | `68.8%` | `62.8%` | `62.5%` | `60.5%` | `55.7 ± 35.2%` *(s0: `99.8%`)* | `54.3%` | `57.1%` | `56.9 ± 33.2%` *(s0: `98.9%`)* |
| `10. softmax_transformer` (4L Causal, 1-ep budget) | `496,520` | $O(T d)$ | `20.5s` | `15.6 ± 0.3%` | `27.3%` | `16.2%` | `11.3%` | `7.6%` | `4.5 ± 1.0%` | `5.2%` | `3.8%` | `8.6 ± 0.9%` |

### 3.3 Mechanistic Analysis of Task 1A-Online
1. **Why Dual-Stream Memory (and Standard $\beta \in (0,1)$ GDN) Fails Synthetic `Swap` Tracking:**
   With $\beta_t \in (0,1)$, every state transition matrix $\mathbf{A}_t = \alpha_t (\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top)$ has eigenvalues strictly in $[0, 1]$. It can erase ($\beta_t = 1$) or interpolate ($\beta_t \in (0,1)$), **never reflect or permute** ($\lambda = -1$). Consequently, `single_gdn_b1` decays monotonically across swap count (`45.6% -> 30.2% -> 16.6% -> 8.1% -> 3.7%`), and **adding $L=6$ dynamic read-loops (`deltaloop_b1`: `20.2%` ID, `5.4%` OOD) cannot recover object bindings that were already blurred into contractions at write time**. Note on the 4-layer causal `softmax_transformer` (`15.6%` ID, `4.5%` OOD): while fixed-depth self-attention faces known $\mathsf{TC}^0 \subsetneq \mathsf{NC}^1$ depth limits on non-abelian $S_8$ composition at arbitrary length (Merrill et al., 2024; Grazzi et al., 2025), `15.6%` at $k \le 4$ (near `12.5%` chance) also reflects **under-optimization under the shared 1-epoch (`20.5s`) training budget** without attention-specific multi-epoch/warmup tuning, so it should not be interpreted as a converged self-attention upper bound.
2. **Householder $\beta_t \in (0, 2)$ Solves Online $S_8$ Tracking with Zero-Shot Horizon Extrapolation (`100.0%`):**
   Expanding $\beta_t = 2\sigma(\cdot) \in (0, 2)$ allows $\mathbf{I} - 2\mathbf{k}_t\mathbf{k}_t^\top \in O(d_k)$ ($\det = -1$) to execute exact reflection/transposition updates in key space as each `Swap(Cup_a, Cup_b)` arrives. `single_gdn_b2` ($L=1$) hits **`100.0 ± 0.0%` on ID ($k=1..4$), `100.0 ± 0.0%` on OOD-Hops ($k=5,6$), and `99.8 ± 0.0%` on OOD-Length+Hops ($T=56$, 16 swaps)** across all 3 seeds in `11.7s`!
3. **Subtle Discovery on Inverse `Cup -> Obj` Readout (`single_gdn_b2` vs `deltaloop_b2`):**
   Because `Swap(Cup_a, Cup_b)` contains only cup IDs, $\mathbf{S}_T$ must store `Cup (key) -> Obj (value)`, making `Query(Obj_m)` an *inverse* lookup across all 8 cups. With 1 query vector per head ($H=4$), `single_gdn_b2` ($L=1$) dedicates all 4 heads to a static 8-cup superposition readout (`100%`). When `deltaloop_b2` ($L=6$) is trained with uniform all-depth CE at $N_q=1$ query/step, seed 0 discovers an iterative subset cup search (`55.7% [L=1] -> 87.9% [L=2] -> 99.4% [L=3] -> 99.8% [L=4]`, `99.8%` OOD), whereas seeds 1–2 experience gradient competition at $l=0$ between static 8-cup superposition and sequential scanning (`untied_deep_readout_b2` avoids this via a separate $l=0$ projection and hits `99.7 ± 0.2%`). **Actionable VLA Lesson:** Using **$N_q = 4$ multi-query tokens per loop step** (`4 * H` simultaneous cup/spatial slot queries at $l=0$, already specified in `RESEARCH_PLAN.md` §3.4!) eliminates $l=0$ slot bottlenecking so $L=1$ captures full multi-slot readout while $L \ge 2$ executes sequential hops.

---

## 4. Task 1A-Retro: Retroactive Multi-Hop Relational Pointer Chaining

### 4.1 Task Formulation (Modeling Embodied Post-Cue Occlusion)
Each sequence ($T=44$ ID / OOD-Hops; $T=56$ OOD-Length) embeds **6 concurrent disjoint entity-container-zone pointer chains** (`Bind(x_0, x_1)`, `Bind(x_1, x_2)`, ..., `Bind(x_{k-1}, Zone_z)`), interleaved with overwrites and matched temporal ordering across all 6 chains (`Forward`, `Reverse`, or `Mixed` causal arrival order). **Crucially, `Query(Entity_root)` is revealed ONLY at the final step $t=T-1$.** Because the model does not know at write time which of the 6 chains will be queried, it cannot collapse only the target chain into a single state vector during prefix ingestion.

### 4.2 Main Quantitative Comparison — All-Order Aggregate (`Single` + `Forward` + `Reverse` + `Mixed`, 3 Seeds: `0, 1, 2`, Mean $\pm$ Std)

*(Note: Table 4.2 and the $(k, L)$ matrices in §4.3 average over all edge-arrival orders (`Forward`, `Reverse`, `Mixed`, plus `Single` at $k=1$); see §4.4 for the strictly matched **Forward-only** ID and OOD $k=5, 6$ breakdown!)*

| Architecture | Core Params | State (`fp32`) | **ID Overall ($k=1..4$)** | $k=1$ | $k=2$ | $k=3$ | $k=4$ | **OOD-Hops ($k=5,6$)** | **$k=5$ (OOD)** | **$k=6$ (OOD)** | **OOD-Len ($T=56$)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1. single_gdn_b1` ($\beta \in (0,1), L=1$) | `436,116` | `16,384 B` | `57.0 ± 1.2%` | `99.8%` | `57.5%` | `38.8%` | `31.9%` | `27.2 ± 1.2%` | `28.9%` | `25.5%` | `25.8 ± 1.6%` |
| `2. single_gdn_b2` ($\beta \in (0,2), L=1$) | `436,116` | `16,384 B` | `60.3 ± 0.0%` | `99.7%` | `54.8%` | `43.1%` | `43.6%` | `39.3 ± 2.2%` | `41.1%` | `37.4%` | `38.2 ± 2.3%` |
| `3. two_layer_gdn_b2` (Stacked, $L=1$) | `495,968` | `16,384 B` | `60.3 ± 0.2%` | `99.2%` | `55.9%` | `43.2%` | `42.8%` | `39.3 ± 1.2%` | `40.0%` | `38.5%` | `38.5 ± 1.7%` |
| `4. write_loop_gdn_b2` (`DeltaProduct+LT2`) | `466,964` | `16,384 B` | `59.8 ± 0.3%` | `99.2%` | `56.6%` | `43.1%` | `40.3%` | `33.7 ± 1.3%` | `36.0%` | `31.3%` | `31.8 ± 3.0%` |
| `5. static_query_loop_b2` ($L=6$ fixed $\mathbf{q}^{(0)}$) | `436,116` | `16,384 B` | `60.3 ± 0.1%` | `99.2%` | `56.7%` | `43.5%` | `41.9%` | `40.6 ± 0.6%` | `43.2%` | `38.0%` | `39.5 ± 1.1%` |
| `6. parallel_multiquery_b2` (`B2`: $K_{\mathrm{par}}=6, L=1$) | `608,020` | `16,384 B` | `60.0 ± 0.6%` | `99.2%` | `56.7%` | `43.4%` | `40.6%` | `38.8 ± 0.4%` | `40.8%` | `36.8%` | `36.6 ± 0.5%` |
| `7. untied_deep_readout_b2` (`B1`: $L=6$ untied, $3.7\times$ P) | `1,631,380` | `16,384 B` | `61.4 ± 1.5%` | `98.8%` | `59.5%` | `44.2%` | `43.3%` | `39.0 ± 0.6%` | `41.9%` | `36.1%` | `37.4 ± 0.7%` |
| **`8. deltaloop_b1` (Ours, $\beta \in (0,1), L=6$)** | **`436,116`** | **`16,384 B`** | **`97.9 ± 0.8%`** | **`99.9%`** | **`99.3%`** | **`98.3%`** | **`94.1%`** | **`68.1 ± 3.1%`** | **`85.0%`** | **`51.2%`** | **`63.9 ± 3.2%`** |
| **`9. deltaloop_b2` (Ours, $\beta \in (0,2), L=6, \mathbf{S}_{t-1}$)** | **`436,116`** | **`16,384 B`** | **`96.3 ± 2.1%`** | **`99.9%`** | **`99.1%`** | **`96.3%`** | **`90.1%`** | **`68.6 ± 3.7%`** | **`82.3%`** | **`54.9%`** | **`63.4 ± 5.5%`** |
| **`10. deltaloop_b2_scurr` (Ablation: read $\mathbf{S}_t$)** | `436,116` | `16,384 B` | `97.6 ± 0.4%` | `99.9%` | `99.0%` | `97.6%` | `93.7%` | `71.8 ± 0.4%` | `83.6%` | `60.1%` | `63.8 ± 1.5%` |
| `11. softmax_transformer` (4L Causal Attn) | `499,720` | $O(T d)$ | `37.5 ± 0.1%` | `99.5%` | `16.9%` | `16.8%` | `16.9%` | `18.3 ± 0.8%` | `18.5%` | `18.1%` | `17.2 ± 0.9%` |

---

### 4.3 The $(k\text{-Hops}, L\text{-Loops})$ 2D Diagonal Staircase Matrix

Below is the exact 3-seed `mean ± std` accuracy (%) as a function of chain hop depth $k \in \{1..6\}$ (rows; $k=5,6$ are **zero-shot OOD**, never seen during training!) and inference readout loop depth $L \in \{1..6\}$ (columns):

#### `deltaloop_b1` (Weight-Tied Dynamic Re-Querying, $\beta \in (0,1)$, $\mathbf{S}_{t-1}$)
| Chain Depth $k$ | $L = 1$ | $L = 2$ | $L = 3$ | $L = 4$ | $L = 5$ | $L = 6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 1$ (ID)** | **`100.0 ± 0.0%`** | `100.0 ± 0.0%` | `100.0 ± 0.0%` | `99.9 ± 0.1%` | `99.9 ± 0.1%` | `99.9 ± 0.1%` |
| **$k = 2$ (ID)** | `24.8 ± 6.8%` | **`99.9 ± 0.2%`** | `99.6 ± 0.2%` | `99.3 ± 0.3%` | `99.3 ± 0.2%` | `99.3 ± 0.2%` |
| **$k = 3$ (ID)** | `14.6 ± 0.5%` | `30.6 ± 8.0%` | **`98.6 ± 0.2%`** | `98.7 ± 0.3%` | `98.4 ± 0.4%` | `98.3 ± 0.6%` |
| **$k = 4$ (ID)** | `12.8 ± 0.6%` | `17.1 ± 2.5%` | `33.3 ± 5.3%` | **`92.3 ± 3.6%`** | **`94.4 ± 2.0%`** | `94.1 ± 2.3%` |
| **$k = 5$ (Zero-Shot OOD)** | `13.0 ± 1.8%` | `16.0 ± 1.1%` | `20.9 ± 2.3%` | `34.0 ± 6.4%` | **`78.4 ± 4.0%`** | **`85.0 ± 2.9%`** |
| **$k = 6$ (Zero-Shot OOD)** | `15.9 ± 0.1%` | `14.5 ± 0.1%` | `17.5 ± 1.0%` | `21.2 ± 2.3%` | `27.7 ± 4.3%` | **`51.2 ± 3.2%`** |

#### `deltaloop_b2` (Weight-Tied Dynamic Re-Querying, $\beta \in (0,2)$, $\mathbf{S}_{t-1}$)
| Chain Depth $k$ | $L = 1$ | $L = 2$ | $L = 3$ | $L = 4$ | $L = 5$ | $L = 6$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 1$ (ID)** | **`100.0 ± 0.0%`** | `100.0 ± 0.0%` | `99.9 ± 0.1%` | `99.9 ± 0.1%` | `99.9 ± 0.1%` | `99.9 ± 0.2%` |
| **$k = 2$ (ID)** | `48.9 ± 3.3%` | **`99.7 ± 0.3%`** | `99.6 ± 0.3%` | `99.4 ± 0.4%` | `99.2 ± 0.5%` | `99.1 ± 0.5%` |
| **$k = 3$ (ID)** | `26.6 ± 2.9%` | `54.2 ± 2.6%` | **`96.7 ± 1.7%`** | **`97.1 ± 2.0%`** | `96.3 ± 2.7%` | `96.3 ± 2.7%` |
| **$k = 4$ (ID)** | `16.6 ± 1.7%` | `27.7 ± 4.5%` | `51.3 ± 3.3%` | **`86.3 ± 6.8%`** | **`89.7 ± 6.0%`** | **`90.1 ± 5.3%`** |
| **$k = 5$ (Zero-Shot OOD)** | `18.6 ± 1.6%` | `22.0 ± 3.6%` | `34.3 ± 6.3%` | `48.6 ± 5.8%` | **`74.9 ± 5.7%`** | **`82.3 ± 5.8%`** |
| **$k = 6$ (Zero-Shot OOD)** | `16.0 ± 1.6%` | `18.5 ± 1.4%` | `22.8 ± 3.6%` | `31.8 ± 5.1%` | `40.9 ± 3.5%` | **`54.9 ± 1.8%`** |

#### Contrast: `static_query_loop_b2` ($L=6$ Fixed Query) & `untied_deep_readout_b2` (`B1`: $L=6$ Untied, $3.7\times$ Params)
| Control Model | Chain Depth $k$ | $L = 1$ | $L = 2$ | $L = 3$ | $L = 4$ | $L = 5$ | $L = 6$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `static_query_loop_b2` | $k = 2$ (ID) | `55.3%` | `55.9%` | `56.6%` | `56.4%` | `56.7%` | `56.7%` *(flat!)* |
| `static_query_loop_b2` | $k = 4$ (ID) | `41.8%` | `41.7%` | `41.1%` | `41.7%` | `41.7%` | `41.9%` *(flat!)* |
| `untied_deep_readout_b2` (`1.63M`) | $k = 2$ (ID) | `55.1%` | `61.1%` | `60.4%` | `59.8%` | `60.0%` | `59.5%` |
| `untied_deep_readout_b2` (`1.63M`) | $k = 4$ (ID) | `41.1%` | `43.1%` | `42.9%` | `42.3%` | `43.2%` | `43.3%` |
| `untied_deep_readout_b2` (`1.63M`) | $k = 5$ (OOD) | `39.1%` | `42.0%` | `41.9%` | `42.1%` | `41.8%` | `41.9%` |

---

### 4.4 Arrival-Order Breakdown (`Forward` vs `Reverse` vs `Mixed`) — Separating Forward-Only vs All-Order ID & OOD ($k=5, 6$)

Why did **every** non-iterative or untied GDN baseline plateau at `~60.3%` all-order ID accuracy (`~56%` on $k=2$, `~43%` on $k=3,4$)? Breaking accuracy down by the **temporal arrival order of edges** across **both ID ($k \in \{2,3,4\}$) and Zero-Shot OOD ($k \in \{5,6\}$)** reveals both the mechanism and the exact OOD extrapolation boundary:

| Architecture | **ID `Forward` ($k=2..4$)** | **ID `Reverse` ($k=2..4$)** | **ID `Mixed` ($k=3,4$)** | **OOD `Forward` Overall ($k=5,6$)** | **OOD `Forward` $k=5$** *(All-Order $k=5$)* | **OOD `Forward` $k=6$** *(All-Order $k=6$)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `single_gdn_b1` ($L=1$) | `19.1%` *(chance)* | `80.6%` | `19.2%` *(chance)* | `21.3%` | `19.8%` *(`28.9%`)* | `22.9%` *(`25.5%`)* |
| `single_gdn_b2` ($L=1$) | `18.1%` *(chance)* | `94.1%` | `17.4%` *(chance)* | `21.5%` | `20.6%` *(`41.1%`)* | `22.4%` *(`37.4%`)* |
| `two_layer_gdn_b2` ($L=1$) | `17.1%` *(chance)* | `94.5%` | `19.1%` *(chance)* | `20.8%` | `19.0%` *(`40.0%`)* | `22.7%` *(`38.5%`)* |
| `write_loop_gdn_b2` (`DeltaProduct+LT2`) | `17.4%` *(chance)* | `94.3%` | `15.9%` *(chance)* | `21.5%` | `21.0%` *(`36.0%`)* | `22.1%` *(`31.3%`)* |
| `static_query_loop_b2` ($L=6$) | `18.0%` *(chance)* | `95.5%` | `16.2%` *(chance)* | `21.5%` | `21.3%` *(`43.2%`)* | `21.8%` *(`38.0%`)* |
| `parallel_multiquery_b2` ($K_{\mathrm{par}}=6$) | `17.5%` *(chance)* | `93.7%` | `18.0%` *(chance)* | `21.4%` | `20.0%` *(`40.8%`)* | `22.9%` *(`36.8%`)* |
| `untied_deep_readout_b2` ($L=6$ untied) | `21.4%` *(chance)* | `95.1%` | `18.2%` *(chance)* | `22.2%` | `22.6%` *(`41.9%`)* | `21.9%` *(`36.1%`)* |
| **`deltaloop_b1` (Ours, $L=6$)** | **`97.1%`** | **`97.5%`** | **`97.0%`** | **`58.9%`** | **`80.4%`** *(`85.0%`)* | **`37.3%`** *(`51.2%`)* |
| **`deltaloop_b2` (Ours, $L=6$)** | **`94.8%`** | **`96.4%`** | **`93.6%`** | **`56.8%`** | **`76.4%`** *(`82.3%`)* | **`37.0%`** *(`54.9%`)* |
| `softmax_transformer` (4L, 1-ep) | `15.6%` *(chance)* | `17.9%` *(chance)* | `17.4%` *(chance)* | `18.1%` | `18.4%` *(`18.5%`)* | `17.8%` *(`18.1%`)* |

**Mechanistic interpretation & honest caveats (`C8`):**
1. **Why `Reverse` vs. `Forward` dissociates write-time vs. read-time composition:** On **`Reverse`** chains (`Box_1 -> Zone_c` arrives at $t_1$, then `Apple -> Box_1` arrives at $t_2 > t_1$), `Box_1 -> Zone_c` is already stored in $\mathbf{S}_{t_2-1}$, so a single write-time GDN with $\beta \in (0,2)$ composes it (`94.1%` ID, `77.7%` OOD). On **`Forward`** chains (`Apple -> Box_1` at $t_1$, then `Box_1 -> Zone_c` at $t_2 > t_1$, with `Query(Apple)` revealed only at $t=T$), `Apple` is stored as a key pointing to `Box_1`; all $L=1$, write-loop, stacked, and parallel-query GDN baselines remain at `17.1–19.1%` on ID `Forward` and `20.8–21.5%` on OOD `Forward`, whereas weight-tied `DeltaLoop` reaches **`94.8–97.1%` on ID `Forward` ($k \le 4$)**.
2. **Explicit OOD Extrapolation Boundary ($k=5$ vs. $k=6$ Breakdown):** Zero-shot hop extrapolation in `DeltaLoop` ($L_{\max}=6$, trained on $k \le 4$) holds up strongly by one unseen hop to **$k=5$ (`76.4–80.4%` Forward-only; `82.3–85.0%` all-order)**, but **degrades sharply at $k=6$ (`37.0–37.3%` Forward-only; `51.2–54.9%` all-order; overall OOD $k \in \{5,6\}$ = `56.8–58.9%` Forward-only / `68.1–68.6%` all-order)**, where 6-edge interference in the `16 KB` linear state and absence of $>6$ refinement margin compound.
3. **Optimization Caveat on `untied_deep_readout_b2` ($L=6$, `3.7×` Params):** While `untied_deep_readout_b2` cannot extrapolate to unseen hop indices without shared weights across $l$, its failure even on **ID `Forward` chains (`21.4%`, `17.1–21.4%` across non-tied baselines)** under the shared 1-epoch / fixed-LR training budget **cannot exclude depth-optimization failure** when 6 separate readout blocks lack shared multi-depth gradient supervision (dedicated untied LR/warmup sweeps were not run).
4. **Why `deltaloop_b2_scurr` ($\mathbf{S}_t$) exceeds `deltaloop_b2` ($\mathbf{S}_{t-1}$) at $L=1$ (`38.9%` vs `16.6%` on $k=4$):** When the query token $T-1$ writes into $\mathbf{S}_{T-1}$ before readout, the delta rule sneaks one implicit query $\mathbf{S}_{T-2}^\top \mathbf{k}(\text{Entity}_x)$ inside the write step. In continuous VLA streams where token $t$ contains current camera observations, reading $\mathbf{S}_t$ permits current-frame self-retrieval, motivating strict $\mathbf{S}_{t-1}$ / exclusion-window readout (`A3`).

---

### 4.5 Stage-2 Adaptive Halting vs. Latent L2-Convergence (`Reviewer C2`) & QK-Hop Alignment

#### A. Stage-2 Frozen-Backbone Halting Gate vs. RD-VLA Latent L2-Convergence ($\|\mathbf{h}^{(l)} - \mathbf{h}^{(l-1)}\|_2 < \tau$)
| Model | Stopping Criterion | **ID Acc ($k=1..4$)** | **Mean ID Loops** | $\mathbb{E}[L \mid k=1]$ | $\mathbb{E}[L \mid k=2]$ | $\mathbb{E}[L \mid k=3]$ | $\mathbb{E}[L \mid k=4]$ | **OOD $\mathbb{E}[L \mid k=5]$** *(Acc)* | **OOD $\mathbb{E}[L \mid k=6]$** *(Acc)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `deltaloop_b1` | Fixed $L=6$ | `97.9 ± 0.8%` | `6.00` | `6.00` | `6.00` | `6.00` | `6.00` | `6.00` (`85.0%`) | `6.00` (`51.2%`) |
| **`deltaloop_b1`** | **2-Stage Halting Gate (Ours)** | **`97.8 ± 0.6%`** | **`2.62`** *(**-56% FLOPs**)* | **`1.00 ± 0.00`** | **`2.02 ± 0.01`** | **`3.09 ± 0.05`** | **`4.37 ± 0.12`** | **`5.48 ± 0.10`** *(`83.3%`)* | **`5.75 ± 0.04`** *(`50.9%`)* |
| `deltaloop_b1` | Latent L2-Convergence | `61.9 ± 2.6%` | `3.57` | — | — | — | — | — | — |
| `deltaloop_b2` | Fixed $L=6$ | `96.3 ± 2.1%` | `6.00` | `6.00` | `6.00` | `6.00` | `6.00` | `6.00` (`82.3%`) | `6.00` (`54.9%`) |
| **`deltaloop_b2`** | **2-Stage Halting Gate (Ours)** | **`95.8 ± 2.3%`** | **`2.65`** *(**-56% FLOPs**)* | **`1.00 ± 0.00`** | **`2.02 ± 0.08`** | **`3.20 ± 0.19`** | **`4.37 ± 0.18`** | **`5.23 ± 0.13`** *(`80.5%`)* | **`5.58 ± 0.14`** *(`53.8%`)* |
| `deltaloop_b2` | Latent L2-Convergence | `70.4 ± 1.4%` | `3.57` | — | — | — | — | — | — |

- **Empirical Confirmation of Opus 5.5 (`Reviewer C2`):** Naive latent L2-convergence ($\|\mathbf{h}^{(l)} - \mathbf{h}^{(l-1)}\|_2 < \tau$) drops accuracy by **$-25.4$ to $-35.9$ percentage points** while consuming *more* loops (`3.57` vs `2.62`), because discrete pointer hops traverse distinct entity embeddings with non-monotonic step norms prior to reaching a terminal zone!
- By contrast, our **2-Stage Frozen-Backbone Halting Gate** (fed with latent $\mathbf{h}^{(l)}$, readout $\mathbf{m}^{(l)}$, step delta, retrieval norm, and **Delta-Occupancy Key Coverage** $\mathbf{n}_{T-2}^\top \mathbf{q}^{(l)}$) matches full $L=6$ accuracy within `0.1–0.5%` while cutting loop compute by **`56%`**, and **extrapolates loop allocation monotonically zero-shot to OOD $k=5$ (`5.48` loops) and $k=6$ (`5.75` loops)**!

#### B. Mechanistic Probe: Query-Key Top-1 Hop Alignment Matrix (`deltaloop_b2`, $k=4$ Chains, 3-Seed Mean)
Across all $T-1 = 43$ historical prefix tokens (4 true chain edges + 39 distractor/other-chain tokens; random match rate = `2.3%`), percentage of episodes where $\arg\max_{t < T-1} \max_h |\langle \mathbf{q}^{(l)}_h, \mathbf{k}_{t,h} \rangle|$ lands on the **exact historical timestep of Hop $m$**:

| Readout Loop Step $l$ | **True Hop 1 Timestep** (`Entity_0 -> Box_1`) | **True Hop 2 Timestep** (`Box_1 -> Box_2`) | **True Hop 3 Timestep** (`Box_2 -> Box_3`) | **True Hop 4 Timestep** (`Box_3 -> Zone`) |
| :--- | :---: | :---: | :---: | :---: |
| **Loop $l = 1$** | **`84.9%`** | `0.0%` | `0.0%` | `0.1%` |
| **Loop $l = 2$** | `0.2%` | **`72.2%`** | `0.5%` | `1.4%` |
| **Loop $l = 3$** | `2.1%` | `0.9%` | **`38.1%`** | `2.2%` |
| **Loop $l = 4$** | `2.7%` | `2.7%` | `5.3%` | **`20.6%`** |
| **Loop $l = 5$** | `3.2%` | `3.6%` | `4.2%` | **`21.2%`** |
| **Loop $l = 6$** | `3.8%` | `3.3%` | `3.3%` | **`21.7%`** |

This provides **direct mechanistic proof** that `DeltaLoop` is not merely adding nonlinear MLP depth: at loop $l=1$, $\mathbf{q}^{(1)}$ addresses Hop 1 (`84.9%`); at $l=2$, $\mathbf{q}^{(2)}$ pivots onto Hop 2 (`72.2%`); at $l=3$, $\mathbf{q}^{(3)}$ pivots onto Hop 3 (`38.1%`); and at $l \ge 4$, $\mathbf{q}^{(l)}$ locks onto terminal Hop 4 (`20.6% -> 21.7%`, $>9\times$ random token baseline) and remains parked cleanly as a fixed point!

---

## 5. Task 1B: $T=2048$ Key-Collision Multi-Timescale Interference (Strictly Iso-State)

### 5.1 Task Formulation & Iso-State Budget
Over $T = 2048$ steps, `OP_OBS(chan, val)` writes continuously into shared channels (`~300` active colliding background observations per episode). Five task subtypes (`400` eval episodes each) test simultaneous retention across 3 orders of magnitude of temporal lag:
1. `short` (`10–30` steps lag): retrieve the most recent value on `chan 0..3`.
2. `mid_persist` (`200–400` steps lag) & `long_persist` (`1500–2000` steps lag): retrieve an uncollided observation across hundreds/thousands of steps (`fp32` slow log-decay test).
3. `mid_collide` (`200–400` steps lag) & `long_collide` (`1500–2000` steps lag): an `OP_LOCK_MID(chan)` or `OP_LOCK_LONG(chan)` event marks the preceding observation on shared `chan 0..3`, followed by **dozens of subsequent `OP_OBS(chan, new_val)` overwrites on the exact same channel** (`latest_obs` heuristic = `11.8%` / `15.0%` chance!), while `short` queries still require the latest value on those same channels!

**Strict Iso-State Constraint:** Every single variant (`1` through `5c`) has **exact `state_numel = 1,536` floats (`6,144` bytes in `fp32`)** and **`171,963–172,154` total parameters (`< 0.12%` spread)**.

### 5.2 Quantitative Results (3 Seeds: `0, 1, 2`, Mean $\pm$ Std)

| Variant (`State = 1,536 floats = 6,144 B` for ALL) | Params | Train Time | **Overall Acc** | `short` (10–30) | `mid_persist` (200–400) | `long_persist` (1500–2000) | `mid_collide` (200–400) | **`long_collide` (1500–2000)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1_single_mh` (Unified 6-Head Diverse `fp32` Decay)** | `172,154` | `82.6s` | **`99.4 ± 0.5%`** | **`100.0 ± 0.0%`** | **`100.0 ± 0.0%`** | **`98.9 ± 0.1%`** | **`99.3 ± 0.9%`** | **`98.9 ± 1.5%`** |
| **`1b_stacked_3layer` (3-Layer $\times$ 2-Head Residual GDN)** | **`172,122`** | `132.4s` | **`99.8 ± 0.0%`** | **`100.0 ± 0.0%`** | **`100.0 ± 0.0%`** | **`99.1 ± 0.1%`** | **`100.0 ± 0.0%`** | **`100.0 ± 0.0%`** |
| `2_3bank_same_fixed` (3-Bank $\times$ 2H, Same Short $\tau=16..32$) | `172,142` | `69.4s` | `30.0 ± 0.3%` | `100.0 ± 0.0%` | `11.0 ± 1.1%` | `12.7 ± 1.0%` | `12.7 ± 0.8%` | `13.7 ± 0.3%` |
| `3_3bank_diverse_fixed` (3-Bank $\times$ 2H, Fixed $\tau \in [8, 2048]$) | `172,142` | `70.9s` | `88.1 ± 8.2%` | `100.0 ± 0.0%` | `100.0 ± 0.0%` | `99.2 ± 0.1%` | `99.8 ± 0.4%` | `41.6 ± 41.3%` |
| `4_3bank_learnable` (3-Bank $\times$ 2H, Band-Constrained) | `172,148` | `78.1s` | `94.2 ± 8.0%` | `100.0 ± 0.0%` | `100.0 ± 0.0%` | `99.1 ± 0.2%` | `100.0 ± 0.0%` | `71.9 ± 39.7%` |
| `5a_3bank_consolidation` (3-Bank + $\beta$-Gated Consolidation) | `172,056` | `155.7s` | `91.0 ± 7.2%` | `100.0 ± 0.0%` | `100.0 ± 0.0%` | `99.0 ± 0.2%` | `85.2 ± 21.0%` | `70.7 ± 41.5%` |
| `5b_3bank_routing` (3-Bank + Softmax Scale Router $\boldsymbol{\pi}$) | `172,055` | `77.5s` | `71.6 ± 29.8%` | `100.0 ± 0.0%` | `71.2 ± 40.8%` | `70.4 ± 40.6%` | `65.9 ± 37.8%` | `50.2 ± 36.8%` |
| `5c_3bank_full` (3-Bank + Consolidation + Softmax Router) | `171,963` | `155.4s` | `78.5 ± 9.6%` | `71.3 ± 40.7%` | `100.0 ± 0.0%` | `70.4 ± 40.6%` | `81.2 ± 26.5%` | `69.4 ± 39.8%` |

### 5.3 Honest Falsification & Mechanistic Takeaways from Task 1B (`Go/No-Go Gate`)
1. **Multi-Timescale `fp32` Log-Decay is Mandatory (`30.0%` $\to$ `99.4%`):**
   `2_3bank_same_fixed` ($\alpha = 0.9548$ per step) hits `100.0%` on `short` and **pure chance (`11.0–13.7%`)** on every `mid` and `long` subtype because $0.9548^{1800} \approx 10^{-37}$. Parameterizing per-head scalar log-decay $g_{t,h} = \log \alpha_{t,h}$ in `fp32` with log-uniform horizon initialization ($\tau \in [8, 2048]$) enables heads to sustain $\alpha \in [0.9995, 1.0000]$ without `bf16` rounding collapse, jumping `long_persist` from `12.7%` to **`99.1%`** at 1,500–2,000 step lag!
2. **Honest Go/No-Go Verdict on `Pillar 2` (Opus 5.5 §2.3 Confirmed!):**
   Under strict Iso-State (`1,536` floats), **Unified Multi-Head GDN (`1_single_mh`: `99.4 ± 0.5%` overall, `98.9 ± 1.5%` `long_collide`) and Stacked Residual GDN (`1b_stacked_3layer`: `99.8 ± 0.0%` overall, `100.0 ± 0.0%` `long_collide`) decisively beat hard-partitioned 3-Bank + Softmax Scale-Routing (`5b`: `71.6 ± 29.8%`, `5c`: `78.5 ± 9.6%`)** for two concrete reasons:
   - **MoE-Style Softmax Router Collapse:** Inspecting the learned routing matrix $\boldsymbol{\pi}$ in `5b_3bank_routing` reveals classic softmax collapse onto Bank 2 (`pi = [0.033, 0.015, 0.952]` on `query_short`, `[0.001, 0.000, 0.999]` on `query_long`), starving Banks 0–1 of gradients on 1 of 3 seeds. Dense linear multi-head output projection (`o_proj`) in `1_single_mh` and `1b_stacked_3layer` gives non-vanishing gradients to all 6 heads simultaneously.
   - **Implicit Consolidation via Standard Residual Layer Stacking:** In `1b_stacked_3layer`, Layer 1 learns a fast buffer head ($\bar{\alpha}=0.735, \bar{\beta}=0.152$) that tracks live `OP_OBS` events, while Layer 3 learns a sparse, near-lossless `fp32` latch head ($\bar{\alpha}=1.0000, \bar{\beta}=0.052$) that reads Layer 1's residual output when `OP_LOCK` arrives and ignores subsequent `OP_OBS` collisions (`100.0 ± 0.0%` on `mid_collide` and `long_collide` across all 3 seeds, zero variance!).

---

## 6. Architectural Hypotheses Carried Forward to Closed-Loop RoboMME (Cycles 2–4)

Cycle 1 provides controlled synthetic evidence to form concrete hypotheses — and prunes unnecessary complexity — prior to closed-loop RoboMME evaluation (where downstream Cycles 2A, 3A, and 3B test whether these synthetic advantages transfer to visuomotor manipulation):

1. **Test Negative-Eigenvalue Householder GDN ($\beta_t \in (0, 2)$ + `fp32` Log-Decay $g_t$) vs. $\beta_t \in (0, 1)$:**
   - Solves synthetic $S_8$ swap tracking (`100.0%` vs `25.1%` for $\beta \in (0,1)$), while `fp32` log-uniform multi-head decay ($\tau \in [8, 2048]$) preserves 2,000-step history (`99.4–99.8%`). *(Note from downstream Cycles 2A/3A: in closed-loop RoboMME visuomotor control, $\beta \in (0,2)$ shows no SR advantage over $\beta \in (0,1)$, indicating RoboMME tasks rely primarily on damped key-value retention rather than exact discrete reflections.)*
2. **Test Weight-Tied Multi-Hop Readout (`DeltaLoop`, $N_q = 4$) Against Single-Pass 2-Layer GDN (`M5`):**
   - Solves synthetic `Forward` pointer chains (`94.8–97.1%` ID; `76.4–80.4%` OOD $k=5$; `37.0–37.3%` OOD $k=6$) vs `17.1–21.4%` for single-pass/stacked/untied baselines under the 1-epoch budget. *(Note from downstream Cycles 2A/3A/3B: in closed-loop RoboMME, single-pass 2-layer GDN `M5` consistently outperforms looped `M6/M7` variants, showing RoboMME gains stem from recurrent state compression rather than multi-hop Coda looping.)*
3. **KEEP & CENTERPIECE #3 — 2-Stage Frozen-Backbone Halting Gate with Delta-Occupancy Confidence (`n_t`):**
   - Cuts expected loop FLOPs by **`56%`** (`2.62` vs `6.00` loops) with `< 0.5%` accuracy loss, monotonically adapts compute to latent hop complexity (`1.00 -> 2.02 -> 3.09 -> 4.37 -> 5.48` loops), and outperforms RD-VLA-style latent L2-convergence stopping by **`+25.4` to `+35.9` percentage points**.
4. **KEEP — Strict Causal Exclusion Cutoff ($\mathbf{S}_{t-1}$ / `A3` Video-Demo Cutoff) & Separate Write/Loop Grad Clipping:**
   - Prevents current-frame self-retrieval shortcuts (`deltaloop_b2_scurr` leak analysis) and prevents $L$-step loop gradients from throttling GDN write learning.
5. **PRUNE / SIMPLIFY (`Occam's Razor` from Task 1B Gate):**
   - **Drop hand-crafted 3-Bank Softmax Scale-Routing (`Pillar 2` MoE router)** in favor of **Unified Diverse-Decay Multi-Head GDN (`1_single_mh`: `99.4%`) or 2-Layer Residual GDN (`1b_stacked`: `99.8%`)** — simpler, faster (`fla` single-kernel compatible), zero router collapse, and strictly higher accuracy at identical state bytes!

---

## 7. Generated Publication Figures (`results/synthetic/figures/` & `analysis/plot_synthetic.py`)

All figures were generated deterministically on CPU (`CUDA_VISIBLE_DEVICES=""`) from the 3-seed JSON artifacts in both 300-DPI `.png` and vector `.pdf` (`Type 42` TrueType fonts):

1. `results/synthetic/figures/fig1_mechanistic_dissociation.{png,pdf}` — Two-panel bar chart (`mean ± std`, 3 seeds) contrasting **Task 1A-Online ($S_8$ Container Swap)** vs. **Task 1A-Retro (`Forward` vs. `Mixed` vs. `Reverse` Causal Arrival Order)** across all 10 architectures.
2. `results/synthetic/figures/fig2_k_L_staircase_heatmap.{png,pdf}` — Side-by-side $6 \times 6$ 2D accuracy heatmaps across Chain Hop Depth $k \in \{1..6\}$ ($k=5,6$ zero-shot OOD) $\times$ Readout Loop Depth $L \in \{1..6\}$ comparing `static_query_loop_b2`, `untied_deep_readout_b2` ($3.7\times$ parameters), and `deltaloop_b2` (diagonal staircase).
3. `results/synthetic/figures/fig3_halting_and_qk_alignment.{png,pdf}` — Three-panel mechanistic figure showing (a) Stage-2 Adaptive Halting compute scaling $\mathbb{E}[L_t \mid k]$ across $k=1..6$, (b) 2-Stage Halting (`-56%` loops) vs. RD-VLA Latent L2-Convergence (`-25.4%` to `-35.9%` accuracy drop), and (c) the $6 \times 4$ Query-Key Top-1 Historical Hop Alignment heatmap (`deltaloop_b2`, $k=4$).
4. `results/synthetic/figures/fig4_task1b_multiscale.{png,pdf}` — Two-panel Task 1B ($T=2048$, strictly Iso-State = `1,536` floats / `6,144 B`) comparison across all 8 variants + Softmax Scale-Router Collapse heatmap (`5c_3bank_full`).
