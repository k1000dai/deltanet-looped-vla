# Cycle 3B — Pretrained `SmolVLA-DeltaLoop` (`U6`) Multi-Suite RoboMME Evaluation Summary

- **Protocol:** `16` RoboMME Tasks across 4 Memory Suites, `test` split (`25` episodes/task = **`400` matched episodes/run** on `seed=42`, `exec_horizon=16`, `max_steps=1300`, 10-step `t=1->0` Euler ODE).
- **Evaluated Runs Completed:** `20` (`11` primary `seed=42` + `6` multi-seed `{43, 44}` + `3` cross-session interference)

## 1. Overall & Per-Suite Success Rates (`k = 0`, `seed = 42`, `N = 400`)

| Evaluation Run | Active Params | State Size | Train FM Loss | Mean $L_t$ (`p95`) | Latency (`ms/ch`) | Overall SR (`95% CI`) | Permanence | Counting | Reference | Imitation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SmolVLA-M0: Pretrained Base (L=1)** (`smolvla_m0_base`) | `407.1M` | `0 B` | `0.0988` | `1.00` (`1.0`) | `140.9` | **1.50%** `[0.50, 2.75]` | 4.0% | 0.0% | 0.0% | 2.0% |
| **SmolVLA-M1: RD-VLA Memoryless Loop (L=4)** (`smolvla_m1_rdvla_loop4`) | `436.8M` | `0 B` | `0.0392` | `4.00` (`4.0`) | `162.8` | **2.50%** `[1.00, 4.01]` | 5.0% | 0.0% | 3.0% | 2.0% |
| **SmolVLA-M2: FrameSamp-32 + Modul/XAttn** (`smolvla_m2_framesamp`) | `436.9M` | `24.0 KB` | `0.0533` | `1.00` (`1.0`) | `245.8` | **6.50%** `[4.25, 9.00]` | 6.0% | 0.0% | 9.0% | 11.0% |
| **SmolVLA-M5: 2L Motion-Gated GDN (L=1)** (`smolvla_m5_gdn2l_l1`) | `444.0M` | `520.0 KB` | `0.0326` | `1.00` (`1.0`) | `238.2` | **5.25%** `[3.25, 7.50]` | 3.0% | 1.0% | 9.0% | 8.0% |
| **SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Fixed L=1)** (`smolvla_m7_deltaloop_L1`) | `444.2M` | `520.0 KB` | `0.0334` | `1.00` (`1.0`) | `238.7` | **2.75%** `[1.25, 4.50]` | 4.0% | 0.0% | 2.0% | 5.0% |
| **SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Fixed L=2)** (`smolvla_m7_deltaloop_L2`) | `444.2M` | `520.0 KB` | `0.0334` | `2.00` (`2.0`) | `239.7` | **4.00%** `[2.25, 6.00]` | 6.0% | 1.0% | 3.0% | 6.0% |
| **SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Adaptive CRN Gate)** (`smolvla_m7_deltaloop_adaptive`) | `444.2M` | `520.0 KB` | `0.0334` | `1.11` (`2.0`) | `240.3` | **4.25%** `[2.50, 6.25]` | 6.0% | 0.0% | 2.0% | 9.0% |
| **SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Latent L2 Conv tau=0.06)** (`smolvla_m7_deltaloop_coda_conv`) | `444.2M` | `520.0 KB` | `0.0334` | `2.28` (`3.0`) | `239.7` | **4.50%** `[2.50, 6.50]` | 5.0% | 2.0% | 3.0% | 8.0% |
| **SmolVLA-M7-Q16: 2L Iso-State DeltaLoop-VLA (N_c=16) (Adaptive CRN Gate)** (`smolvla_m7_deltaloop_q16_adaptive`) | `444.2M` | `520.0 KB` | `0.0310` | `1.42` (`2.0`) | `237.3` | **4.00%** `[2.25, 6.00]` | 2.0% | 1.0% | 5.0% | 8.0% |
| **SmolVLA-M7-Anchor: 2L Iso-State DeltaLoop-VLA (4+4 Hop-0 Anchor) (Adaptive CRN Gate)** (`smolvla_m7_deltaloop_anchor_adaptive`) | `444.2M` | `520.0 KB` | `0.0334` | `1.04` (`1.0`) | `239.6` | **3.00%** `[1.50, 4.75]` | 4.0% | 1.0% | 3.0% | 4.0% |
| **SmolVLA-M7-Anchor: 2L Iso-State DeltaLoop-VLA (4+4 Hop-0 Anchor) (Fixed L=2)** (`smolvla_m7_deltaloop_anchor_L2`) | `444.2M` | `520.0 KB` | `0.0334` | `2.00` (`2.0`) | `239.1` | **3.75%** `[2.00, 5.75]` | 5.0% | 1.0% | 3.0% | 6.0% |

### 1b. Episode-Matched Multi-Seed Verification (`seeds = {42, 43, 44}`, Exact Matched `eps 0..14` = `240`/Seed)

| Contender Group | Seeds (`N/seed`, `Pooled N`) | Seed Breakdown (`s42 / s43 / s44` Counts & SR) | Mean $\pm$ SD SR | Permanence | Counting | Reference | Imitation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `smolvla_m0_base__fixed__1` | `[42, 43, 44]` (`240`, `720`) | `s42=3/240 (1.25%)` / `s43=6/240 (2.50%)` / `s44=7/240 (2.92%)` | **`2.22% ± 0.87%`** | `3.3±0.0%` | `0.0±0.0%` | `2.2±1.9%` | `3.3±1.7%` |
| `smolvla_m2_framesamp__fixed__1` | `[42, 43, 44]` (`240`, `720`) | `s42=14/240 (5.83%)` / `s43=16/240 (6.67%)` / `s44=25/240 (10.42%)` | **`7.64% ± 2.44%`** | `4.4±3.5%` | `0.6±1.0%` | `12.8±7.5%` | `12.8±4.2%` |
| `smolvla_m7_deltaloop__learned_gate__4` | `[42, 43, 44]` (`240`, `720`) | `s42=10/240 (4.17%)` / `s43=10/240 (4.17%)` / `s44=9/240 (3.75%)` | **`4.03% ± 0.24%`** | `4.4±1.0%` | `0.6±1.0%` | `4.4±1.9%` | `6.7±2.9%` |

## 2. Per-Task Success Rates Across `16` RoboMME Tasks

| Evaluation Run | Overall | `ButtonUnmask` | `ButtonUnmaskSwap` | `VideoUnmask` | `VideoUnmaskSwap` | `BinFill` | `PickXtimes` | `SwingXtimes` | `StopCube` | `PickHighlight` | `VideoRepick` | `VideoPlaceButton` | `VideoPlaceOrder` | `MoveCube` | `InsertPeg` | `PatternLock` | `RouteStick` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `smolvla_m0_base` | **1.5%** | 0.0% | 0.0% | 8.0% | 8.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 4.0% | 0.0% | 0.0% | 4.0% |
| `smolvla_m1_rdvla_loop4` | **2.5%** | 4.0% | 0.0% | 12.0% | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 12.0% | 0.0% | 4.0% | 0.0% | 0.0% | 4.0% |
| `smolvla_m2_framesamp` | **6.5%** | 4.0% | 4.0% | 8.0% | 8.0% | 0.0% | 0.0% | 0.0% | 0.0% | 8.0% | 0.0% | 20.0% | 8.0% | 0.0% | 0.0% | 4.0% | 40.0% |
| `smolvla_m5_gdn2l_l1` | **5.2%** | 4.0% | 0.0% | 8.0% | 0.0% | 0.0% | 0.0% | 0.0% | 4.0% | 16.0% | 0.0% | 12.0% | 8.0% | 8.0% | 0.0% | 12.0% | 12.0% |
| `smolvla_m7_deltaloop_L1` | **2.8%** | 4.0% | 0.0% | 8.0% | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% | 8.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 8.0% | 12.0% |
| `smolvla_m7_deltaloop_L2` | **4.0%** | 4.0% | 0.0% | 12.0% | 8.0% | 0.0% | 0.0% | 0.0% | 4.0% | 4.0% | 0.0% | 4.0% | 4.0% | 4.0% | 0.0% | 12.0% | 8.0% |
| `smolvla_m7_deltaloop_adaptive` | **4.2%** | 4.0% | 0.0% | 12.0% | 8.0% | 0.0% | 0.0% | 0.0% | 0.0% | 8.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 20.0% | 16.0% |
| `smolvla_m7_deltaloop_coda_conv` | **4.5%** | 4.0% | 0.0% | 12.0% | 4.0% | 0.0% | 0.0% | 0.0% | 8.0% | 8.0% | 0.0% | 0.0% | 4.0% | 4.0% | 0.0% | 20.0% | 8.0% |
| `smolvla_m7_deltaloop_q16_adaptive` | **4.0%** | 4.0% | 0.0% | 4.0% | 0.0% | 0.0% | 0.0% | 0.0% | 4.0% | 0.0% | 0.0% | 8.0% | 12.0% | 4.0% | 0.0% | 8.0% | 20.0% |
| `smolvla_m7_deltaloop_anchor_adaptive` | **3.0%** | 4.0% | 0.0% | 8.0% | 4.0% | 0.0% | 0.0% | 0.0% | 4.0% | 4.0% | 0.0% | 0.0% | 8.0% | 0.0% | 0.0% | 8.0% | 8.0% |
| `smolvla_m7_deltaloop_anchor_L2` | **3.8%** | 4.0% | 0.0% | 12.0% | 4.0% | 0.0% | 0.0% | 0.0% | 4.0% | 8.0% | 0.0% | 0.0% | 4.0% | 0.0% | 0.0% | 4.0% | 20.0% |
| `smolvla_m0_base_s43` | **2.5%** | 0.0% | 0.0% | 6.7% | 6.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 6.7% | 6.7% | 6.7% | 0.0% | 0.0% | 6.7% |
| `smolvla_m0_base_s44` | **2.9%** | 0.0% | 0.0% | 6.7% | 6.7% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 13.3% | 13.3% | 0.0% | 0.0% | 6.7% |
| `smolvla_m2_framesamp_s43` | **6.7%** | 6.7% | 0.0% | 6.7% | 0.0% | 0.0% | 0.0% | 0.0% | 6.7% | 6.7% | 0.0% | 0.0% | 13.3% | 13.3% | 0.0% | 6.7% | 46.7% |
| `smolvla_m2_framesamp_s44` | **10.4%** | 6.7% | 0.0% | 13.3% | 13.3% | 0.0% | 0.0% | 0.0% | 0.0% | 13.3% | 0.0% | 26.7% | 40.0% | 6.7% | 0.0% | 6.7% | 40.0% |
| `smolvla_m7_deltaloop_s43_adaptive` | **4.2%** | 0.0% | 0.0% | 13.3% | 6.7% | 0.0% | 0.0% | 0.0% | 6.7% | 6.7% | 0.0% | 6.7% | 13.3% | 0.0% | 0.0% | 6.7% | 6.7% |
| `smolvla_m7_deltaloop_s44_adaptive` | **3.8%** | 0.0% | 0.0% | 13.3% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 6.7% | 0.0% | 0.0% | 6.7% | 0.0% | 0.0% | 6.7% | 26.7% |

## 3. Live Closed-Loop Cross-Session Interference (`k = 3` Prior Distractor Episodes, `N = 240` Matched `eps 0..14`)

| Interference Run | $k$ | State Size | Mean $L_t$ | Matched `k=0` SR (`eps 0..14`) | `k=3` SR (`Count`, `95% CI`) | Permanence | Counting | Reference | Imitation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SmolVLA-M2: FrameSamp-32 + Modul/XAttn [Interf k=3]** (`smolvla_m2_framesamp_interfk3`) | `3` | `24.0 KB` | `1.00` | `14/240` (`5.83%`) | **`1.25%`** (`3/240`, `[0.00, 2.92]`) | 1.7% | 0.0% | 3.3% | 0.0% |
| **SmolVLA-M5: 2L Motion-Gated GDN (L=1) [Interf k=3]** (`smolvla_m5_gdn2l_l1_interfk3`) | `3` | `520.0 KB` | `1.00` | `12/240` (`5.00%`) | **`0.83%`** (`2/240`, `[0.00, 2.08]`) | 1.7% | 0.0% | 1.7% | 0.0% |
| **SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Adaptive CRN Gate) [Interf k=3]** (`smolvla_m7_deltaloop_adaptive_interfk3`) | `3` | `520.0 KB` | `1.09` | `10/240` (`4.17%`) | **`2.92%`** (`7/240`, `[0.83, 5.00]`) | 6.7% | 0.0% | 5.0% | 0.0% |

## 4. Paired Bootstrap Hypothesis Tests (`10,000` Resamples; Holm-Bonferroni Step-Down Across `m = 38` Tests)

| Hypothesis / Comparison (`A - B`) | $N$ | $\text{SR}_A$ | $\text{SR}_B$ | $\Delta \text{SR}$ (`95% Paired CI`) | Two-Sided $p$ | Holm $p_{\text{Holm}}$ | Permanence $\Delta$ | Counting $\Delta$ | Reference $\Delta$ | Imitation $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H0a_M1_RDVLA_Loop4_vs_M0_Base**: `smolvla_m1_rdvla_loop4` vs `smolvla_m0_base` | `400` | `2.50%` | `1.50%` | **`+1.00%`** `[-1.00%, +3.00%]` | `0.3806` | `1.0000` | `+1.0%` | `+0.0%` | `+3.0%` | `+0.0%` |
| **H0b_M2_FrameSamp_vs_M0_Base**: `smolvla_m2_framesamp` vs `smolvla_m0_base` | `400` | `6.50%` | `1.50%` | **`+5.00%`** `[+2.50%, +7.50%]` | `0.0002` | **`0.0074`** | `+2.0%` | `+0.0%` | `+9.0%` | `+9.0%` |
| **H1a_M7_Adapt_vs_M0_Base**: `smolvla_m7_deltaloop_adaptive` vs `smolvla_m0_base` | `400` | `4.25%` | `1.50%` | **`+2.75%`** `[+0.50%, +5.00%]` | `0.0136` | `0.4352` | `+2.0%` | `+0.0%` | `+2.0%` | `+7.0%` |
| **H1d_M7_CodaConv_vs_M0_Base**: `smolvla_m7_deltaloop_coda_conv` vs `smolvla_m0_base` | `400` | `4.50%` | `1.50%` | **`+3.00%`** `[+0.75%, +5.50%]` | `0.0146` | `0.4526` | `+1.0%` | `+2.0%` | `+3.0%` | `+6.0%` |
| **H1b_M7_Adapt_vs_M1_RDVLA_Loop4**: `smolvla_m7_deltaloop_adaptive` vs `smolvla_m1_rdvla_loop4` | `400` | `4.25%` | `2.50%` | **`+1.75%`** `[-0.50%, +4.25%]` | `0.1796` | `1.0000` | `+1.0%` | `+0.0%` | `-1.0%` | `+7.0%` |
| **H1c_M5_GDN2L_vs_M0_Base**: `smolvla_m5_gdn2l_l1` vs `smolvla_m0_base` | `400` | `5.25%` | `1.50%` | **`+3.75%`** `[+1.50%, +6.25%]` | `0.0012` | **`0.0420`** | `-1.0%` | `+1.0%` | `+9.0%` | `+6.0%` |
| **H3a_M7_Adapt_vs_M2_FrameSamp**: `smolvla_m7_deltaloop_adaptive` vs `smolvla_m2_framesamp` | `400` | `4.25%` | `6.50%` | **`-2.25%`** `[-5.00%, +0.50%]` | `0.1344` | `1.0000` | `+0.0%` | `+0.0%` | `-7.0%` | `-2.0%` |
| **H3b_M5_GDN2L_vs_M2_FrameSamp**: `smolvla_m5_gdn2l_l1` vs `smolvla_m2_framesamp` | `400` | `5.25%` | `6.50%` | **`-1.25%`** `[-4.25%, +1.75%]` | `0.4732` | `1.0000` | `-3.0%` | `+1.0%` | `+0.0%` | `-3.0%` |
| **H4b_M7_L2_vs_M7_L1_IntraCkpt**: `smolvla_m7_deltaloop_L2` vs `smolvla_m7_deltaloop_L1` | `400` | `4.00%` | `2.75%` | **`+1.25%`** `[-0.25%, +3.00%]` | `0.1652` | `1.0000` | `+2.0%` | `+1.0%` | `+1.0%` | `+1.0%` |
| **H4d_M7_Adapt_vs_M7_L1_IntraCkpt**: `smolvla_m7_deltaloop_adaptive` vs `smolvla_m7_deltaloop_L1` | `400` | `4.25%` | `2.75%` | **`+1.50%`** `[+0.00%, +3.00%]` | `0.0696` | `1.0000` | `+2.0%` | `+0.0%` | `+0.0%` | `+4.0%` |
| **H4e_M7_CodaConv_vs_M7_L1_IntraCkpt**: `smolvla_m7_deltaloop_coda_conv` vs `smolvla_m7_deltaloop_L1` | `400` | `4.50%` | `2.75%` | **`+1.75%`** `[+0.00%, +3.50%]` | `0.0654` | `1.0000` | `+1.0%` | `+2.0%` | `+1.0%` | `+3.0%` |
| **H4c_M7_Adapt_vs_M5_GDN2L_L1**: `smolvla_m7_deltaloop_adaptive` vs `smolvla_m5_gdn2l_l1` | `400` | `4.25%` | `5.25%` | **`-1.00%`** `[-3.50%, +1.50%]` | `0.4722` | `1.0000` | `+3.0%` | `-1.0%` | `-7.0%` | `+1.0%` |
| **H5b_M7_Adapt_vs_M7_CodaConv**: `smolvla_m7_deltaloop_adaptive` vs `smolvla_m7_deltaloop_coda_conv` | `400` | `4.25%` | `4.50%` | **`-0.25%`** `[-2.25%, +1.75%]` | `0.8990` | `1.0000` | `+1.0%` | `-2.0%` | `-1.0%` | `+1.0%` |
| **H8a_M7_Q16_vs_M0_Base**: `smolvla_m7_deltaloop_q16_adaptive` vs `smolvla_m0_base` | `400` | `4.00%` | `1.50%` | **`+2.50%`** `[+0.50%, +4.75%]` | `0.0246` | `0.7134` | `-2.0%` | `+1.0%` | `+5.0%` | `+6.0%` |
| **H8b_M7_Q16_vs_M7_Adapt**: `smolvla_m7_deltaloop_q16_adaptive` vs `smolvla_m7_deltaloop_adaptive` | `400` | `4.00%` | `4.25%` | **`-0.25%`** `[-2.50%, +2.00%]` | `0.9266` | `1.0000` | `-4.0%` | `+1.0%` | `+3.0%` | `-1.0%` |
| **H8c_M7_Q16_vs_M2_FrameSamp**: `smolvla_m7_deltaloop_q16_adaptive` vs `smolvla_m2_framesamp` | `400` | `4.00%` | `6.50%` | **`-2.50%`** `[-5.25%, +0.25%]` | `0.0958` | `1.0000` | `-4.0%` | `+1.0%` | `-4.0%` | `-3.0%` |
| **H8d_M7_Q16_vs_M5_GDN2L_L1**: `smolvla_m7_deltaloop_q16_adaptive` vs `smolvla_m5_gdn2l_l1` | `400` | `4.00%` | `5.25%` | **`-1.25%`** `[-3.75%, +1.25%]` | `0.3656` | `1.0000` | `-1.0%` | `+0.0%` | `-4.0%` | `+0.0%` |
| **H9a_M7_Anchor_vs_M0_Base**: `smolvla_m7_deltaloop_anchor_adaptive` vs `smolvla_m0_base` | `400` | `3.00%` | `1.50%` | **`+1.50%`** `[-0.50%, +3.50%]` | `0.1626` | `1.0000` | `+0.0%` | `+1.0%` | `+3.0%` | `+2.0%` |
| **H9b_M7_Anchor_vs_M7_Adapt**: `smolvla_m7_deltaloop_anchor_adaptive` vs `smolvla_m7_deltaloop_adaptive` | `400` | `3.00%` | `4.25%` | **`-1.25%`** `[-3.25%, +0.75%]` | `0.2782` | `1.0000` | `-2.0%` | `+1.0%` | `+1.0%` | `-5.0%` |
| **H9c_M7_Anchor_vs_M7_Q16**: `smolvla_m7_deltaloop_anchor_adaptive` vs `smolvla_m7_deltaloop_q16_adaptive` | `400` | `3.00%` | `4.00%` | **`-1.00%`** `[-3.25%, +1.25%]` | `0.4290` | `1.0000` | `+2.0%` | `+0.0%` | `-2.0%` | `-4.0%` |
| **H9d_M7_Anchor_vs_M2_FrameSamp**: `smolvla_m7_deltaloop_anchor_adaptive` vs `smolvla_m2_framesamp` | `400` | `3.00%` | `6.50%` | **`-3.50%`** `[-6.25%, -0.75%]` | `0.0148` | `0.4526` | `-2.0%` | `+1.0%` | `-6.0%` | `-7.0%` |
| **H9e_M7_Anchor_vs_M5_GDN2L_L1**: `smolvla_m7_deltaloop_anchor_adaptive` vs `smolvla_m5_gdn2l_l1` | `400` | `3.00%` | `5.25%` | **`-2.25%`** `[-4.25%, -0.25%]` | `0.0364` | `1.0000` | `+1.0%` | `+0.0%` | `-6.0%` | `-4.0%` |
| **H9f_M7_Anchor_L2_vs_Anchor_Adapt**: `smolvla_m7_deltaloop_anchor_L2` vs `smolvla_m7_deltaloop_anchor_adaptive` | `400` | `3.75%` | `3.00%` | **`+0.75%`** `[-1.00%, +2.50%]` | `0.4832` | `1.0000` | `+1.0%` | `+0.0%` | `+0.0%` | `+2.0%` |
| **H9g_M7_Anchor_L2_vs_M0_Base**: `smolvla_m7_deltaloop_anchor_L2` vs `smolvla_m0_base` | `400` | `3.75%` | `1.50%` | **`+2.25%`** `[+0.00%, +4.50%]` | `0.0612` | `1.0000` | `+1.0%` | `+1.0%` | `+3.0%` | `+4.0%` |
| **H9h_M7_Anchor_L2_vs_M7_L2**: `smolvla_m7_deltaloop_anchor_L2` vs `smolvla_m7_deltaloop_L2` | `400` | `3.75%` | `4.00%` | **`-0.25%`** `[-2.25%, +1.75%]` | `0.9092` | `1.0000` | `-1.0%` | `+0.0%` | `+0.0%` | `+0.0%` |
| **H9i_M7_Anchor_L2_vs_M7_Adapt**: `smolvla_m7_deltaloop_anchor_L2` vs `smolvla_m7_deltaloop_adaptive` | `400` | `3.75%` | `4.25%` | **`-0.50%`** `[-2.75%, +1.75%]` | `0.7408` | `1.0000` | `-1.0%` | `+1.0%` | `+1.0%` | `-3.0%` |
| **H7a_Interf_k3_M7_vs_M2**: `smolvla_m7_deltaloop_adaptive_interfk3` vs `smolvla_m2_framesamp_interfk3` | `240` | `2.92%` | `1.25%` | **`+1.67%`** `[-0.42%, +4.17%]` | `0.1950` | `1.0000` | `+5.0%` | `+0.0%` | `+1.7%` | `+0.0%` |
| **H7b_Interf_Drop_M2_k3_vs_k0**: `smolvla_m2_framesamp_interfk3` vs `smolvla_m2_framesamp` | `240` | `1.25%` | `5.83%` | **`-4.58%`** `[-7.92%, -1.67%]` | `0.0032` | `0.1056` | `+0.0%` | `+0.0%` | `-10.0%` | `-8.3%` |
| **H7c_Interf_Drop_M7_k3_vs_k0**: `smolvla_m7_deltaloop_adaptive_interfk3` vs `smolvla_m7_deltaloop_adaptive` | `240` | `2.92%` | `4.17%` | **`-1.25%`** `[-3.75%, +1.25%]` | `0.4544` | `1.0000` | `+1.7%` | `+0.0%` | `+1.7%` | `-8.3%` |
| **H7d_M5_InterfDrop_k3_vs_k0**: `smolvla_m5_gdn2l_l1_interfk3` vs `smolvla_m5_gdn2l_l1` | `240` | `0.83%` | `5.00%` | **`-4.17%`** `[-6.67%, -1.67%]` | `0.0020` *(McN; boot `<= 0.0002`)* | `0.0680` *(McN NS; boot `<= 0.0076`)* | `+0.0%` | `+0.0%` | `-8.3%` | `-8.3%` |
| **H7e_Interf_k3_M5_vs_M2**: `smolvla_m5_gdn2l_l1_interfk3` vs `smolvla_m2_framesamp_interfk3` | `240` | `0.83%` | `1.25%` | **`-0.42%`** `[-2.08%, +1.25%]` | `0.8028` | `1.0000` | `+0.0%` | `+0.0%` | `-1.7%` | `+0.0%` |
| **H7f_Interf_k3_M7_vs_M5**: `smolvla_m7_deltaloop_adaptive_interfk3` vs `smolvla_m5_gdn2l_l1_interfk3` | `240` | `2.92%` | `0.83%` | **`+2.08%`** `[+0.00%, +4.58%]` | `0.0670` | `1.0000` | `+5.0%` | `+0.0%` | `+3.3%` | `+0.0%` |
| **H7g_DiD_InterfPenalty_M5_vs_M2**: `smolvla_m5_gdn2l_l1_interfk3_minus_k0` vs `smolvla_m2_framesamp_interfk3_minus_k0` | `240` | `-4.17%` | `-4.58%` | **`+0.42%`** `[-3.33%, +4.58%]` | `0.9140` | `1.0000` | `+0.0%` | `+0.0%` | `+1.7%` | `+0.0%` |
| **H7h_DiD_InterfPenalty_M7_vs_M2**: `smolvla_m7_deltaloop_adaptive_interfk3_minus_k0` vs `smolvla_m2_framesamp_interfk3_minus_k0` | `240` | `-1.25%` | `-4.58%` | **`+3.33%`** `[-0.42%, +7.08%]` | `0.0976` | `1.0000` | `+1.7%` | `+0.0%` | `+11.7%` | `+0.0%` |
| **H7i_DiD_InterfPenalty_M7_vs_M5**: `smolvla_m7_deltaloop_adaptive_interfk3_minus_k0` vs `smolvla_m5_gdn2l_l1_interfk3_minus_k0` | `240` | `-1.25%` | `-4.17%` | **`+2.92%`** `[-0.42%, +6.67%]` | `0.1238` | `1.0000` | `+1.7%` | `+0.0%` | `+10.0%` | `+0.0%` |
| **H_MS1_Pooled_M7_vs_M0**: `smolvla_m7_deltaloop_3seed` vs `smolvla_m0_base_3seed` | `720` | `4.03%` | `2.22%` | **`+1.81%`** `[+0.14%, +3.61%]` | `0.0414` | `1.0000` | `+1.1%` | `+0.6%` | `+2.2%` | `+3.3%` |
| **H_MS2_Pooled_M2_vs_M0**: `smolvla_m2_framesamp_3seed` vs `smolvla_m0_base_3seed` | `720` | `7.64%` | `2.22%` | **`+5.42%`** `[+3.47%, +7.50%]` | `<= 0.0001` | **`<= 0.0038`** | `+1.1%` | `+0.6%` | `+10.6%` | `+9.4%` |
| **H_MS3_Pooled_M7_vs_M2**: `smolvla_m7_deltaloop_3seed` vs `smolvla_m2_framesamp_3seed` | `720` | `4.03%` | `7.64%` | **`-3.61%`** `[-5.97%, -1.39%]` | `0.0008` | **`0.0288`** | `+0.0%` | `+0.0%` | `-8.3%` | `-6.1%` |

## 5. Adaptive Loop-Depth ($L_t$) Trajectory Diagnostics

### `smolvla_m7_deltaloop_adaptive` (SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Adaptive CRN Gate))
- **Overall Chunk Loop Allocation (`25522` chunks):** Mean $L_t = 1.105 \pm 0.307$ | Histogram: `L=1: 89.5%`, `L=2: 10.5%`, `L=3: 0.0%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.119`, Mid `[33..67%]` = `1.116`, Late `[67..100%]` = `1.081` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.116` vs `Subsequent` = `1.149`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmask` | Permanence | `4.0%` | `1.04` | `1.00` | `1.05` | `1.03` | `1.03` | `96.3%` | `3.7%` | `0.0%` | `0.0%` |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.05` | `1.00` | `1.07` | `1.08` | `1.02` | `94.6%` | `5.4%` | `0.0%` | `0.0%` |
| `VideoUnmask` | Permanence | `12.0%` | `1.10` | `1.00` | `1.17` | `1.11` | `1.03` | `89.5%` | `10.5%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `8.0%` | `1.11` | `1.00` | `1.15` | `1.14` | `1.05` | `88.6%` | `11.4%` | `0.0%` | `0.0%` |
| `BinFill` | Counting | `0.0%` | `1.10` | `1.00` | `1.11` | `1.13` | `1.05` | `90.4%` | `9.6%` | `0.0%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.04` | `1.00` | `1.07` | `1.04` | `1.03` | `95.7%` | `4.3%` | `0.0%` | `0.0%` |
| `SwingXtimes` | Counting | `0.0%` | `1.08` | `1.00` | `1.10` | `1.11` | `1.03` | `92.0%` | `8.0%` | `0.0%` | `0.0%` |
| `StopCube` | Counting | `0.0%` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `100.0%` | `0.0%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `8.0%` | `1.03` | `1.00` | `1.03` | `1.04` | `1.02` | `96.8%` | `3.2%` | `0.0%` | `0.0%` |
| `VideoRepick` | Reference | `0.0%` | `1.05` | `1.00` | `1.05` | `1.07` | `1.03` | `95.0%` | `5.0%` | `0.0%` | `0.0%` |
| `VideoPlaceButton` | Reference | `0.0%` | `1.02` | `1.00` | `1.03` | `1.03` | `1.00` | `97.7%` | `2.3%` | `0.0%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `0.0%` | `1.03` | `1.00` | `1.04` | `1.02` | `1.02` | `97.1%` | `2.9%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `0.0%` | `1.12` | `1.08` | `1.18` | `1.11` | `1.06` | `88.4%` | `11.6%` | `0.0%` | `0.0%` |
| `InsertPeg` | Imitation | `0.0%` | `1.56` | `1.04` | `1.43` | `1.60` | `1.63` | `44.4%` | `55.6%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `20.0%` | `1.07` | `1.28` | `1.09` | `1.08` | `1.05` | `92.5%` | `7.5%` | `0.0%` | `0.0%` |
| `RouteStick` | Imitation | `16.0%` | `1.17` | `1.64` | `1.26` | `1.13` | `1.14` | `82.7%` | `17.3%` | `0.0%` | `0.0%` |

### `smolvla_m7_deltaloop_coda_conv` (SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Latent L2 Conv tau=0.06))
- **Overall Chunk Loop Allocation (`25296` chunks):** Mean $L_t = 2.276 \pm 0.447$ | Histogram: `L=1: 0.0%`, `L=2: 72.5%`, `L=3: 27.6%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `2.230`, Mid `[33..67%]` = `2.295`, Late `[67..100%]` = `2.301` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `2.653` vs `Subsequent` = `2.266`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmask` | Permanence | `4.0%` | `2.22` | `3.00` | `2.21` | `2.21` | `2.24` | `0.0%` | `78.0%` | `22.0%` | `0.0%` |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `2.23` | `3.00` | `2.17` | `2.28` | `2.23` | `0.0%` | `77.0%` | `23.0%` | `0.0%` |
| `VideoUnmask` | Permanence | `12.0%` | `2.24` | `2.00` | `2.21` | `2.24` | `2.27` | `0.0%` | `75.8%` | `24.2%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `4.0%` | `2.24` | `2.24` | `2.22` | `2.26` | `2.24` | `0.0%` | `76.2%` | `23.8%` | `0.0%` |
| `BinFill` | Counting | `0.0%` | `2.31` | `3.00` | `2.29` | `2.32` | `2.33` | `0.0%` | `68.8%` | `31.2%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `2.35` | `3.00` | `2.28` | `2.40` | `2.36` | `0.0%` | `65.5%` | `34.5%` | `0.0%` |
| `SwingXtimes` | Counting | `0.0%` | `2.46` | `3.00` | `2.33` | `2.46` | `2.58` | `0.0%` | `54.4%` | `45.6%` | `0.0%` |
| `StopCube` | Counting | `8.0%` | `2.09` | `3.00` | `2.19` | `2.03` | `2.05` | `0.0%` | `91.3%` | `8.7%` | `0.0%` |
| `PickHighlight` | Reference | `8.0%` | `2.18` | `3.00` | `2.16` | `2.21` | `2.18` | `0.0%` | `81.7%` | `18.3%` | `0.0%` |
| `VideoRepick` | Reference | `0.0%` | `2.28` | `3.00` | `2.25` | `2.25` | `2.34` | `0.0%` | `71.9%` | `28.1%` | `0.0%` |
| `VideoPlaceButton` | Reference | `0.0%` | `2.11` | `2.76` | `2.09` | `2.09` | `2.15` | `0.0%` | `88.8%` | `11.2%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `4.0%` | `2.12` | `2.84` | `2.09` | `2.13` | `2.16` | `0.0%` | `87.5%` | `12.5%` | `0.0%` |
| `MoveCube` | Imitation | `4.0%` | `2.35` | `3.00` | `2.35` | `2.37` | `2.32` | `0.0%` | `65.1%` | `34.9%` | `0.0%` |
| `InsertPeg` | Imitation | `0.0%` | `2.58` | `3.00` | `2.38` | `2.72` | `2.64` | `0.0%` | `42.1%` | `57.9%` | `0.0%` |
| `PatternLock` | Imitation | `20.0%` | `2.23` | `2.64` | `2.32` | `2.21` | `2.17` | `0.0%` | `77.0%` | `23.0%` | `0.0%` |
| `RouteStick` | Imitation | `8.0%` | `2.15` | `2.40` | `2.09` | `2.14` | `2.20` | `0.0%` | `85.5%` | `14.5%` | `0.0%` |

### `smolvla_m7_deltaloop_q16_adaptive` (SmolVLA-M7-Q16: 2L Iso-State DeltaLoop-VLA (N_c=16) (Adaptive CRN Gate))
- **Overall Chunk Loop Allocation (`24383` chunks):** Mean $L_t = 1.423 \pm 0.495$ | Histogram: `L=1: 57.7%`, `L=2: 42.2%`, `L=3: 0.1%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.430`, Mid `[33..67%]` = `1.429`, Late `[67..100%]` = `1.411` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.951` vs `Subsequent` = `1.473`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmask` | Permanence | `4.0%` | `1.35` | `2.00` | `1.32` | `1.38` | `1.36` | `64.6%` | `35.4%` | `0.0%` | `0.0%` |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.38` | `2.00` | `1.31` | `1.44` | `1.40` | `61.6%` | `38.4%` | `0.0%` | `0.0%` |
| `VideoUnmask` | Permanence | `4.0%` | `1.45` | `1.88` | `1.48` | `1.43` | `1.43` | `55.2%` | `44.8%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `0.0%` | `1.42` | `2.00` | `1.46` | `1.42` | `1.39` | `58.1%` | `41.9%` | `0.1%` | `0.0%` |
| `BinFill` | Counting | `0.0%` | `1.42` | `2.00` | `1.44` | `1.39` | `1.42` | `58.5%` | `41.4%` | `0.1%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.41` | `2.00` | `1.40` | `1.40` | `1.43` | `59.2%` | `40.8%` | `0.0%` | `0.0%` |
| `SwingXtimes` | Counting | `0.0%` | `1.34` | `2.00` | `1.43` | `1.29` | `1.31` | `65.8%` | `34.1%` | `0.1%` | `0.0%` |
| `StopCube` | Counting | `4.0%` | `1.00` | `1.00` | `1.00` | `1.01` | `1.01` | `99.6%` | `0.4%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `0.0%` | `1.34` | `2.00` | `1.32` | `1.36` | `1.34` | `66.0%` | `33.9%` | `0.1%` | `0.0%` |
| `VideoRepick` | Reference | `0.0%` | `1.30` | `1.68` | `1.32` | `1.31` | `1.28` | `69.8%` | `30.1%` | `0.1%` | `0.0%` |
| `VideoPlaceButton` | Reference | `8.0%` | `1.39` | `2.00` | `1.38` | `1.41` | `1.39` | `60.9%` | `39.0%` | `0.1%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `12.0%` | `1.38` | `2.00` | `1.37` | `1.36` | `1.41` | `62.1%` | `37.9%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `4.0%` | `1.48` | `2.00` | `1.58` | `1.47` | `1.40` | `51.8%` | `48.1%` | `0.1%` | `0.0%` |
| `InsertPeg` | Imitation | `0.0%` | `1.75` | `2.00` | `1.70` | `1.83` | `1.74` | `24.6%` | `75.3%` | `0.1%` | `0.0%` |
| `PatternLock` | Imitation | `8.0%` | `1.69` | `2.00` | `1.71` | `1.70` | `1.67` | `30.6%` | `69.4%` | `0.0%` | `0.0%` |
| `RouteStick` | Imitation | `20.0%` | `1.66` | `2.00` | `1.76` | `1.62` | `1.62` | `33.6%` | `66.4%` | `0.0%` | `0.0%` |

### `smolvla_m7_deltaloop_anchor_adaptive` (SmolVLA-M7-Anchor: 2L Iso-State DeltaLoop-VLA (4+4 Hop-0 Anchor) (Adaptive CRN Gate))
- **Overall Chunk Loop Allocation (`26146` chunks):** Mean $L_t = 1.039 \pm 0.193$ | Histogram: `L=1: 96.1%`, `L=2: 3.9%`, `L=3: 0.0%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.042`, Mid `[33..67%]` = `1.042`, Late `[67..100%]` = `1.032` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.009` vs `Subsequent` = `1.052`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmask` | Permanence | `4.0%` | `1.02` | `1.00` | `1.04` | `1.03` | `1.00` | `97.6%` | `2.4%` | `0.0%` | `0.0%` |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.02` | `1.00` | `1.03` | `1.04` | `1.01` | `97.5%` | `2.5%` | `0.0%` | `0.0%` |
| `VideoUnmask` | Permanence | `8.0%` | `1.04` | `1.00` | `1.06` | `1.04` | `1.02` | `95.8%` | `4.2%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `4.0%` | `1.04` | `1.00` | `1.07` | `1.04` | `1.02` | `95.6%` | `4.4%` | `0.0%` | `0.0%` |
| `BinFill` | Counting | `0.0%` | `1.01` | `1.00` | `1.02` | `1.01` | `1.01` | `98.6%` | `1.4%` | `0.0%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.04` | `1.00` | `1.06` | `1.03` | `1.03` | `96.2%` | `3.8%` | `0.0%` | `0.0%` |
| `SwingXtimes` | Counting | `0.0%` | `1.02` | `1.00` | `1.03` | `1.03` | `1.01` | `97.5%` | `2.5%` | `0.0%` | `0.0%` |
| `StopCube` | Counting | `4.0%` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `100.0%` | `0.0%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `4.0%` | `1.02` | `1.00` | `1.03` | `1.03` | `1.01` | `97.5%` | `2.5%` | `0.0%` | `0.0%` |
| `VideoRepick` | Reference | `0.0%` | `1.02` | `1.00` | `1.04` | `1.02` | `1.01` | `97.8%` | `2.2%` | `0.0%` | `0.0%` |
| `VideoPlaceButton` | Reference | `0.0%` | `1.04` | `1.00` | `1.04` | `1.03` | `1.04` | `96.4%` | `3.6%` | `0.0%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `8.0%` | `1.03` | `1.00` | `1.04` | `1.03` | `1.03` | `96.6%` | `3.4%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `0.0%` | `1.11` | `1.00` | `1.09` | `1.10` | `1.14` | `88.9%` | `11.1%` | `0.0%` | `0.0%` |
| `InsertPeg` | Imitation | `0.0%` | `1.10` | `1.08` | `1.04` | `1.15` | `1.11` | `90.1%` | `9.9%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `8.0%` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `100.0%` | `0.0%` | `0.0%` | `0.0%` |
| `RouteStick` | Imitation | `8.0%` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `100.0%` | `0.0%` | `0.0%` | `0.0%` |

### `smolvla_m7_deltaloop_s43_adaptive` (SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Adaptive CRN Gate) [seed=43])
- **Overall Chunk Loop Allocation (`14406` chunks):** Mean $L_t = 1.293 \pm 0.455$ | Histogram: `L=1: 70.7%`, `L=2: 29.3%`, `L=3: 0.0%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.249`, Mid `[33..67%]` = `1.331`, Late `[67..100%]` = `1.298` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.459` vs `Subsequent` = `1.343`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmask` | Permanence | `0.0%` | `1.18` | `1.40` | `1.14` | `1.19` | `1.20` | `82.2%` | `17.8%` | `0.0%` | `0.0%` |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.18` | `1.00` | `1.22` | `1.18` | `1.15` | `81.6%` | `18.4%` | `0.0%` | `0.0%` |
| `VideoUnmask` | Permanence | `13.3%` | `1.22` | `1.00` | `1.16` | `1.29` | `1.20` | `78.3%` | `21.7%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `6.7%` | `1.23` | `1.00` | `1.19` | `1.30` | `1.21` | `76.9%` | `23.1%` | `0.0%` | `0.0%` |
| `BinFill` | Counting | `0.0%` | `1.30` | `1.00` | `1.15` | `1.38` | `1.36` | `70.4%` | `29.6%` | `0.0%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.17` | `1.00` | `1.08` | `1.19` | `1.23` | `83.1%` | `16.9%` | `0.0%` | `0.0%` |
| `SwingXtimes` | Counting | `0.0%` | `1.26` | `1.00` | `1.18` | `1.26` | `1.33` | `74.3%` | `25.7%` | `0.0%` | `0.0%` |
| `StopCube` | Counting | `6.7%` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `100.0%` | `0.0%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `6.7%` | `1.37` | `1.00` | `1.17` | `1.41` | `1.51` | `63.4%` | `36.6%` | `0.0%` | `0.0%` |
| `VideoRepick` | Reference | `0.0%` | `1.21` | `1.40` | `1.17` | `1.20` | `1.25` | `79.3%` | `20.6%` | `0.0%` | `0.0%` |
| `VideoPlaceButton` | Reference | `6.7%` | `1.15` | `1.00` | `1.12` | `1.14` | `1.17` | `85.4%` | `14.6%` | `0.0%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `13.3%` | `1.13` | `1.00` | `1.11` | `1.14` | `1.14` | `87.0%` | `13.0%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `0.0%` | `1.39` | `1.80` | `1.47` | `1.39` | `1.31` | `61.3%` | `38.6%` | `0.1%` | `0.0%` |
| `InsertPeg` | Imitation | `0.0%` | `1.59` | `1.93` | `1.58` | `1.80` | `1.41` | `40.8%` | `59.2%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `6.7%` | `1.99` | `2.00` | `1.99` | `1.99` | `1.98` | `1.5%` | `98.5%` | `0.0%` | `0.0%` |
| `RouteStick` | Imitation | `6.7%` | `1.97` | `2.00` | `1.97` | `1.99` | `1.96` | `3.0%` | `97.0%` | `0.0%` | `0.0%` |

### `smolvla_m7_deltaloop_s44_adaptive` (SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Adaptive CRN Gate) [seed=44])
- **Overall Chunk Loop Allocation (`15262` chunks):** Mean $L_t = 1.033 \pm 0.179$ | Histogram: `L=1: 96.7%`, `L=2: 3.3%`, `L=3: 0.0%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.044`, Mid `[33..67%]` = `1.035`, Late `[67..100%]` = `1.021` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.119` vs `Subsequent` = `1.040`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmask` | Permanence | `0.0%` | `1.01` | `1.00` | `1.01` | `1.02` | `1.00` | `98.9%` | `1.1%` | `0.0%` | `0.0%` |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.03` | `1.00` | `1.03` | `1.06` | `1.01` | `96.6%` | `3.4%` | `0.0%` | `0.0%` |
| `VideoUnmask` | Permanence | `13.3%` | `1.02` | `1.00` | `1.06` | `1.01` | `1.00` | `97.5%` | `2.5%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `0.0%` | `1.02` | `1.00` | `1.03` | `1.03` | `1.01` | `97.8%` | `2.2%` | `0.0%` | `0.0%` |
| `BinFill` | Counting | `0.0%` | `1.04` | `1.00` | `1.03` | `1.07` | `1.04` | `95.5%` | `4.5%` | `0.0%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.01` | `1.00` | `1.01` | `1.02` | `1.00` | `98.9%` | `1.1%` | `0.0%` | `0.0%` |
| `SwingXtimes` | Counting | `0.0%` | `1.02` | `1.00` | `1.03` | `1.03` | `1.01` | `97.6%` | `2.4%` | `0.0%` | `0.0%` |
| `StopCube` | Counting | `0.0%` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `100.0%` | `0.0%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `6.7%` | `1.03` | `1.00` | `1.01` | `1.03` | `1.03` | `97.5%` | `2.5%` | `0.0%` | `0.0%` |
| `VideoRepick` | Reference | `0.0%` | `1.02` | `1.00` | `1.02` | `1.02` | `1.03` | `97.6%` | `2.4%` | `0.0%` | `0.0%` |
| `VideoPlaceButton` | Reference | `0.0%` | `1.01` | `1.00` | `1.01` | `1.01` | `1.01` | `99.1%` | `0.9%` | `0.0%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `6.7%` | `1.01` | `1.00` | `1.01` | `1.01` | `1.01` | `98.8%` | `1.2%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `0.0%` | `1.07` | `1.47` | `1.14` | `1.04` | `1.04` | `92.5%` | `7.5%` | `0.0%` | `0.0%` |
| `InsertPeg` | Imitation | `0.0%` | `1.10` | `1.53` | `1.17` | `1.09` | `1.05` | `89.6%` | `10.4%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `6.7%` | `1.01` | `1.07` | `1.01` | `1.01` | `1.01` | `99.0%` | `1.0%` | `0.0%` | `0.0%` |
| `RouteStick` | Imitation | `26.7%` | `1.04` | `1.00` | `1.04` | `1.03` | `1.06` | `95.8%` | `4.2%` | `0.0%` | `0.0%` |

### `smolvla_m7_deltaloop_adaptive_interfk3` (SmolVLA-M7: 2L Iso-State DeltaLoop-VLA (Adaptive CRN Gate) [Interf k=3])
- **Overall Chunk Loop Allocation (`15170` chunks):** Mean $L_t = 1.091 \pm 0.287$ | Histogram: `L=1: 90.9%`, `L=2: 9.1%`, `L=3: 0.0%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.087`, Mid `[33..67%]` = `1.109`, Late `[67..100%]` = `1.077` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.074` vs `Subsequent` = `1.120`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmask` | Permanence | `0.0%` | `1.04` | `1.00` | `1.01` | `1.04` | `1.06` | `96.3%` | `3.7%` | `0.0%` | `0.0%` |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.03` | `1.00` | `1.01` | `1.03` | `1.04` | `97.2%` | `2.9%` | `0.0%` | `0.0%` |
| `VideoUnmask` | Permanence | `26.7%` | `1.08` | `1.00` | `1.07` | `1.10` | `1.08` | `91.7%` | `8.3%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `0.0%` | `1.10` | `1.00` | `1.08` | `1.11` | `1.09` | `90.4%` | `9.6%` | `0.0%` | `0.0%` |
| `BinFill` | Counting | `0.0%` | `1.09` | `1.00` | `1.13` | `1.11` | `1.04` | `90.8%` | `9.2%` | `0.0%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.06` | `1.00` | `1.11` | `1.05` | `1.03` | `93.8%` | `6.2%` | `0.0%` | `0.0%` |
| `SwingXtimes` | Counting | `0.0%` | `1.08` | `1.00` | `1.12` | `1.07` | `1.05` | `91.9%` | `8.1%` | `0.0%` | `0.0%` |
| `StopCube` | Counting | `0.0%` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `100.0%` | `0.0%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `20.0%` | `1.05` | `1.00` | `1.05` | `1.05` | `1.04` | `95.4%` | `4.6%` | `0.0%` | `0.0%` |
| `VideoRepick` | Reference | `0.0%` | `1.04` | `1.00` | `1.06` | `1.05` | `1.02` | `95.8%` | `4.2%` | `0.0%` | `0.0%` |
| `VideoPlaceButton` | Reference | `0.0%` | `1.01` | `1.00` | `1.01` | `1.02` | `1.01` | `98.8%` | `1.2%` | `0.0%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `0.0%` | `1.03` | `1.00` | `1.03` | `1.04` | `1.02` | `97.2%` | `2.8%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `0.0%` | `1.07` | `1.00` | `1.08` | `1.07` | `1.04` | `93.4%` | `6.6%` | `0.0%` | `0.0%` |
| `InsertPeg` | Imitation | `0.0%` | `1.51` | `1.07` | `1.36` | `1.68` | `1.49` | `49.1%` | `50.9%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `0.0%` | `1.05` | `1.47` | `1.09` | `1.05` | `1.00` | `95.2%` | `4.8%` | `0.0%` | `0.0%` |
| `RouteStick` | Imitation | `0.0%` | `1.06` | `1.13` | `1.11` | `1.05` | `1.03` | `93.8%` | `6.2%` | `0.0%` | `0.0%` |

## 6. Executive Scientific Synthesis

- **Cumulative Closed-Loop Official `test` Rollouts Evaluated:** **`6,560`** episodes across `20` runs on all `16` RoboMME tasks (`10,720` total closed-loop RoboMME rollouts across Cycle 2A `2,400` + Cycle 3A `1,760` + Cycle 3B/4 `6,560`).
- **1. Recurrent State Compression (`520 KB` 2L GDN) — Not Multi-Hop Looping — Drives Closed-Loop Gains (`H1c`, `H4c`):** Looping a memoryless pretrained VLA (`smolvla_m1_rdvla_loop4`, `L=4`, `10/400 = 2.50%`) yields no significant gain over `smolvla_m0_base` (`6/400 = 1.50%`; `+1.00%`, two-sided $p = 0.3806$, $p_{\text{Holm}} = 1.0000$ NS). Equipping `SmolVLA` with single-pass `520 KB` 2-layer GDN (`smolvla_m5_gdn2l_l1`, `L=1`, `21/400 = 5.25%`) achieves a **Holm-Bonferroni significant gain** over `M0` (`+3.75%`, two-sided $p = 0.0012$, **$p_{\text{Holm}} = 0.0420$**) and outperforms all multi-hop `SmolVLA-DeltaLoop` variants (`4.00%–4.50%`; `H4c` `M7 CRN` `4.25%` vs `M5` `5.25%`: `-1.00%`, two-sided $p = 0.4722$ NS; `H1a` `M7 CRN` vs `M0`: `+2.75%`, unadjusted two-sided $p = 0.0136$, $p_{\text{Holm}} = 0.4352$ NS after multiplicity correction), consistent with Scratch Cycle 2A (`11.25% vs 7.50%`) and Cycle 3A (`12.50% vs 8.12%/6.25%`).
- **2. Intra-Checkpoint Loop Sensitivity & Halting Pareto Efficiency (`H4b`, `H4d`, `H4e`, `H5b`):** Within the same random-`L`-trained `smolvla_m7_deltaloop.pt` weights (`N = 400`), truncating to `Fixed L=1` yields `11/400` (`2.75%`), compared to `16/400` (`4.00%`, `Fixed L=2`, `H4b` two-sided $p = 0.1652$ NS), `17/400` (`4.25%`, `Adaptive CRN` at $L_t = 1.11$, `H4d` two-sided $p = 0.0696$ NS), and `18/400` (`4.50%`, `Latent L2 Conv` at $L_t = 2.28$, `H4e` two-sided $p = 0.0654$ NS; all $p_{\text{Holm}} = 1.0000$). These non-significant intra-checkpoint trends reflect sensitivity of random-loop-trained weights to `L=1` truncation while remaining below single-pass `M5` (`5.25%`); within `M7`, Stage-II `CRN` (`1.11` hops) matches `Latent L2 Conv` (`2.28` hops; `H5b: -0.25%`, two-sided $p = 0.8990$) at `-51.3%` fewer recurrent hops.
- **3. Live Cross-Session Interference (`k = 3`, `N = 240` Matched `eps 0..14`, `H7a–H7i`):** Under `k = 3` un-reset prior distractor episodes, 32-slot `FrameSamp-32` (`M2`) drops from `14/240` (`5.83%`) to `3/240` (`1.25%`; `H7b: -4.58%`, unadjusted two-sided $p = 0.0032$, $p_{\text{Holm}} = 0.1056$). Decisively, single-pass `M5` (`smolvla_m5_gdn2l_l1_interfk3`) also drops from `12/240` (`5.00%`) to `2/240` (`0.83%`; `H7d: -4.17%`, exact two-sided McNemar $p = 0.0020$, $p_{\text{Holm}} = 0.0680$ borderline NS on `0` gained vs `10` lost [finite-sample bootstrap $p \le 0.0002$, $p_{\text{Holm}} \le 0.0074$, $1/(B+1)$ resolution $\le 0.0038$; zero-gain bootstrap underestimates variance]), indicating that static linear readout over ~`960` accumulated distractor frames degrades in closed loop despite high raw-state cosine similarity in offline probes. Meanwhile, `SmolVLA-DeltaLoop` (`M7`) moves from `10/240` (`4.17%`) to `7/240` (`2.92%`; `H7c: -1.25%`, two-sided $p = 0.4544$ NS). However, all `k = 3` head-to-head and Difference-in-Differences comparisons (`H7a` `M7 vs M2`: `7/240 vs 3/240`, two-sided $p = 0.1950$ NS; `H7f` `M7 vs M5`: `7/240 vs 2/240`, two-sided $p = 0.0670$ NS; `H7h` DiD `M7 vs M2` $p = 0.0976$ NS; `H7i` DiD `M7 vs M5` $p = 0.1238$ NS) are **not statistically significant** in this low-count floor regime.
- **4. Exploratory Readout Bandwidth (`U7`) & Anchor-Shortcut Observations (`U8`):** Expanding mutable Coda queries to `N_c = 16` (`U7`: `16/400 = 4.00%`, $L_t = 1.42$; `H8a` vs `M0` unadjusted two-sided $p = 0.0246$, $p_{\text{Holm}} = 0.7134$ NS) shifts successes toward `Reference` (`5.0%`) while lowering `Permanence` (`2.0%`). Partitioning `4+4` Hop-0 Anchor + Hop-$l$ Pointer registers (`U8`: `12/400 = 3.00%`) causes Stage-II CRN to halt at $L=1$ on `96.1%` of chunks ($L_t = 1.04$); evaluating `U8` at `Fixed L=2` yields `15/400` (`3.75%`, `H9f: +0.75%`, two-sided $p = 0.4832$ NS), an exploratory non-significant observation.
- **5. Multi-Seed Reproducibility (`seeds = {42, 43, 44}`, Pooled `N = 720`) & `<10.5%` Absolute SR Floor Caveat:** Across 3 training seeds (`N = 240` matched `eps 0..14`/seed), `SmolVLA-M7` achieves `4.03% ± 0.24%` (`10, 10, 9 / 240`) vs `SmolVLA-M0` `2.22% ± 0.87%` (`3, 6, 7 / 240`; `H_MS1: +1.81%`, unadjusted two-sided $p = 0.0414$, $p_{\text{Holm}} = 1.0000$ NS after multiplicity correction) and `FrameSamp-32 (M2)` `7.64% ± 2.44%` (`14, 16, 25 / 240` at $k=0$; `H_MS2: +5.42%`, **$p_{\text{Holm}} = 0.0038$**; `H_MS3 M7 vs M2: -3.61%`, two-sided $p = 0.0008$, **$p_{\text{Holm}} = 0.0288$**). All frozen-VLM `lerobot/smolvla_base` (`450M`, `8x8`-pooled cached tokens) runs remain in a `<10.5%` absolute SR floor regime well below end-to-end 3B `\pi_{0.5}` (`17.9%` base / `44.5%` `FrameSamp+Modul`).

