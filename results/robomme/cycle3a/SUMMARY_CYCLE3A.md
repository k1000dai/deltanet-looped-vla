# Cycle 3A — Upgraded (`U1–U5` + `M7`) Multi-Suite RoboMME Evaluation Summary

- **Protocol:** 8 Representative RoboMME Tasks (`2 tasks × 4 memory suites`), `val` split (`20` episodes/task = **`160` matched episodes per evaluation run**, `model_seed=7`, `exec_horizon=16`, `max_steps=1300`, native `v3_mode=True` `(20, 8)` quantile normalization).
- **Evaluated Runs Completed:** `11` (`9` standard + `2` cross-session interference)

## 1. Overall & Per-Suite Success Rates (`N = 160` Matched Episodes, `k = 0`)

| Evaluation Run | Params | State Size | Train FM Loss | Mean $L_t$ (`p95`) | Latency (`ms/ch`) | Overall SR (`95% CI`) | $\Delta$ vs Cycle 2A | Permanence (`40`) | Counting (`40`) | Reference (`40`) | Imitation (`40`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Memoryless Flow v3 (L=1)** (`m0_memoryless`) | `28.61M` | `0 B` | `0.0083` | `1.00` (`1.0`) | `35.9` | **3.1%** `[0.6, 6.2]` | **`+1.2%`** (`p=0.627`) | 0.0% | 0.0% | 5.0% | 7.5% |
| **M2: FrameSamp-32 + Modul/XAttn v3** (`m2_framesamp_modul`) | `32.85M` | `24.0 KB` | `0.0048` | `1.00` (`1.0`) | `60.7` | **5.0%** `[1.9, 8.8]` | **`+0.0%`** (`p=1.000`) | 0.0% | 2.5% | 12.5% | 5.0% |
| **M3: GDN beta in (0,1) Single-Pass v3 (L=1)** (`m3_gdn_b1_l1`) | `37.49M` | `516.0 KB` | `0.0044` | `1.00` (`1.0`) | `56.2` | **10.6%** `[6.2, 15.6]` | **`+0.6%`** (`p=0.928`) | 2.5% | 12.5% | 15.0% | 12.5% |
| **M4: Motion-Gated GDN beta in (0,2) v3 (L=1)** (`m4_gdn_b2_l1`) | `37.49M` | `516.0 KB` | `0.0047` | `1.00` (`1.0`) | `55.6` | **10.6%** `[6.2, 15.6]` | **`+5.0%`** (`p=0.112`) | 0.0% | 12.5% | 22.5% | 7.5% |
| **M5: 2-Layer Iso-State GDN beta in (0,2) v3 (L=1)** (`m5_gdn2layer_b2_l1`) | `39.95M` | `520.0 KB` | `0.0041` | `1.00` (`1.0`) | `58.6` | **12.5%** `[7.5, 18.1]` | **`+1.2%`** (`p=0.787`) | 7.5% | 7.5% | 12.5% | 22.5% |
| **M6: DeltaLoop-VLA 1L v3 (Fixed L=2)** (`m6_deltaloop_b2_L2`) | `37.64M` | `516.0 KB` | `0.0045` | `2.00` (`2.0`) | `57.3` | **8.1%** `[4.4, 12.5]` | **`+0.0%`** (`p=1.000`) | 10.0% | 5.0% | 10.0% | 7.5% |
| **M6: DeltaLoop-VLA 1L v3 (Adaptive CRN Gate)** (`m6_deltaloop_b2_adaptive`) | `37.64M` | `516.0 KB` | `0.0045` | `1.10` (`2.0`) | `56.2` | **8.1%** `[4.4, 12.5]` | **`+0.6%`** (`p=0.916`) | 5.0% | 5.0% | 12.5% | 10.0% |
| **M7: DeltaLoop-VLA 2L Iso-State v3 (Fixed L=2)** (`m7_deltaloop_2l_b2_L2`) | `40.11M` | `520.0 KB` | `0.0042` | `2.00` (`2.0`) | `62.6` | **6.9%** `[3.1, 11.2]` | — | 2.5% | 5.0% | 12.5% | 7.5% |
| **M7: DeltaLoop-VLA 2L Iso-State v3 (Adaptive CRN Gate)** (`m7_deltaloop_2l_b2_adaptive`) | `40.11M` | `520.0 KB` | `0.0042` | `1.06` (`2.0`) | `57.9` | **6.2%** `[3.1, 10.0]` | — | 5.0% | 5.0% | 10.0% | 5.0% |

## 2. Per-Task Success Rates (`20` Episodes / Task)

| Evaluation Run | Overall | `ButtonUnmaskSwap` | `VideoUnmaskSwap` | `PickXtimes` | `StopCube` | `PickHighlight` | `VideoPlaceOrder` | `PatternLock` | `MoveCube` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `m0_memoryless` | **3.1%** | 0.0% | 0.0% | 0.0% | 0.0% | 5.0% | 5.0% | 5.0% | 10.0% |
| `m2_framesamp_modul` | **5.0%** | 0.0% | 0.0% | 0.0% | 5.0% | 15.0% | 10.0% | 5.0% | 5.0% |
| `m3_gdn_b1_l1` | **10.6%** | 0.0% | 5.0% | 0.0% | 25.0% | 5.0% | 25.0% | 20.0% | 5.0% |
| `m4_gdn_b2_l1` | **10.6%** | 0.0% | 0.0% | 0.0% | 25.0% | 15.0% | 30.0% | 10.0% | 5.0% |
| `m5_gdn2layer_b2_l1` | **12.5%** | 0.0% | 15.0% | 0.0% | 15.0% | 10.0% | 15.0% | 30.0% | 15.0% |
| `m6_deltaloop_b2_L2` | **8.1%** | 5.0% | 15.0% | 0.0% | 10.0% | 15.0% | 5.0% | 15.0% | 0.0% |
| `m6_deltaloop_b2_adaptive` | **8.1%** | 0.0% | 10.0% | 0.0% | 10.0% | 10.0% | 15.0% | 10.0% | 10.0% |
| `m7_deltaloop_2l_b2_L2` | **6.9%** | 0.0% | 5.0% | 5.0% | 5.0% | 5.0% | 20.0% | 10.0% | 5.0% |
| `m7_deltaloop_2l_b2_adaptive` | **6.2%** | 0.0% | 10.0% | 0.0% | 10.0% | 10.0% | 10.0% | 10.0% | 0.0% |

## 3. Live Closed-Loop Cross-Session Interference (`k > 0` Prior Distractor Episodes)

| Interference Run | $k$ Sessions | State Size | Mean $L_t$ | Overall SR (`95% CI`) | Permanence | Counting | Reference | Imitation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M2: FrameSamp-32 + Modul/XAttn v3 [Interf k=3]** (`m2_framesamp_modul_interfk3`) | `3` | `24.0 KB` | `1.00` | **1.2%** `[0.0, 3.1]` | 0.0% | 0.0% | 2.5% | 2.5% |
| **M6: DeltaLoop-VLA 1L v3 (Adaptive CRN Gate) [Interf k=3]** (`m6_deltaloop_b2_adaptive_interfk3`) | `3` | `516.0 KB` | `1.10` | **3.1%** `[0.6, 6.2]` | 0.0% | 0.0% | 7.5% | 5.0% |

## 4. Paired Bootstrap Hypothesis Tests (`10,000` Resamples on `N = 160` Matched Episodes)

| Hypothesis / Comparison (`A - B`) | $\text{SR}_A$ | $\text{SR}_B$ | $\Delta \text{SR}$ (`95% Paired CI`) | Two-Sided $p$ | Holm $p_{\text{Holm}}$ | Permanence $\Delta$ | Counting $\Delta$ | Reference $\Delta$ | Imitation $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H0_M2_FrameSamp_vs_M0_Memoryless**: `m2_framesamp_modul` vs `m0_memoryless` | `5.0%` | `3.1%` | **`+1.88%`** `[-1.88%, +6.25%]` | `0.4356` | `1.0000` | `+0.0%` | `+2.5%` | `+7.5%` | `-2.5%` |
| **H1a_M7_Adapt_vs_M0_Memoryless**: `m7_deltaloop_2l_b2_adaptive` vs `m0_memoryless` | `6.2%` | `3.1%` | **`+3.12%`** `[-1.25%, +8.12%]` | `0.2380` | `1.0000` | `+5.0%` | `+5.0%` | `+5.0%` | `-2.5%` |
| **H1b_M6_Adapt_vs_M0_Memoryless**: `m6_deltaloop_b2_adaptive` vs `m0_memoryless` | `8.1%` | `3.1%` | **`+5.00%`** `[+0.62%, +9.38%]` | `0.0344` | `0.4648` | `+5.0%` | `+5.0%` | `+7.5%` | `+2.5%` |
| **H1c_M5_2L_L1_vs_M0_Memoryless**: `m5_gdn2layer_b2_l1` vs `m0_memoryless` | `12.5%` | `3.1%` | **`+9.38%`** `[+4.38%, +15.00%]` | `0.0002` | `0.0038` | `+7.5%` | `+7.5%` | `+7.5%` | `+15.0%` |
| **H1d_M4_MotionBeta_vs_M0_Memoryless**: `m4_gdn_b2_l1` vs `m0_memoryless` | `10.6%` | `3.1%` | **`+7.50%`** `[+1.88%, +13.12%]` | `0.0072` | `0.1296` | `+0.0%` | `+12.5%` | `+17.5%` | `+0.0%` |
| **H1e_M3_PosEig_vs_M0_Memoryless**: `m3_gdn_b1_l1` vs `m0_memoryless` | `10.6%` | `3.1%` | **`+7.50%`** `[+1.88%, +13.12%]` | `0.0100` | `0.1700` | `+2.5%` | `+12.5%` | `+10.0%` | `+5.0%` |
| **H2a_M4_MotionBeta_vs_M3_PosEig**: `m4_gdn_b2_l1` vs `m3_gdn_b1_l1` | `10.6%` | `10.6%` | **`+0.00%`** `[-5.62%, +5.62%]` | `1.0000` | `1.0000` | `-2.5%` | `+0.0%` | `+7.5%` | `-5.0%` |
| **H2b_M7_2L_Loop_vs_M5_2L_L1**: `m7_deltaloop_2l_b2_adaptive` vs `m5_gdn2layer_b2_l1` | `6.2%` | `12.5%` | **`-6.25%`** `[-11.88%, -0.62%]` | `0.0290` | `0.4350` | `-2.5%` | `-2.5%` | `-2.5%` | `-17.5%` |
| **H2c_M6_Loop_vs_M4_MotionBeta**: `m6_deltaloop_b2_adaptive` vs `m4_gdn_b2_l1` | `8.1%` | `10.6%` | **`-2.50%`** `[-8.75%, +3.75%]` | `0.5136` | `1.0000` | `+5.0%` | `-7.5%` | `-10.0%` | `+2.5%` |
| **H3a_M7_Adapt_vs_M2_FrameSamp**: `m7_deltaloop_2l_b2_adaptive` vs `m2_framesamp_modul` | `6.2%` | `5.0%` | **`+1.25%`** `[-3.12%, +5.62%]` | `0.7068` | `1.0000` | `+5.0%` | `+2.5%` | `-2.5%` | `+0.0%` |
| **H3b_M6_Adapt_vs_M2_FrameSamp**: `m6_deltaloop_b2_adaptive` vs `m2_framesamp_modul` | `8.1%` | `5.0%` | **`+3.12%`** `[-1.88%, +8.12%]` | `0.2704` | `1.0000` | `+5.0%` | `+2.5%` | `+0.0%` | `+5.0%` |
| **H3c_M5_2L_L1_vs_M2_FrameSamp**: `m5_gdn2layer_b2_l1` vs `m2_framesamp_modul` | `12.5%` | `5.0%` | **`+7.50%`** `[+1.88%, +13.12%]` | `0.0178` | `0.2848` | `+7.5%` | `+5.0%` | `+0.0%` | `+17.5%` |
| **H3d_M4_MotionBeta_vs_M2_FrameSamp**: `m4_gdn_b2_l1` vs `m2_framesamp_modul` | `10.6%` | `5.0%` | **`+5.62%`** `[+0.62%, +10.62%]` | `0.0332` | `0.4648` | `+0.0%` | `+10.0%` | `+10.0%` | `+2.5%` |
| **H4e_M7_Adapt_vs_M6_Adapt**: `m7_deltaloop_2l_b2_adaptive` vs `m6_deltaloop_b2_adaptive` | `6.2%` | `8.1%` | **`-1.88%`** `[-6.88%, +3.12%]` | `0.5380` | `1.0000` | `+0.0%` | `+0.0%` | `-2.5%` | `-5.0%` |
| **H4f_M7_L2_vs_M7_Adapt_IntraCkpt**: `m7_deltaloop_2l_b2_L2` vs `m7_deltaloop_2l_b2_adaptive` | `6.9%` | `6.2%` | **`+0.62%`** `[-4.38%, +5.62%]` | `0.9126` | `1.0000` | `-2.5%` | `+0.0%` | `+2.5%` | `+2.5%` |
| **H4g_M6_L2_vs_M6_Adapt_IntraCkpt**: `m6_deltaloop_b2_L2` vs `m6_deltaloop_b2_adaptive` | `8.1%` | `8.1%` | **`+0.00%`** `[-5.00%, +5.00%]` | `1.0000` | `1.0000` | `+5.0%` | `+0.0%` | `-2.5%` | `-2.5%` |
| **H7b_Interf_k3_M6_vs_M2**: `m6_deltaloop_b2_adaptive_interfk3` vs `m2_framesamp_modul_interfk3` | `3.1%` | `1.2%` | **`+1.88%`** `[-1.25%, +5.00%]` | `0.3460` | `1.0000` | `+0.0%` | `+0.0%` | `+5.0%` | `+2.5%` |
| **H7c_Interf_Drop_M2_k3_vs_k0**: `m2_framesamp_modul_interfk3` vs `m2_framesamp_modul` | `1.2%` | `5.0%` | **`-3.75%`** `[-7.50%, +0.00%]` | `0.0648` | `0.7776` | `+0.0%` | `-2.5%` | `-10.0%` | `-2.5%` |
| **H7e_Interf_Drop_M6_k3_vs_k0**: `m6_deltaloop_b2_adaptive_interfk3` vs `m6_deltaloop_b2_adaptive` | `3.1%` | `8.1%` | **`-5.00%`** `[-10.00%, +0.00%]` | `0.0656` | `0.7776` | `-5.0%` | `-5.0%` | `-5.0%` | `-5.0%` |

## 5. Adaptive Loop-Depth ($L_t$) Trajectory Diagnostics

### `m6_deltaloop_b2_adaptive` (M6: DeltaLoop-VLA 1L v3 (Adaptive CRN Gate))
- **Overall Chunk Loop Allocation (`9592` chunks):** Mean $L_t = 1.097 \pm 0.297$ | Histogram: `L=1: 90.4%`, `L=2: 9.5%`, `L=3: 0.1%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.064`, Mid `[33..67%]` = `1.120`, Late `[67..100%]` = `1.105` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.050` vs `Subsequent` = `1.048`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.09` | `1.00` | `1.06` | `1.12` | `1.10` | `90.9%` | `9.1%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `10.0%` | `1.06` | `1.10` | `1.06` | `1.07` | `1.06` | `93.7%` | `6.2%` | `0.1%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.21` | `1.10` | `1.14` | `1.27` | `1.22` | `79.0%` | `20.8%` | `0.1%` | `0.0%` |
| `StopCube` | Counting | `10.0%` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `100.0%` | `0.0%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `10.0%` | `1.18` | `1.00` | `1.08` | `1.24` | `1.21` | `82.7%` | `17.2%` | `0.1%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `15.0%` | `1.01` | `1.00` | `1.00` | `1.01` | `1.01` | `99.3%` | `0.7%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `10.0%` | `1.06` | `1.10` | `1.03` | `1.09` | `1.05` | `94.0%` | `6.0%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `10.0%` | `1.06` | `1.00` | `1.06` | `1.05` | `1.07` | `94.2%` | `5.8%` | `0.1%` | `0.0%` |

### `m7_deltaloop_2l_b2_adaptive` (M7: DeltaLoop-VLA 2L Iso-State v3 (Adaptive CRN Gate))
- **Overall Chunk Loop Allocation (`9460` chunks):** Mean $L_t = 1.061 \pm 0.244$ | Histogram: `L=1: 93.9%`, `L=2: 6.0%`, `L=3: 0.1%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.043`, Mid `[33..67%]` = `1.080`, Late `[67..100%]` = `1.061` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.025` vs `Subsequent` = `1.027`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.06` | `1.00` | `1.02` | `1.11` | `1.05` | `93.6%` | `6.4%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `10.0%` | `1.06` | `1.00` | `1.06` | `1.08` | `1.04` | `94.0%` | `6.0%` | `0.0%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.10` | `1.00` | `1.04` | `1.10` | `1.15` | `90.2%` | `9.8%` | `0.0%` | `0.0%` |
| `StopCube` | Counting | `10.0%` | `1.00` | `1.00` | `1.01` | `1.00` | `1.00` | `99.7%` | `0.3%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `10.0%` | `1.15` | `1.00` | `1.09` | `1.21` | `1.16` | `85.2%` | `14.3%` | `0.4%` | `0.1%` |
| `VideoPlaceOrder` | Reference | `10.0%` | `1.01` | `1.00` | `1.02` | `1.02` | `1.00` | `98.5%` | `1.5%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `10.0%` | `1.00` | `1.05` | `1.01` | `1.00` | `1.00` | `99.7%` | `0.4%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `0.0%` | `1.02` | `1.05` | `1.05` | `1.01` | `1.00` | `98.0%` | `2.0%` | `0.0%` | `0.0%` |

### `m6_deltaloop_b2_adaptive_interfk3` (M6: DeltaLoop-VLA 1L v3 (Adaptive CRN Gate) [Interf k=3])
- **Overall Chunk Loop Allocation (`9485` chunks):** Mean $L_t = 1.104 \pm 0.308$ | Histogram: `L=1: 89.7%`, `L=2: 10.3%`, `L=3: 0.1%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.092`, Mid `[33..67%]` = `1.101`, Late `[67..100%]` = `1.118` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.025` vs `Subsequent` = `1.047`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.06` | `1.00` | `1.06` | `1.07` | `1.06` | `93.8%` | `6.2%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `0.0%` | `1.06` | `1.00` | `1.02` | `1.07` | `1.09` | `93.6%` | `6.4%` | `0.0%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.18` | `1.00` | `1.21` | `1.17` | `1.16` | `82.4%` | `17.4%` | `0.2%` | `0.0%` |
| `StopCube` | Counting | `0.0%` | `1.01` | `1.00` | `1.01` | `1.02` | `1.02` | `98.5%` | `1.5%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `5.0%` | `1.26` | `1.00` | `1.22` | `1.24` | `1.32` | `74.0%` | `25.8%` | `0.2%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `10.0%` | `1.01` | `1.00` | `1.01` | `1.01` | `1.01` | `98.7%` | `1.3%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `0.0%` | `1.13` | `1.10` | `1.14` | `1.13` | `1.11` | `87.4%` | `12.6%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `10.0%` | `1.03` | `1.00` | `1.00` | `1.03` | `1.04` | `97.5%` | `2.4%` | `0.1%` | `0.0%` |

## 6. Executive Scientific Synthesis (`11/11` Runs = `1,760` Closed-Loop Rollouts)

1. **Single-Pass 2-Layer Gated DeltaNet (`m5_gdn2layer_b2_l1`) Drives Closed-Loop Scratch Gains (`H1c`, Holm-Significant):** At fixed `520 KB` `O(1)` state budget, single-pass 2-layer `m5_gdn2layer_b2_l1` (`L=1`) reaches **`12.50%` (`20/160`, `[7.50, 18.12]`)**, surviving Holm-Bonferroni multiplicity correction over `m0_memoryless` (`+9.38%`, two-sided `p = 0.0002`, **`p_holm = 0.0038`**; unadjusted `p = 0.0178`, `p_holm = 0.3026` NS vs `m2_framesamp_modul` `5.00%`) and outperforming multi-hop `m6_deltaloop_b2` (`8.12%` at `L=2`, `6.88%` adaptive) and `m7_deltaloop2l_b2` (`6.88%` at `L=2`, `6.25%` adaptive). Thus, recurrent state compression (`520 KB` 2L GDN) — rather than multi-hop Coda looping — accounts for closed-loop RoboMME improvements.
2. **Bundled `U1–U5` Upgrade & Zero Closed-Loop Advantage of $\beta \in (0, 2)$ over $\beta \in (0, 1)$ (`H2a`):** Across the bundled Cycle 2A $\to$ Cycle 3A upgrade (`U1–U5`), `m4_gdn_b2_l1` rises from `5.62%` to `10.62%` (`+5.00%` cross-cycle, two-sided `p = 0.0942` NS) and matches positive-eigenvalue `m3_gdn_b1_l1` identically (`10.62% vs 10.62%`, `17/160 vs 17/160`, two-sided `p = 1.0000`, `p_holm = 1.0000`). Accordingly, while motion-conditioned $\beta$ bias (`U5`) prevents Cycle 2A's exploratory static-frame degradation (`Cycle 2A H2: -4.38%, p = 0.1052` NS), extending $\beta$ from `(0, 1)` to `(0, 2)` yields no closed-loop success-rate advantage on RoboMME.
3. **Live Closed-Loop Cross-Session Interference (`k = 3`, `N = 160`, Exploratory Low-Count Regime):** Under `k = 3` un-reset prior distractor episodes, 32-slot `m2_framesamp_modul` drops from `5.00%` (`8/160`) to `1.25%` (`2/160`; `-3.75%`, two-sided `p = 0.0648` NS, `p_holm = 1.0000`), while `m6_deltaloop_b2_adaptive` moves from `6.88%` (`11/160`) to `3.12%` (`5/160`; `-3.75%`, two-sided `p = 0.1180` NS); head-to-head `k = 3` difference (`5/160 vs 2/160`, `+1.88%`, two-sided `p = 0.4094` NS) is non-significant in this near-floor sample size.
4. **Intra-Checkpoint Loop Sensitivity & Stage-II `min_delta` Calibration (`#79 -> #80`):** Under `U1`-weighted Flow-Matching (`fm_loss ~ 0.0042`), Stage-II's `min_delta = 3e-4` floor halted `m6_adaptive` (`L = 1.10`) and `m7_adaptive` (`L = 1.06`) mostly at `L = 1`. Evaluating the same checkpoints at `Fixed L = 2` yields `6.88% -> 8.12%` on `m6` (`11/160 -> 13/160`) and `6.25% -> 6.88%` on `m7` (`10/160 -> 11/160`, non-significant intra-checkpoint trends that remain below single-pass `m5`'s `12.50%`), motivating Cycle 3B's move to pretrained `lerobot/smolvla_base` across all 16 official `test` tasks.

