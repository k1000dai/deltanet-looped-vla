# Cycle 2A — Controlled Multi-Suite RoboMME Evaluation Summary

- **Protocol:** 8 Representative RoboMME Tasks (`2 tasks × 4 memory suites`), `val` split (`20` episodes/task = **`160` matched episodes per evaluation run**, `model_seed=7`, `exec_horizon=16`, `max_steps=1300`, 10-step Flow-Matching Euler ODE).
- **Evaluated Runs Completed:** `15` (`results/robomme/cycle2a/*.json`)

## 1. Overall & Per-Suite Success Rates (`N = 160` Matched Episodes)

| Evaluation Run | Params | State Size | Train FM Loss | Mean $L_t$ (`p95`) | Latency (`ms/ch`) | Overall SR (`95% CI`) | Permanence (`40`) | Counting (`40`) | Reference (`40`) | Imitation (`40`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M0: Memoryless Flow (L=1)** (`m0_memoryless`) | `27.47M` | `0 B` | `0.1017` | `1.00` (`1.0`) | `36.2` | **1.9%** `[0.0, 4.4]` | 2.5% | 0.0% | 0.0% | 5.0% |
| **M1: RD-VLA Memoryless Loop (L=4)** (`m1_rdvla_loop4`) | `31.77M` | `0 B` | `0.0997` | `4.00` (`4.0`) | `47.3` | **0.6%** `[0.0, 1.9]` | 0.0% | 0.0% | 2.5% | 0.0% |
| **M2: FrameSamp-32 + Modul/XAttn** (`m2_framesamp_modul`) | `31.48M` | `24.0 KB` | `0.0686` | `1.00` (`1.0`) | `58.9` | **5.0%** `[1.9, 8.8]` | 5.0% | 2.5% | 7.5% | 5.0% |
| **M3: GDN beta in (0,1) Single-Pass (L=1)** (`m3_gdn_b1_l1`) | `35.14M` | `130.0 KB` | `0.0647` | `1.00` (`1.0`) | `65.2` | **10.0%** `[5.6, 15.0]` | 12.5% | 2.5% | 15.0% | 10.0% |
| **M4: GDN beta in (0,2) Householder (L=1)** (`m4_gdn_b2_l1`) | `35.14M` | `130.0 KB` | `0.0647` | `1.00` (`1.0`) | `56.7` | **5.6%** `[2.5, 9.4]` | 7.5% | 5.0% | 5.0% | 5.0% |
| **M5: 2-Layer Iso-State GDN beta in (0,2) (L=1)** (`m5_gdn2layer_b2_l1`) | `36.81M` | `132.0 KB` | `0.0632` | `1.00` (`1.0`) | `65.5` | **11.2%** `[6.9, 16.2]` | 15.0% | 10.0% | 15.0% | 5.0% |
| **M6: DeltaLoop-VLA (Fixed L=1)** (`m6_deltaloop_b2_L1`) | `35.29M` | `130.0 KB` | `0.0629` | `1.00` (`1.0`) | `64.2` | **5.6%** `[2.5, 9.4]` | 5.0% | 5.0% | 12.5% | 0.0% |
| **M6: DeltaLoop-VLA (Fixed L=2)** (`m6_deltaloop_b2_L2`) | `35.29M` | `130.0 KB` | `0.0629` | `2.00` (`2.0`) | `101.1` | **8.1%** `[4.4, 12.5]` | 7.5% | 10.0% | 12.5% | 2.5% |
| **M6: DeltaLoop-VLA (Fixed L=4)** (`m6_deltaloop_b2_L4`) | `35.29M` | `130.0 KB` | `0.0629` | `4.00` (`4.0`) | `73.2` | **7.5%** `[3.8, 11.9]` | 7.5% | 5.0% | 17.5% | 0.0% |
| **M6: DeltaLoop-VLA (Adaptive CRN Gate)** (`m6_deltaloop_b2_adaptive`) | `35.29M` | `130.0 KB` | `0.0629` | `1.52` (`2.0`) | `60.0` | **7.5%** `[3.8, 11.9]` | 7.5% | 5.0% | 10.0% | 7.5% |
| **M6: DeltaLoop-VLA (Latent L2 Conv tau=0.22)** (`m6_deltaloop_b2_coda_conv`) | `35.29M` | `130.0 KB` | `0.0629` | `2.38` (`3.0`) | `61.8` | **6.9%** `[3.1, 11.2]` | 10.0% | 5.0% | 10.0% | 2.5% |
| **M6b: DeltaLoop-VLA + WriteBack (Fixed L=1)** (`m6b_deltaloop_wb_L1`) | `35.44M` | `130.0 KB` | `0.1122` | `1.00` (`1.0`) | `64.4` | **2.5%** `[0.6, 5.0]` | 0.0% | 0.0% | 10.0% | 0.0% |
| **M6b: DeltaLoop-VLA + WriteBack (Fixed L=2)** (`m6b_deltaloop_wb_L2`) | `35.44M` | `130.0 KB` | `0.1122` | `2.00` (`2.0`) | `61.6` | **3.1%** `[0.6, 6.2]` | 0.0% | 0.0% | 12.5% | 0.0% |
| **M6b: DeltaLoop-VLA + WriteBack (Fixed L=4)** (`m6b_deltaloop_wb_L4`) | `35.44M` | `130.0 KB` | `0.1122` | `4.00` (`4.0`) | `68.4` | **4.4%** `[1.2, 7.5]` | 0.0% | 0.0% | 15.0% | 2.5% |
| **M6b: DeltaLoop-VLA + WriteBack (Adaptive CRN Gate)** (`m6b_deltaloop_wb_adaptive`) | `35.44M` | `130.0 KB` | `0.1122` | `1.63` (`2.0`) | `61.5` | **2.5%** `[0.6, 5.0]` | 0.0% | 0.0% | 7.5% | 2.5% |

## 2. Per-Task Success Rates (`20` Episodes / Task)

| Evaluation Run | Overall | `ButtonUnmaskSwap` | `VideoUnmaskSwap` | `PickXtimes` | `StopCube` | `PickHighlight` | `VideoPlaceOrder` | `PatternLock` | `MoveCube` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `m0_memoryless` | **1.9%** | 0.0% | 5.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 10.0% |
| `m1_rdvla_loop4` | **0.6%** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 5.0% | 0.0% | 0.0% |
| `m2_framesamp_modul` | **5.0%** | 0.0% | 10.0% | 0.0% | 5.0% | 10.0% | 5.0% | 0.0% | 10.0% |
| `m3_gdn_b1_l1` | **10.0%** | 10.0% | 15.0% | 0.0% | 5.0% | 10.0% | 20.0% | 5.0% | 15.0% |
| `m4_gdn_b2_l1` | **5.6%** | 10.0% | 5.0% | 0.0% | 10.0% | 5.0% | 5.0% | 5.0% | 5.0% |
| `m5_gdn2layer_b2_l1` | **11.2%** | 15.0% | 15.0% | 0.0% | 20.0% | 15.0% | 15.0% | 0.0% | 10.0% |
| `m6_deltaloop_b2_L1` | **5.6%** | 5.0% | 5.0% | 0.0% | 10.0% | 0.0% | 25.0% | 0.0% | 0.0% |
| `m6_deltaloop_b2_L2` | **8.1%** | 5.0% | 10.0% | 5.0% | 15.0% | 0.0% | 25.0% | 0.0% | 5.0% |
| `m6_deltaloop_b2_L4` | **7.5%** | 5.0% | 10.0% | 0.0% | 10.0% | 10.0% | 25.0% | 0.0% | 0.0% |
| `m6_deltaloop_b2_adaptive` | **7.5%** | 5.0% | 10.0% | 0.0% | 10.0% | 0.0% | 20.0% | 0.0% | 15.0% |
| `m6_deltaloop_b2_coda_conv` | **6.9%** | 10.0% | 10.0% | 5.0% | 5.0% | 0.0% | 20.0% | 0.0% | 5.0% |
| `m6b_deltaloop_wb_L1` | **2.5%** | 0.0% | 0.0% | 0.0% | 0.0% | 10.0% | 10.0% | 0.0% | 0.0% |
| `m6b_deltaloop_wb_L2` | **3.1%** | 0.0% | 0.0% | 0.0% | 0.0% | 15.0% | 10.0% | 0.0% | 0.0% |
| `m6b_deltaloop_wb_L4` | **4.4%** | 0.0% | 0.0% | 0.0% | 0.0% | 15.0% | 15.0% | 0.0% | 5.0% |
| `m6b_deltaloop_wb_adaptive` | **2.5%** | 0.0% | 0.0% | 0.0% | 0.0% | 15.0% | 0.0% | 0.0% | 5.0% |

## 3. Paired Bootstrap Hypothesis Tests (`10,000` Resamples on `N = 160` Matched Episodes)

| Hypothesis / Comparison (`A - B`) | $\text{SR}_A$ | $\text{SR}_B$ | $\Delta \text{SR}$ (`95% Paired CI`) | Two-Sided $p$ | Holm $p_{\text{Holm}}$ | Permanence $\Delta$ | Counting $\Delta$ | Reference $\Delta$ | Imitation $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H0a_M1_RDVLA_Loop4_vs_M0_Memoryless**: `m1_rdvla_loop4` vs `m0_memoryless` | `0.6%` | `1.9%` | **`-1.25%`** `[-3.75%, +1.25%]` | `0.4388` | `1.0000` | `-2.5%` | `+0.0%` | `+2.5%` | `-5.0%` |
| **H0b_M2_FrameSamp_vs_M0_Memoryless**: `m2_framesamp_modul` vs `m0_memoryless` | `5.0%` | `1.9%` | **`+3.12%`** `[-0.62%, +6.88%]` | `0.1118` | `1.0000` | `+2.5%` | `+2.5%` | `+7.5%` | `+0.0%` |
| **H0c_M4_GDN_L1_vs_M0_Memoryless**: `m4_gdn_b2_l1` vs `m0_memoryless` | `5.6%` | `1.9%` | **`+3.75%`** `[-0.62%, +8.12%]` | `0.1038` | `1.0000` | `+5.0%` | `+5.0%` | `+5.0%` | `+0.0%` |
| **H1a_M6_Adapt_vs_M0_Memoryless**: `m6_deltaloop_b2_adaptive` vs `m0_memoryless` | `7.5%` | `1.9%` | **`+5.62%`** `[+1.25%, +10.62%]` | `0.0234` | `0.3978` | `+5.0%` | `+5.0%` | `+10.0%` | `+2.5%` |
| **H1b_M6_Adapt_vs_M1_RDVLA_Loop4**: `m6_deltaloop_b2_adaptive` vs `m1_rdvla_loop4` | `7.5%` | `0.6%` | **`+6.88%`** `[+2.50%, +11.25%]` | `0.0010` | `0.0180` | `+7.5%` | `+5.0%` | `+7.5%` | `+7.5%` |
| **H2_M4_NegEig_b2_vs_M3_PosEig_b1**: `m4_gdn_b2_l1` vs `m3_gdn_b1_l1` | `5.6%` | `10.0%` | **`-4.38%`** `[-9.38%, +0.62%]` | `0.1046` | `1.0000` | `-5.0%` | `+2.5%` | `-10.0%` | `-5.0%` |
| **H3a_M6_Adapt_vs_M2_FrameSamp**: `m6_deltaloop_b2_adaptive` vs `m2_framesamp_modul` | `7.5%` | `5.0%` | **`+2.50%`** `[-1.88%, +6.88%]` | `0.3534` | `1.0000` | `+2.5%` | `+2.5%` | `+2.5%` | `+2.5%` |
| **H3b_M4_GDN_L1_vs_M2_FrameSamp**: `m4_gdn_b2_l1` vs `m2_framesamp_modul` | `5.6%` | `5.0%` | **`+0.62%`** `[-4.38%, +5.00%]` | `0.9016` | `1.0000` | `+2.5%` | `+2.5%` | `-2.5%` | `+0.0%` |
| **H4a_M6_L4_vs_M6_L1_IntraCkpt**: `m6_deltaloop_b2_L4` vs `m6_deltaloop_b2_L1` | `7.5%` | `5.6%` | **`+1.88%`** `[-2.50%, +6.25%]` | `0.4852` | `1.0000` | `+2.5%` | `+0.0%` | `+5.0%` | `+0.0%` |
| **H4b_M6_L2_vs_M6_L1_IntraCkpt**: `m6_deltaloop_b2_L2` vs `m6_deltaloop_b2_L1` | `8.1%` | `5.6%` | **`+2.50%`** `[-1.88%, +6.88%]` | `0.3116` | `1.0000` | `+2.5%` | `+5.0%` | `+0.0%` | `+2.5%` |
| **H4c_M6_L4_vs_M4_SinglePass_GDN**: `m6_deltaloop_b2_L4` vs `m4_gdn_b2_l1` | `7.5%` | `5.6%` | **`+1.88%`** `[-3.12%, +6.88%]` | `0.5418` | `1.0000` | `+0.0%` | `+0.0%` | `+12.5%` | `-5.0%` |
| **H4d_M6_Adapt_vs_M4_SinglePass_GDN**: `m6_deltaloop_b2_adaptive` vs `m4_gdn_b2_l1` | `7.5%` | `5.6%` | **`+1.88%`** `[-3.12%, +6.88%]` | `0.5172` | `1.0000` | `+0.0%` | `+0.0%` | `+5.0%` | `+2.5%` |
| **H4e_M6_L4_vs_M5_2Layer_IsoState**: `m6_deltaloop_b2_L4` vs `m5_gdn2layer_b2_l1` | `7.5%` | `11.2%` | **`-3.75%`** `[-10.00%, +2.50%]` | `0.2632` | `1.0000` | `-7.5%` | `-5.0%` | `+2.5%` | `-5.0%` |
| **H4f_M6_Adapt_vs_M5_2Layer_IsoState**: `m6_deltaloop_b2_adaptive` vs `m5_gdn2layer_b2_l1` | `7.5%` | `11.2%` | **`-3.75%`** `[-9.38%, +1.88%]` | `0.2374` | `1.0000` | `-7.5%` | `-5.0%` | `-5.0%` | `+2.5%` |
| **H5a_M6_Adapt_vs_M6_L4**: `m6_deltaloop_b2_adaptive` vs `m6_deltaloop_b2_L4` | `7.5%` | `7.5%` | **`+0.00%`** `[-5.00%, +5.00%]` | `1.0000` | `1.0000` | `+0.0%` | `+0.0%` | `-7.5%` | `+7.5%` |
| **H5b_M6_Adapt_vs_M6_CodaConv**: `m6_deltaloop_b2_adaptive` vs `m6_deltaloop_b2_coda_conv` | `7.5%` | `6.9%` | **`+0.62%`** `[-3.12%, +4.38%]` | `0.8588` | `1.0000` | `-2.5%` | `+0.0%` | `+0.0%` | `+5.0%` |
| **H6a_M6b_WB_Adapt_vs_M6_Adapt**: `m6b_deltaloop_wb_adaptive` vs `m6_deltaloop_b2_adaptive` | `2.5%` | `7.5%` | **`-5.00%`** `[-10.00%, -0.62%]` | `0.0478` | `0.7648` | `-7.5%` | `-5.0%` | `-2.5%` | `-5.0%` |
| **H6b_M6b_WB_L4_vs_M6_L4**: `m6b_deltaloop_wb_L4` vs `m6_deltaloop_b2_L4` | `4.4%` | `7.5%` | **`-3.12%`** `[-8.12%, +1.25%]` | `0.2376` | `1.0000` | `-7.5%` | `-5.0%` | `-2.5%` | `+2.5%` |

## 4. Adaptive Loop-Depth ($L_t$) Trajectory Diagnostics

### `m6_deltaloop_b2_adaptive` (M6: DeltaLoop-VLA (Adaptive CRN Gate))
- **Overall Chunk Loop Allocation (`8166` chunks):** Mean $L_t = 1.521 \pm 0.500$ | Histogram: `L=1: 47.9%`, `L=2: 52.1%`, `L=3: 0.0%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.394`, Mid `[33..67%]` = `1.590`, Late `[67..100%]` = `1.575` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.325` vs `Subsequent` = `1.534`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmaskSwap` | Permanence | `5.0%` | `1.60` | `1.00` | `1.37` | `1.68` | `1.75` | `39.8%` | `60.2%` | `0.1%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `10.0%` | `1.62` | `1.35` | `1.54` | `1.67` | `1.66` | `37.5%` | `62.5%` | `0.0%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.42` | `1.00` | `1.36` | `1.46` | `1.42` | `58.5%` | `41.5%` | `0.0%` | `0.0%` |
| `StopCube` | Counting | `10.0%` | `1.01` | `1.00` | `1.00` | `1.00` | `1.03` | `99.0%` | `1.0%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `0.0%` | `1.60` | `1.00` | `1.47` | `1.66` | `1.68` | `39.7%` | `60.2%` | `0.1%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `20.0%` | `1.47` | `1.00` | `1.31` | `1.47` | `1.63` | `52.6%` | `47.4%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `0.0%` | `1.54` | `1.90` | `1.33` | `1.75` | `1.53` | `46.0%` | `54.0%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `15.0%` | `1.46` | `1.05` | `1.39` | `1.54` | `1.46` | `53.8%` | `46.2%` | `0.0%` | `0.0%` |

### `m6_deltaloop_b2_coda_conv` (M6: DeltaLoop-VLA (Latent L2 Conv tau=0.22))
- **Overall Chunk Loop Allocation (`8375` chunks):** Mean $L_t = 2.380 \pm 0.485$ | Histogram: `L=1: 0.0%`, `L=2: 62.0%`, `L=3: 38.0%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `2.241`, Mid `[33..67%]` = `2.427`, Late `[67..100%]` = `2.467` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `2.013` vs `Subsequent` = `2.434`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmaskSwap` | Permanence | `10.0%` | `2.29` | `2.00` | `2.15` | `2.33` | `2.38` | `0.0%` | `71.2%` | `28.8%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `10.0%` | `2.39` | `2.00` | `2.27` | `2.40` | `2.51` | `0.0%` | `60.7%` | `39.3%` | `0.0%` |
| `PickXtimes` | Counting | `5.0%` | `2.37` | `2.00` | `2.27` | `2.42` | `2.42` | `0.0%` | `62.9%` | `37.1%` | `0.0%` |
| `StopCube` | Counting | `5.0%` | `2.25` | `2.00` | `2.02` | `2.26` | `2.44` | `0.0%` | `75.1%` | `24.9%` | `0.0%` |
| `PickHighlight` | Reference | `0.0%` | `2.36` | `2.00` | `2.20` | `2.41` | `2.44` | `0.0%` | `64.4%` | `35.6%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `20.0%` | `2.31` | `2.00` | `2.22` | `2.35` | `2.35` | `0.0%` | `69.3%` | `30.7%` | `0.0%` |
| `PatternLock` | Imitation | `0.0%` | `2.50` | `2.05` | `2.16` | `2.64` | `2.68` | `0.0%` | `50.1%` | `49.9%` | `0.0%` |
| `MoveCube` | Imitation | `5.0%` | `2.49` | `2.00` | `2.41` | `2.54` | `2.52` | `0.0%` | `51.0%` | `49.0%` | `0.0%` |

### `m6b_deltaloop_wb_adaptive` (M6b: DeltaLoop-VLA + WriteBack (Adaptive CRN Gate))
- **Overall Chunk Loop Allocation (`9887` chunks):** Mean $L_t = 1.633 \pm 0.482$ | Histogram: `L=1: 36.7%`, `L=2: 63.3%`, `L=3: 0.0%`, `L=4: 0.0%`
- **Trajectory Phase Profile:** Early `[0..33%]` = `1.405`, Mid `[33..67%]` = `1.705`, Late `[67..100%]` = `1.781` | Video-Demo Tasks: `Chunk 0 (post-demo)` = `1.512` vs `Subsequent` = `1.635`

| Task | Suite | Task SR | Mean $L_t$ | Chunk-0 $L_t$ | Early $L_t$ | Mid $L_t$ | Late $L_t$ | `% L=1` | `% L=2` | `% L=3` | `% L=4` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ButtonUnmaskSwap` | Permanence | `0.0%` | `1.74` | `1.00` | `1.39` | `1.87` | `1.95` | `26.2%` | `73.8%` | `0.0%` | `0.0%` |
| `VideoUnmaskSwap` | Permanence | `0.0%` | `1.74` | `2.00` | `1.57` | `1.74` | `1.90` | `26.2%` | `73.8%` | `0.0%` | `0.0%` |
| `PickXtimes` | Counting | `0.0%` | `1.62` | `1.00` | `1.34` | `1.72` | `1.81` | `37.5%` | `62.5%` | `0.0%` | `0.0%` |
| `StopCube` | Counting | `0.0%` | `1.00` | `1.00` | `1.00` | `1.00` | `1.00` | `100.0%` | `0.0%` | `0.0%` | `0.0%` |
| `PickHighlight` | Reference | `15.0%` | `1.69` | `1.00` | `1.44` | `1.76` | `1.85` | `31.3%` | `68.7%` | `0.0%` | `0.0%` |
| `VideoPlaceOrder` | Reference | `0.0%` | `1.73` | `1.00` | `1.47` | `1.80` | `1.93` | `26.6%` | `73.4%` | `0.0%` | `0.0%` |
| `PatternLock` | Imitation | `0.0%` | `1.36` | `1.20` | `1.05` | `1.67` | `1.36` | `63.5%` | `36.5%` | `0.0%` | `0.0%` |
| `MoveCube` | Imitation | `5.0%` | `1.55` | `1.85` | `1.47` | `1.53` | `1.65` | `44.7%` | `55.3%` | `0.0%` | `0.0%` |

