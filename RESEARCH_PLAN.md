# Research Plan (v3.5 Final — Verified Empirical Record): DeltaLoop-VLA

**Working Title:** *DeltaLoop-VLA: Expressive Associative Memory Writing and Iterative Multi-Hop Readout for Long-Horizon Robotic Manipulation*  
**Status:** **ALL CAMPAIGNS COMPLETE** — **Gates 0, 1, 2A, 3A = COMPLETED (`87` Synthetic GPU Runs + `4,160` Scratch Rollouts)** | **All `14/14` Pretrained `SmolVLA` Checkpoints & `202` Mechanistic Probes (`64 + 56 + 82`) = COMPLETE** | **Pretrained `lerobot/smolvla_base` Official `test` Closed-Loop Suite (Incl. Single-Pass `M5 > M7` Finding, Holm-Bonferroni Multiplicity Control, 3-Seed `{42,43,44}` `Pooled N=720`, & Live `k=3` Cross-Session Controls) = COMPLETE**  
**Primary Pretrained Backbone & Benchmark:** Pretrained `lerobot/smolvla_base` (`450M` Total / `407M–444M` Active: Frozen `SmolVLM2` 16L w/ `8×8`-pooled cached tokens + Pretrained `99.88M` Flow Action Expert; [docs/PRETRAINED_VLA_DESIGN.md](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/docs/PRETRAINED_VLA_DESIGN.md)) + Controlled `v3` Scratch Suite on [RoboMME](https://github.com/RoboMME/robomme_benchmark) (`arXiv:2603.04639`, All 16 Tasks, 10 Hz) & Live Closed-Loop Cross-Session Interference (`arXiv:2606.22338`)  
**Companion Documents:** [RESULTS_REPORT.md](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/RESULTS_REPORT.md) | [README.md](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/README.md) | [docs/SURVEY.md](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/docs/SURVEY.md) | [docs/PRETRAINED_VLA_DESIGN.md](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/docs/PRETRAINED_VLA_DESIGN.md) | [SUMMARY_CYCLE1.md](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/results/synthetic/SUMMARY_CYCLE1.md) | [SUMMARY_CYCLE2A.md](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/results/robomme/cycle2a/SUMMARY_CYCLE2A.md) | [SUMMARY_CYCLE3A.md](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/results/robomme/cycle3a/SUMMARY_CYCLE3A.md) | [SUMMARY_CYCLE3B.md](file:///home/kohei/codes/ghq/github.com/k1000dai/deltanet-looped-vla/results/robomme/cycle3b_smolvla/SUMMARY_CYCLE3B.md)

---

## 0. 研究推進ループと検証済み計算基盤（`EXP_TIPS.md` 準拠 — 全6回転 ＆ `202` Probes 完結）

本研究計画は、以下の**高速科学検証ループ**に沿って Cycle 1 $\to$ Cycle 2A $\to$ Cycle 3A (`U1–U5`) $\to$ Cycle 3B (`U6`) $\to$ Cycle 4A (`U7`) $\to$ Cycle 4B (`U8`・同一重み深さ比較・Cross-Regime Halting・3-Seed `{42,43,44}` 統合検定 `H_MS1..H_MS3`・Holm-Bonferroni 多重比較補正) まで全工程をデータ駆動で完結させた最終計画記録（v3.5 Final）である：

$$\textbf{実験 (Exp)} \longrightarrow \textbf{考察・Probe診断 (Analysis)} \longrightarrow \textbf{深掘り調査 (VLA \& LLM Survey)} \longrightarrow \textbf{仮説・設計更新 (IDEA / Plan)} \longrightarrow \textbf{実験}$$

- **計算資源の非対称ハイブリッド運用：**
  - **Miyabi GH200 (`aarch64`, `sm_90` Grace Hopper 98GB HBM3e, FREE `interact-g` `0` budget pts)：** Cycle 2A (`8` models + `64` Probes)・Cycle 3A (`7` models + `56` Probes)・Cycle 3B (`11` multi-seed)・Cycle 4A `U7` (`2` seeds)・および Cycle 4B `U8` (`smolvla_m7_deltaloop_anchor` + `70 + 12 = 82` Probes；累計 **`202` Mechanistic Probe エントリ**) を完走。
  - **手元 NVIDIA RTX 5080 16GB (`x86_64`, `sm_120` Blackwell)：** Cycle 1（`87` synthetic GPU runs）、Cycle 2A/3A（`2,400 + 1,760 = 4,160` rollouts across `26` runs）、および **Pretrained `SmolVLA` 公式 `test` 閉ループ評価全 `20/20` runs = `6,560` rollouts**（`seed=42` `k=0` `11` runs + Multi-Seed `{s43, s44}` `6` runs + Live `k=3` 妨害対照 `3` runs [`M2`, `M5`, `M7`]；**閉ループ累計 `10,720` rollouts across `46` runs**）を完走。

---

## 1. 実証済み科学軌跡サマリー（誠実な帰無・対立仮説検証）

### 1.1 Cycle 1（87 Runs）＆ Cycle 2A/3A スクラッチ統制系列（4,160 Rollouts & 120 Probes）
1. **Write vs. Read 直交性の合成実証と外挿限界（Cycle 1, `87` Runs）：**
   - `Task 1A-Online`（$S_8$ 置換）では $\beta\in(0,2)$ が **ID/OOD `100.0 ± 0.0%`** を達成（$\beta\in(0,1)$ `25.1%`；なお 4層 Softmax Transformer の `15.6%` は共通1エポック予算下の学習不足を反映し理論上限ではない）。
   - `Task 1A-Retro`（Post-Cue 連鎖）では評価基準を統一すると、**`Forward` 順のみ**で単発・`Write-Loop`・`Parallel-MQ`・非共有6層（`21.4%` ID；勾配非共有に伴う深層最適化困難の可能性は排除できない）が近チャンス域に留まる一方、重み共有 **`DeltaLoop` は `Forward` ID `97.1±1.1%` (`b1`) / `94.8±3.4%` (`b2`)（全順序 `97.9% / 96.3%`）、ゼロショット OOD $k=5$ で `Forward` `80.4% / 76.4%`（全順序 `85.0% / 82.3%`）を達成するが、$k=6$ では `Forward` `37.3% / 37.0%`（全順序 `51.2% / 54.9%`；OOD 全体 `Forward` `58.9% / 56.8%`、全順序 `68.1% / 68.6%`）へ急落する**。CRN 停止ゲートは ID ループ計算を `-56%` 削減し、Task 1B 等状態では残差 GDN（`99.8%`）が 3バンク Softmax ルータ（`71.6–78.5%`）を上回った。
2. **スクラッチ閉ループ検証：単発 2層 GDN `m5` の優位と $\beta\in(0,2)$ 閉ループ利得不在（Cycles 2A & 3A, `4,160` Rollouts）：**
   - チャンク境界書き戻し `m6b` は劣化傾向を示し（**H6a:** `2.50% vs 7.50%`, 両側 `p=0.0478` 未補正, `p_holm=0.7648`）棄却。Cycle 2A における 1層 `m4` (`5.62%`) vs `m3` (`10.00%`) は非有意（**H2:** `p=0.1046`）ながら静止フレーム共線符号反転仮説を示唆し、共通改修 `U1–U5`＋状態拡張を施した Cycle 3A `v3` では `m4` (`10.62%`, `17/160`) と `m3` (`10.62%`, `17/160`) が完全一致（**H2a:** `p=1.0000`）— **閉ループ RoboMME では $\beta\in(0,2)$ の $\beta\in(0,1)$ に対する SR 利得はゼロであった**。
   - **最重要知見（C1）：** Cycle 2A（`m5` `11.25%` vs `m6` `7.50%`）および Cycle 3A（**`m5` `12.50%` [`20/160`, Holm 有意 `H1c: p=0.0002`, `p_holm=0.0038`] vs ループ付き `m6` `8.12%` / `m7` `6.25%` [`H2b: -6.25%, p=0.0290` 未補正]**）の双方で、**単層読み出し 2層 GDN（`m5`, $L=1$）がすべてのループ付き変種を上回った**。

### 1.2 Pretrained `SmolVLA` 公式 `test` 閉ループ評価 ＆ `82` Mechanistic Probes 実証

| Pretrained `SmolVLA` Run (`test` Split, 16 Tasks) | State (`Loop->Exp`) | Train `fm_loss` | Mean $L_t$ (`Chunk-0`) | Latency (`ms/ch`) | **Overall SR (`Count`, `95% CI`)** | Perm | Count | Ref | Imit | 両側ブートストラップ $p$ ＆ Holm 判定 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`smolvla_m0_base`** ($L=1$, $k=0$, `s42`) | `0 B` (`—`) | `0.0991±0.0016` | `1.00` (`1.00`) | `140.9` | **1.50%** (`6/400`, `[0.50, 2.75]`) | `4.0%` | `0.0%` | `0.0%` | `2.0%` | `12/16` タスク `0%`（マルコフ基底） |
| **`smolvla_m1_rdvla_loop4`** ($L=4$, $k=0$, `s42`) | `0 B` (`4->4`) | `0.0392` | `4.00` (`4.00`) | `162.8` | **2.50%** (`10/400`, `[1.00, 4.01]`) | `5.0%` | `0.0%` | `3.0%` | `2.0%` | **H0a:** `p=0.3806` NS vs `m0`（無記憶ループ不発） |
| **`smolvla_m2_framesamp`** ($k=0$, `s42`) | `24.0 KB` (`32->32`) | `0.0538±0.0003` | `1.00` (`1.00`) | `245.8` | **6.50%** (`26/400`, `[4.25, 9.00]`) | `6.0%` | `0.0%` | `9.0%` | `11.0%` | **H0b:** `p=0.0002`（**Holm 有意**；単一セッション首位） |
| **`smolvla_m5_gdn2l_l1`** ($L=1$, $k=0$, `s42`) | **`520.0 KB` (`4->4`)** | `0.0326` | `1.00` (`1.00`) | **`238.2`** | **5.25%** (`21/400`, `[3.25, 7.50]`) | `3.0%` | `1.0%` | **`9.0%`** | `8.0%` | **H1c:** `p=0.0012`（**Holm 有意**；`M7` 全種を上回る） |
| **`smolvla_m7_deltaloop_L1`** (`Fixed L=1`, `s42`) | **`520.0 KB` (`4->4`)** | `0.0334` | `1.00` (`1.00`) | `238.7` | **2.75%** (`11/400`, `[1.25, 4.50]`) | `4.0%` | `0.0%` | `2.0%` | `5.0%` | 多段学習重みの `L=1` 打ち切り基底 |
| **`smolvla_m7_deltaloop_L2`** (`Fixed L=2`, `s42`) | **`520.0 KB` (`4->4`)** | `0.0334` | `2.00` (`2.00`) | `239.7` | **4.00%** (`16/400`, `[2.25, 6.00]`) | `6.0%` | `1.0%` | `3.0%` | `6.0%` | **H4b vs L=1:** `+1.25%`, 両側 `p=0.1652` NS |
| **`smolvla_m7_deltaloop_adaptive`** ($k=0$, `s42`) | **`520.0 KB` (`4->4`)** | `0.0326±0.0008` | `1.11` (`1.12`) | `240.3` | **4.25%** (`17/400`, `[2.50, 6.25]`) | `6.0%` | `0.0%` | `2.0%` | `9.0%` | **H1a vs m0:** `p=0.0136` 未補正（Holm NS）；**H4d vs L=1:** `p=0.0696` NS（`p₁=0.0348`） |
| **`smolvla_m7_deltaloop_coda_conv`** (`tau=0.06`, `s42`) | **`520.0 KB` (`4->4`)** | `0.0334` | `2.28` (`2.65`) | `239.7` | **4.50%** (`18/400`, `[2.50, 6.50]`) | `5.0%` | `2.0%` | `3.0%` | `8.0%` | **H1d vs m0:** `p=0.0146` 未補正（Holm NS）；**H4e vs L=1:** `p=0.0654` NS（`p₁=0.0327`） |
| **`smolvla_m7_deltaloop_q16_adaptive` (`U7`)** | **`520.0 KB` (`16->16`)** | **`0.0315±0.0005`** | `1.42` (`1.95`) | **`237.3`** | **4.00%** (`16/400`, `[2.25, 6.00]`) | `2.0%` | `1.0%` | `5.0%` | `8.0%` | **H8a vs m0:** `p=0.0246` 未補正（Holm NS）；`VidPlOrd 3/25 (12%)` |
| **`smolvla_m7_deltaloop_anchor_adaptive` (`U8`)** | **`520.0 KB` (`4->4+4`)** | `0.0334` | `1.04` (`1.01`) | `239.6` | **3.00%** (`12/400`, `[1.50, 4.75]`) | `4.0%` | `1.0%` | `3.0%` | `4.0%` | **H9a:** `p=0.1626` NS；**Anchor-Shortcut (`96.1% L=1`)** 観察 |
| **`smolvla_m7_deltaloop_anchor_L2` (`U8 L=2`)** | **`520.0 KB` (`4->4+4`)** | `0.0334` | `2.00` (`2.00`) | `239.1` | **3.75%** (`15/400`, `[2.00, 5.75]`) | `5.0%` | `1.0%` | `3.0%` | `6.0%` | **H9f vs Anchor CRN:** `+0.75%`, `p=0.4832` NS；**H9g vs m0:** `p=0.0612` NS |
| **`smolvla_m2_framesamp_interfk3`** *($N=240, k=3$)* | `24.0 KB` (`32->32`) | `0.0533` | `1.00` (`1.00`) | `288.1` | **1.25%** (`3/240`, `[0.00, 2.92]`) | `1.7%` | `0.0%` | `3.3%` | `0.0%` | **H7b vs k=0 (`14/240`):** `-4.58%`, 両側 `p=0.0032` 未補正（`p_holm=0.1056` NS） |
| **`smolvla_m5_gdn2l_l1_interfk3`** *($N=240, k=3$)* | **`520.0 KB` (`4->4`)** | `0.0326` | `1.00` (`1.00`) | `234.6` | **0.83%** (`2/240`, `[0.00, 2.08]`) | `1.7%` | `0.0%` | `1.7%` | `0.0%` | **H7d vs k=0 (`12/240`):** `-4.17%`, 主検定 McNemar `p=0.0020` 未補正（`p_holm=0.0680` NS；boot `p<=0.0002`, `p_holm<=0.0074`）；**H7g DiD vs M2:** `p=0.9140` NS |
| **`smolvla_m7_deltaloop_adaptive_interfk3`** *($N=240, k=3$)* | **`520.0 KB` (`4->4`)** | `0.0334` | `1.09` (`1.07`) | `240.3` | **2.92%** (`7/240`, `[0.83, 5.00]`) | `6.7%` | `0.0%` | `5.0%` | `0.0%` | **H7c vs k=0 (`10/240`):** `p=0.4544` NS；**H7a vs M2:** `p=0.1950` NS；**H7f vs M5:** `p=0.0670` NS |

**Episode-Matched 3-Seed Closed-Loop Verification (`seeds = {42, 43, 44}`, Matched `eps 0..14`, `N = 240`/Seed, `Pooled N = 720`):**

| 3-Seed Contender (`k = 0`) | `seed=42` (`240`) | `seed=43` (`240`) | `seed=44` (`240`) | **3-Seed Mean ± SD** | Perm (`180`) | Count (`180`) | Ref (`180`) | Imit (`180`) | Pooled `N = 720` Paired Bootstrap (`Two-Sided p` & `Holm`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SmolVLA-M0` (`Base L=1`, `0 B`)** | `1.25%` (`3`) | `2.50%` (`6`) | `2.92%` (`7`) | **`2.22% ± 0.87%`** (`16/720`) | `3.33±0.00%` | `0.00±0.00%` | `2.22±1.92%` | `3.33±1.67%` | マルコフ基底 |
| **`SmolVLA-M2` (`FrameSamp-32`, `24 KB`)** | `5.83%` (`14`) | `6.67%` (`16`) | `10.42%` (`25`) | **`7.64% ± 2.44%`** (`55/720`) | `4.44±3.47%` | `0.56±0.96%` | `12.78±7.52%` | `12.78±4.19%` | **H_MS2 vs M0:** **`+5.42%`, `[+3.47, +7.50]%`, `p <= 0.0002`, `p_holm <= 0.0076`（Holm 有意）** |
| **`SmolVLA-M7` (`DeltaLoop CRN`, `520 KB`)** | `4.17%` (`10`) | `4.17%` (`10`) | `3.75%` (`9`) | **`4.03% ± 0.24%`** (`29/720`) | `4.44±0.96%` | `0.56±0.96%` | `4.44±1.92%` | `6.67±2.89%` | **H_MS1 vs M0:** `+1.81%`, `[+0.14, +3.61]%`, `p = 0.0414` 未補正（Holm NS）；**H_MS3 vs M2:** **`-3.61%`, `p = 0.0008`, `p_holm = 0.0288`（Holm 有意で `M2` 優位）** |

### 1.3 最終科学総括（C1–C8 に基づく厳密な結論）
1. **閉ループ利得の真の源泉は「持続的 2層 GDN 状態圧縮（`M5`）」であり、Coda ループ（`M7`）ではない：** 3つの全閉ループスイート（Cycle 2A `11.25% vs 7.50%`、Cycle 3A `12.50% vs 8.12%/6.25%`、Cycle 3B `5.25% vs 4.00–4.50%`）で単発 `M5` ($L=1$) が多段ループ変種を上回り、Holm 補正後も `M5 > M0` (`p=0.0012`, `p_holm=0.0420`) は有意に残る一方、`M7` の対 `M0` 優位は多重比較補正後に残らない。多段ループの実証利得は**合成 `Task 1A-Retro` `Forward` 連鎖**および**多段学習済み `M7` 重み内部での計算節約パレート（`CRN` `1.11` hops vs `Conv` `2.28` hops, `H5b: p=0.8990` NS；対 `L=1` 打ち切りは両側 `p=0.065–0.070` の非有意傾向）**に限定される。
2. **Cross-Regime Halting と `U8` Anchor-Shortcut の機構的知見：** 離散ポインタ跳躍（Cycle 1）では非単調ステップノルムにより L2 収束が破綻する一方、連続視覚運動 Coda 多様体（Cycle 3B）では減衰不動点収縮（`Bai et al., 2019`）として動作する。また役割分離 `U8`（`4+4`）はオフライン `aux_loss` / `ground_loss`・Probe 1 先鋭化（`-0.41 px`）で最良を示す一方、閉ループ `CRN` では `96.1% L=1` 停止（Anchor-Shortcut）を生じ、`Fixed L=2` との差（`15/400 vs 12/400`, `H9f: p=0.4832` NS）は床効果下の非有意な方向性観察に留まる。
3. **絶対成功率 `< 10%` 床効果と `k=3` セッション間妨害の厳密検証：** 凍結 `SmolVLM2` キャッシュ＋`100` demos/task 条件では全モデルが `1.5–10.4%` の床領域（公表 3B `π0.5` の `17.9% / 44.5%` より大幅に低い）にあり、小数カウント（`Perm 8/180`・`Count 1/180` の一致や 3シード標本 SD `0.24 pt`）は同等性や安定性の証明にはならない。`k=3` 非リセット妨害（約 `960` フレーム）では `FrameSamp-32` の劣化（`14/240 -> 3/240`, `H7b: p=0.0032` 未補正, `p_holm=0.1056` NS）に加え、**単発 `M5` もオフライン `cos(k=3)=0.992` にもかかわらず閉ループで低下（`12/240 [5.00%] -> 2/240 [0.83%]`, `H7d`: `10`脱落/`0`獲得のため正確両側 McNemar を主検定とし `p=0.0020` 未補正, `p_holm=0.0680` NS；有限標本 bootstrap 下限は `p<=0.0002`, `p_holm<=0.0074`；`H7g` DiD vs `M2` `p=0.9140` NS；長半減期ヘッド $t_{1/2}\le 512$ の残存重畳が単発 `Reference`/`Imitation` を攪乱）**した一方、多段学習 `M7` はモデル内低下が非有意（`10/240 -> 7/240`, `H7c: p=0.4544` NS）であるものの、**`k=3` 直接対決および DiD 検定（`H7a p=0.1950`, `H7f p=0.0670`, `H7h p=0.0976`, `H7i p=0.1238`, 全て `p_holm=1.0000`）は小標本床効果下でいずれも統計的に非有意**である。

---

## 2. 統制モデル体系（全 `7` アーキテクチャ・`14` チェックポイント）

| モデルID | Active Params (Stage-I Trainable) | 記憶状態 | Loop 幅 $\to$ Expert 幅 | Read 深さ $L$ | 役割・最終検証ステータス |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`smolvla_m0_base`** | `407.1M` (`104,218,851`) | `0 B` | — | $L=1$ | **事前学習マルコフ下限**（`s42: 6/400 [1.50%]`；**3-seed: `2.22±0.87%`**） |
| **`smolvla_m1_rdvla_loop4`** | `436.8M` (`133,979,676`) | `0 B` | `4 -> 4` | $L=4$ | **無記憶ループ統制 (`RD-VLA`)**（`10/400 [2.50%]`, `H0a p=0.3806` NS） |
| **`smolvla_m2_framesamp`** | `436.9M` (`134,072,524`) | `24.0 KB` | `32 -> 32` | $L=1$ | **外部バッファ主競合**（`s42: 6.50%` [**Holm 有意**], **3-seed: `7.64±2.44%`** $\to$ `k=3: 1.25%` [`p=0.0032` 未補正]） |
| **`smolvla_m5_gdn2l_l1`** | `444.0M` (`141,176,844`) | **`520.0 KB`** | `4 -> 4` | $L=1$ | **単発 2L GDN 統制**（**`21/400 [5.25%]`, `H1c p_holm=0.0420` 単一セッション最高 GDN**；`k=3: 2/240 [0.83%]`） |
| **`smolvla_m7_deltaloop`** | `444.2M` (`141,176,844` + `154K`) | **`520.0 KB`** | `4 -> 4` | `L=1,2` / `CRN` / `Conv` | **多段 Readout**（3-seed `4.03±0.24%` [`p=0.0414` 未補正, `H_MS3 < M2` Holm 有意]；同一ckpt `2.75->4.50%` [`p=0.0654` NS]；`k=3: 2.92%` [`p>=0.067` NS]） |
| **`smolvla_m7_deltaloop_q16` (`U7`)** | `444.2M` (`141,181,452` + `154K`) | **`520.0 KB`** | `16 -> 16` | 動的 $L_t$ | **均一広帯域アブレーション**（`16/400 [4.00%]`, `p=0.0246` 未補正；ZS `U4` `4.36 px`） |
| **`smolvla_m7_deltaloop_anchor` (`U8`)** | `444.2M` (`141,177,612` + `154K`) | **`520.0 KB`** | `4 -> 4+4=8` | 動的 $L_t$ / `L=2` | **役割分離 Anchor+Pointer**（Probe `-0.41 px`；`CRN 3.00%` vs `L=2 3.75%` [`H9f p=0.4832` NS]） |

---

## 3. 最終評価・監査対応ログ（ALL COMPLETED）

1. **COMPLETED — 同一重み深さ比較 (`L=1`, `L=2`, `CRN`, `coda_conv`)：** `2.75% (11/400) -> 4.00% (16/400) -> 4.25% (17/400) -> 4.50% (18/400)`（両側 `p = 0.1652 / 0.0696 / 0.0654` NS；単一セッションでは単発 `M5 = 5.25%` が優位）。
2. **COMPLETED — `U8` Anchor-Shortcut 診断 (`smolvla_m7_deltaloop_anchor_L2`, `N = 400`)：** `CRN` `96.1% L=1` 停止（`12/400 = 3.00%`）対 `Fixed L=2`（`15/400 = 3.75%`, 両側 `p = 0.4832` NS）。
3. **COMPLETED — 3-Seed `{42, 43, 44}` 閉ループ検証 (`Pooled N = 720`) ＆ Holm-Bonferroni (`m = 38`) 多重比較補正：** 無条件で Holm 有意となる4検定 `H_MS2` (`M2 > M0`, `p_holm <= 0.0076`), `H0b` (`p_holm = 0.0074`), `H_MS3` (`M7 < M2`, `p_holm = 0.0288`), `H1c` (`M5 > M0`, `p_holm = 0.0420`) を特定し、`H7d` (`M5 k=3 < k=0`, `10`脱落/`0`獲得) は主検定の正確両側 McNemar では `p = 0.0020`, `p_holm = 0.0680` で Holm 非有意（有限標本 bootstrap 条件付きでのみ `p <= 0.0002`, `p_holm <= 0.0074`）であることを厳密に報告。
4. **COMPLETED — `k = 3` クロスセッション妨害全3対照 (`M2`, `M5`, `M7`, 各 `N = 240`)：** `M2` (`14->3/240`, `p=0.0032` 未補正, `p_holm=0.1056` NS)・単発 `M5` (`12->2/240`, 主検定 McNemar `p=0.0020`, `p_holm=0.0680` NS; boot `p<=0.0002`, `p_holm<=0.0074`) の低下と、`M7` (`10->7/240`, `p=0.4544` NS; 対 `M2`/`M5` 直接差および DiD は `p=0.067–0.195` で全て非有意) を完全実証。
