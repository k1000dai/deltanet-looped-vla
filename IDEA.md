メモリを一つではなく、時間スケール別に分けます。
\[
S_t^{\mathrm{short}},\quad
S_t^{\mathrm{mid}},\quad
S_t^{\mathrm{long}}
\]

例えば、

- Short Memory：直近1〜5秒の動き
- Mid Memory：直近数分の操作履歴
- Long Memory：数十分にわたる物体位置やタスク進捗
それぞれ異なる更新・忘却率を学習させます。
\[
S_t^{(i)}
=
\operatorname{GatedDelta}
\left(S_{t-1}^{(i)},z_t\right)
\]

これによって、短期的な物体追跡と長期的なタスク記憶を分離できる可能性があります。

Gated DeltaNetを用いた学習.?

### Experiment 1：Synthetic Multi-timescale Memory Benchmark

例えば、3種類の情報を入力系列に埋め込みます。

- Short：数十ステップ前の動き
- Mid：数百ステップ前の操作
- Long：数千ステップ前に変化した物体状態

時間が経過した後に、それぞれの情報を問い合わせます。

比較するモデルは以下です。

1. Single Gated DeltaNet
2. 3× Gated DeltaNet（同一忘却率）
3. 3× Gated DeltaNet（異なる固定忘却率）
4. 3× Gated DeltaNet（学習可能な忘却率）
5. 提案モデル（学習可能な忘却率＋Consolidation Gate）

この比較なら、複数のメモリを持つ効果と、学習可能な時間スケールの効果、階層間転送の効果を分離して評価できます。

IDEA

- Gated Delta Net x Looped Transformer x Robot Foundation MOdel(VLA

| 論文 | GDN | Loop | Robotics |
| --- | --- | --- | --- |
| RoboTTT | 比較対象 | — | ✓ |
| Dual-Stream Memory | ✓ | — | ✓ |
| RD-VLA | — | ✓ | ✓ |
| Looped Actor | — | ✓ | ✓ |
| LT2 | ✓ | ✓ | — |
| MixerLoop | ✓ | ✓ | — |
| 考えている提案 | ✓ | ✓ | ✓ |

「長期記憶の内容や不確実性に応じて、Loop回数とメモリの読み出し方を同時に決定する」

例えば、ロボットが現在の画像だけで判断できる状況なら1回のLoopで十分です。

しかし、過去に見た物体が遮蔽されている状況では、

1. GDNから過去の状態を読み出す
2. 現在の観測と比較する
3. 別のQueryでもう一度メモリを読む
4. 必要ならさらに推論する

という処理を行います。

数式では、

$$
h_t^{(l+1)}
F_\theta

\left(

h_t^{(l)},

\operatorname{Read}(S_t,q_t^{(l)})

\right)

]
[

q_t^{(l)}=W_qh_t^{(l)}
$$

さらに、停止確率を

$$
[
p_{\mathrm{stop}}^{(l)}
=
\sigma\left(f(h_t^{(l)},m_t^{(l)})\right)
]
$$

として学習させます。

GDNの長期記憶をLoopごとに再検索することで、同じ計算予算の通常VLAやRD-VLAより、遮蔽・物体追跡・長期タスクの成功率を改善できるか？

# 検証タスク

robomme : https://github.com/RoboMME/robomme_benchmark

baseline : robomme のpi05
