---
title: "t-SNE"
pre: "2.4.5 "
weight: 5
title_suffix: "高次元データを可視化する定番手法"
---

{{< lead >}}
t-SNE（t-Distributed Stochastic Neighbor Embedding）は高次元データの局所構造を 2 次元や 3 次元に可視化するための手法です。クラスタの分離具合や潜在構造を見る探索的分析に最適です。
{{< /lead >}}

---

## 1. 仕組みの概要

- 高次元空間で近傍確率 \\(P_{ij}\\) を構築
- 低次元空間の点でも同様に \\(Q_{ij}\\) を定義
- Kullback-Leibler ダイバージェンス \\(\mathrm{KL}(P \parallel Q)\\) を最小化するよう座標を更新
- 対称化した確率と Student-t 分布を使うことでクラスタ間を離す

---

## 2. Python 実装

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)

model = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
emb = model.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=15)
plt.colorbar(label="digit")
plt.title("t-SNE による手書き数字可視化")
plt.tight_layout()
plt.show()
```

![tsne block 1](/images/basic/dimensionality-reduction/tsne_block01.svg)

---

## 3. ハイパーパラメータ

- `perplexity`: 有効近傍数に相当。データ数の 5〜50 程度で探索
- `learning_rate`: 低すぎると収束しない、高すぎると構造が崩れる
- `n_iter`: 1000 以上を推奨。`early_exaggeration` でクラスタを離しやすくする

---

## 4. 注意点

- 座標はランダム初期化に依存し、絶対位置に意味はない
- 新しい点を既存埋め込みに追加するのは難しい（`openTSNE` などが必要）
- 距離の大小を厳密に解釈してはいけない（近傍関係を見るためのツール）

---

## 5. まとめ

- t-SNE は探索的データ分析やレポート用の可視化に強力
- パラメータ感度が高いため複数設定で比較し、再現性を確保する
- UMAP など後継手法も合わせて検討すると洞察が広がります

---
