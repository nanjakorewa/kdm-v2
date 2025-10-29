---
title: "階層型クラスタリング"
pre: "2.5.7 "
weight: 7
title_suffix: "デンドログラムで構造を読み解く"
---

{{% summary %}}
- 階層型クラスタリングはサンプルを段階的に結合・分割しながらクラスタ構造を可視化する手法で、デンドログラムを用いてクラスタ数を柔軟に決められる。
- 凝集型（Agglomerative）の場合は各サンプルを 1 クラスタから開始し、リンケージ基準に従って距離が近いクラスタ同士を結合する。
- `linkage`（Ward, complete, average, single 等）と `affinity`（距離指標）を組み合わせることで、クラスタ間距離の定義を変えられる。
- `AgglomerativeClustering` や `scipy.cluster.hierarchy` を利用すると、木構造の生成とクラスタ割り当てを容易に実装できる。
{{% /summary %}}

## 直感
階層型クラスタリングは「似ているものからまとめていく」または「大きなクラスタを分割していく」ことで、クラスタの親子関係を構築します。  
デンドログラム（樹形図）で距離の大きいところを切ると自然なクラスタ数を選べるため、クラスタ数を事前に決めたくない場合に便利です。凝集型（一般的な手法）では以下を繰り返します。

1. 各サンプルを独立したクラスタとして初期化。
2. 現在最も近い 2 クラスタを結合。
3. 距離が閾値を超えるまで、またはクラスタ数が指定数になるまで繰り返す。

## 具体的な数式
クラスタ間距離はリンケージによって定義が異なります。

- **Ward**: クラスタ内平方和の増加量を最小化する。
- **Complete**: 2 クラスタ間の最大距離。
- **Average**: 2 クラスタ間の平均距離。
- **Single**: 2 クラスタ間の最小距離。

例えば Ward 法では、クラスタ \(C_i, C_j\) を結合したときのコスト増分 \(\Delta\) を

$$
\Delta = \frac{|C_i||C_j|}{|C_i| + |C_j|} \lVert \boldsymbol{\mu}_i - \boldsymbol{\mu}_j \rVert^2
$$

で評価します（\(\boldsymbol{\mu}_i\) はクラスタの重心）。このような距離関数を用いてクラスタを結合し、デンドログラムを構築します。

## Pythonを用いた実験や説明
`scipy` でデンドログラムを描画し、`AgglomerativeClustering` で実際のラベルを得る例を示します。

```python
import numpy as np
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.7, random_state=42)

Z = linkage(X, method="ward")  # クラスタ内平方和が最小になるよう結合

plt.figure(figsize=(8, 4))
dendrogram(Z, truncate_mode="level", p=5)
plt.title("階層型クラスタリング（Ward 法）")
plt.xlabel("サンプル")
plt.ylabel("距離")
plt.show()
```

![hierarchical-clustering block 1](/images/basic/clustering/hierarchical-clustering_block01.svg)

```python
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(
    n_clusters=3,
    linkage="average",
    affinity="euclidean",
)
labels = clustering.fit_predict(X)
```

`distance_threshold` を指定する場合は `n_clusters` と排他なので、必要な方だけ設定します。

## 参考文献
{{% references %}}
<li>Ward, J. H. (1963). Hierarchical Grouping to Optimize an Objective Function. <i>Journal of the American Statistical Association</i>.</li>
<li>Müllner, D. (2011). Modern Hierarchical, Agglomerative Clustering Algorithms. <i>arXiv:1109.2378</i>.</li>
<li>scikit-learn developers. (2024). <i>Hierarchical clustering</i>. https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering</li>
{{% /references %}}
