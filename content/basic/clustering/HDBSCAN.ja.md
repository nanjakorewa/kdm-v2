---
title: "HDBSCAN"
pre: "2.5.6 "
weight: 6
title_suffix: "密度に階層構造を導入する"
---

{{% summary %}}
- HDBSCAN（Hierarchical DBSCAN）は DBSCAN を階層化し、密度の異なる領域を自動的にクラスタ化するアルゴリズム。
- `min_cluster_size` と `min_samples` の 2 つで密度基準を制御し、ノイズ点（ラベル -1）を自然に切り離す。
- クラスタごとの「安定度」（persistence）を計算でき、信頼度の高いクラスタのみ採用する運用が可能。
- UMAP などの次元削減と組み合わせると、高次元データでも柔軟に利用できる。
{{% /summary %}}

## 直感
DBSCAN は 1 つの `eps`（半径）を用いて密度の高い領域を探しますが、データによっては適切な半径がクラスタごとに異なります。  
HDBSCAN は `eps` を徐々に広げながら密度が高い領域を追跡し、クラスタの階層構造を構築します。その後、クラスタの安定度を評価し、もっとも安定した部分クラスタを最終的な出力とします。

## 具体的な数式
HDBSCAN は「距離を密度に変換する」ために、各点のコア距離 \(d_\mathrm{core}(x)\) を計算します。

$$
d_\mathrm{core}(x) = \text{距離}(x, \text{min\_samples 番目に近い点})
$$

緩和距離（mutual reachability distance）は

$$
d_\mathrm{mreach}(x, y) = \max\{d_\mathrm{core}(x), d_\mathrm{core}(y), \lVert x - y \rVert\}
$$

で定義され、この距離に基づいて最小全域木を構築します。枝を切る閾値を変化させながら得られるクラスタの「滞在時間」を安定度として評価し、信頼度の高いクラスタを残します。

## Pythonを用いた実験や説明
`hdbscan` ライブラリで月型データをクラスタリングし、ノイズ点とクラスタ安定度を確認します。

```python
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=400, noise=0.08, random_state=42)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=10,
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(X)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=20)
plt.title("HDBSCAN によるクラスタリング")
plt.show()

print("クラスタ安定度:", clusterer.cluster_persistence_)
print("ノイズ点数:", np.sum(labels == -1))
```

## 参考文献
{{% references %}}
<li>Campello, R. J. G. B., Moulavi, D., Zimek, A., &amp; Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. <i>Pacific-Asia Conference on Knowledge Discovery and Data Mining</i>.</li>
<li>McInnes, L., Healy, J., &amp; Astels, S. (2017). hdbscan: Hierarchical Density Based Clustering. <i>Journal of Open Source Software</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
