---
title: "スペクトラルクラスタリング"
pre: "2.5.8 "
weight: 8
title_suffix: "グラフ固有ベクトルで分割する"
---

{{% summary %}}
- スペクトラルクラスタリングはデータを類似度グラフに変換し、グラフラプラシアンの固有ベクトルを使って低次元空間に埋め込んだ後に k-means などでクラスタ分割する。
- 非凸クラスタやグラフ構造を持つデータにも対応でき、円環や同心円など k-means が苦手な形状をうまく分離できる。
- ハイパーパラメータとして類似度の計算方法（`affinity`）、スケール（`gamma` や `n_neighbors`）、クラスタ割り当て手法（`assign_labels`）を調整する。
- 固有ベクトルの次元はクラスタ数と同じだけ抽出するため、計算コストやメモリ消費に気をつける必要がある。
{{% /summary %}}

## 直感
データ点をノード、類似度（距離の小ささやカーネル）をエッジ重みとするグラフを考えると、密に繋がった部分同士がクラスタ候補になります。  
グラフラプラシアンの低次固有ベクトルは、クラスタ間の切断コストを最小化する方向を示すため、これを座標として並べ替え、最後に通常のクラスタリングを適用すれば複雑な構造も分離できます。

## 具体的な数式
類似度行列を \(W\)、次数行列を \(D = \mathrm{diag}(d_1, \dots, d_n)\)（\(d_i = \sum_j W_{ij}\)）とすると、非正規化ラプラシアンは

$$
L = D - W.
$$

ノーマライズ版は
$$
L_\mathrm{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}.
$$

クラスタ数 \(k\) に対して、\(L_\mathrm{sym}\) の下位 \(k\) 個の固有ベクトルを列に持つ行列 \(U\) を作り、行単位で正規化した行列を k-means などで分割します。  
これはグラフの切断問題（Normalized Cut）を緩和した解に相当します。

## Pythonを用いた実験や説明
二重円（同心円）データを `SpectralClustering` で分割し、非線形構造でもクラスタリングできることを確認します。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_circles
from sklearn.cluster import SpectralClustering

X, _ = make_circles(n_samples=500, factor=0.4, noise=0.05, random_state=0)

model = SpectralClustering(
    n_clusters=2,
    affinity="rbf",
    gamma=50,
    assign_labels="kmeans",
    random_state=0,
)
labels = model.fit_predict(X)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm", s=20)
plt.title("スペクトラルクラスタリング")
plt.tight_layout()
plt.show()
```

![spectral-clustering block 1](/images/basic/clustering/spectral-clustering_block01.svg)

## 参考文献
{{% references %}}
<li>Shi, J., &amp; Malik, J. (2000). Normalized Cuts and Image Segmentation. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>.</li>
<li>Ng, A. Y., Jordan, M. I., &amp; Weiss, Y. (2002). On Spectral Clustering: Analysis and an Algorithm. <i>Advances in Neural Information Processing Systems</i>.</li>
<li>scikit-learn developers. (2024). <i>Spectral clustering</i>. https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering</li>
{{% /references %}}
