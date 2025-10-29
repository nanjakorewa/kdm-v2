---
title: "DBSCAN"
pre: "2.5.4 "
weight: 4
title_suffix: "密度ベースでクラスタを抽出する"
---

{{% summary %}}
- DBSCAN（Density-Based Spatial Clustering of Applications with Noise）は「密度が高い領域をクラスタ、疎な領域をノイズ」とみなすアルゴリズム。
- 2 つのハイパーパラメータ `eps`（半径）と `min_samples`（近傍の必要点数）で密度を規定し、コア・境界・ノイズの 3 種類の点に分類する。
- 距離ベースの k-means と異なりクラスタ数を事前に指定する必要がなく、非凸な形状やノイズを含むデータに強い。
- `min_samples`-距離プロット（k-distance graph）を用いると `eps` の当たりを付けやすく、前処理として標準化を行うと距離が安定する。
{{% /summary %}}

## 直感
DBSCAN では「点の周囲に十分な仲間が存在するか」を基準にクラスタを構成します。

- **コアポイント**: 半径 `eps` の近傍に `min_samples` 個以上の点が存在する。
- **境界ポイント**: コアポイントの近傍に含まれるが、自身は条件を満たさない。
- **ノイズ**: コアにも境界にも属さない孤立した点。

コアポイントは同じ密度領域に属する点をつなぎ、密度領域ごとにクラスタを形成します。クラスタ形状が不規則でも分離できる一方、密度差が大きい場合にはパラメータ調整が必要です。

## 具体的な数式
データ集合 \(\mathcal{X}\) の点 \(x_i\) に対し、`eps` の \(\varepsilon\)-近傍を

$$
\mathcal{N}_\varepsilon(x_i) = \{x_j \in \mathcal{X} \mid \lVert x_i - x_j \rVert \le \varepsilon \}
$$

と定義します。以下の条件を満たす点をコアポイントと呼びます。

$$
|\mathcal{N}_\varepsilon(x_i)| \ge \texttt{min\_samples}
$$

コアポイント同士がチェーン状につながると「密度連結」とみなされ、同じクラスタに属します。境界ポイントはコアポイントに密度到達可能（density reachable）な点であり、ノイズはそれ以外の点です。

## Pythonを用いた実験や説明
`scikit-learn` で月型データをクラスタリングし、ノイズ点の自動検出を確認します。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=600, noise=0.08, random_state=0)
X = StandardScaler().fit_transform(X)

model = DBSCAN(eps=0.3, min_samples=10)
labels = model.fit_predict(X)

print("クラスタ数 (ノイズ除く):", len(set(labels)) - (1 if -1 in labels else 0))
print("ノイズ点数:", np.sum(labels == -1))

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="Spectral", s=20)
plt.title("DBSCAN によるクラスタリング")
plt.xlabel("特徴量1")
plt.ylabel("特徴量2")
plt.tight_layout()
plt.show()
```

![dbscan block 1](/images/basic/clustering/dbscan_block01.svg)

## 参考文献
{{% references %}}
<li>Ester, M., Kriegel, H.-P., Sander, J., &amp; Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. <i>KDD</i>.</li>
<li>Schubert, E. et al. (2017). DBSCAN Revisited, Revisited. <i>ACM Transactions on Database Systems</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
