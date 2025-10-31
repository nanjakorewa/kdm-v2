---
title: "線形判別分析 (LDA)"
pre: "2.2.4 "
weight: 4
title_suffix: "見分けやすい方向を学ぶ"
---

{{% summary %}}
- LDA はクラス間分散とクラス内分散の比を最大化する方向を求め、分類と次元圧縮の両方に使える。
- 決定境界は \\(\mathbf{w}^\top \mathbf{x} + b = 0\\) の形になるため、2 次元なら直線、3 次元なら平面として解釈できる。
- 各クラスが同一共分散を持つガウス分布だと仮定すると、ベイズ最適に近い分類器を構築できる。
- scikit-learn の `LinearDiscriminantAnalysis` を使えば、射影後の特徴量や決定境界の可視化を簡単に行える。
{{% /summary %}}

## 直感
LDA は「同じクラス同士を近づけ、異なるクラス同士を遠ざける」方向を探すアルゴリズムです。データをその方向に射影するとクラスが分離しやすくなり、そのまま分類に用いたり、低次元に圧縮してから別の分類器に渡したりできます。

## 具体的な数式
2 クラスの場合、射影方向 \\(\mathbf{w}\\) は

$$
J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}}
$$

を最大化することで求めます。ここで \\(\mathbf{S}_B\\) はクラス間散布行列、\\(\mathbf{S}_W\\) はクラス内散布行列です。多クラスでは最大でクラス数マイナス 1 個の射影方向が得られ、次元圧縮に利用できます。

## Pythonを用いた実験や説明
以下は 2 クラスの人工データに LDA を適用し、決定境界と射影方向を描画した例です。`transform` を使うと射影後の 1 次元特徴量も取得できます。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

rs = 42
X, y = make_blobs(
    n_samples=200, centers=2, n_features=2, cluster_std=2, random_state=rs
)

clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X, y)

w = clf.coef_[0]
b = clf.intercept_[0]

xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
ys_boundary = -(w[0] / w[1]) * xs - b / w[1]

scale = 3.0
origin = X.mean(axis=0)
arrow_end = origin + scale * (w / np.linalg.norm(w))

plt.figure(figsize=(7, 7))
plt.title("LDA の決定境界と射影方向", fontsize=16)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.8, label="サンプル")
plt.plot(xs, ys_boundary, "k--", lw=1.2, label=r"$\mathbf{w}^\top \mathbf{x} + b = 0$")
plt.arrow(
    origin[0],
    origin[1],
    (arrow_end - origin)[0],
    (arrow_end - origin)[1],
    head_width=0.5,
    length_includes_head=True,
    color="k",
    alpha=0.7,
    label="射影方向 $\mathbf{w}$",
)
plt.xlabel("特徴量 1")
plt.ylabel("特徴量 2")
plt.legend()
plt.xlim(X[:, 0].min() - 2, X[:, 0].max() + 2)
plt.ylim(X[:, 1].min() - 2, X[:, 1].max() + 2)
plt.grid(alpha=0.25)
plt.show()

X_1d = clf.transform(X)[:, 0]

plt.figure(figsize=(8, 4))
plt.title("LDA による 1 次元への射影結果", fontsize=15)
plt.hist(X_1d[y == 0], bins=20, alpha=0.7, label="クラス 0")
plt.hist(X_1d[y == 1], bins=20, alpha=0.7, label="クラス 1")
plt.xlabel("射影後の特徴量")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

![linear-discriminant-analysis block 2](/images/basic/classification/linear-discriminant-analysis_block02.svg)

## 参考文献
{{% references %}}
<li>Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic Problems. <i>Annals of Eugenics</i>, 7(2), 179–188.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
