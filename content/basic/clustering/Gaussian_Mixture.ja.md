---
title: "ガウス混合モデル (GMM)"
pre: "2.5.5 "
weight: 5
title_suffix: "確率的クラスタリングとソフト割り当て"
---

{{% summary %}}
- ガウス混合モデルは複数の多変量正規分布を線形結合し、データ分布全体を表現する生成モデル。
- 各点のクラスタ所属確率を計算できるため「一番近いクラスタだけでなく、どれくらい所属しそうか」を出力できる。
- パラメータ推定には EM アルゴリズム（E-step と M-step）が用いられ、共分散行列の形状（`full`, `tied`, `diag`, `spherical`）を選べる。
- モデル選択には BIC/AIC や対数尤度を用い、初期化を複数回行うことで安定性が向上する。
{{% /summary %}}

## 直感
「データは複数のガウス分布が混ざったもの」と仮定すると、各クラスタは平均ベクトルと共分散行列を持つ楕円体として表現できます。  
k-means が「硬い割り当て（hard assignment）」を行うのに対し、GMM は点ごとに所属確率（責務）を出力する「ソフトクラスタリング」が可能です。

## 具体的な数式
入力ベクトル \(\mathbf{x}\) の確率密度は

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
$$

ここで \(\pi_k \ge 0\) は混合係数（\(\sum_k \pi_k = 1\)）、\(\boldsymbol{\mu}_k\) は平均ベクトル、\(\boldsymbol{\Sigma}_k\) は共分散行列です。  
EM アルゴリズムでは以下を収束まで繰り返します。

- **E-step**: クラスタ \(k\) がサンプル \(\mathbf{x}_i\) を生成した確率（責務）を計算。
  $$
  \gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
  {\sum_{j=1}^K \pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
  $$
- **M-step**: 責務を重みにして \(\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\) を更新。

この反復で対数尤度が単調に増加し、局所最大に収束します。

## Pythonを用いた実験や説明
混合分布から生成したデータを `GaussianMixture` で学習し、共分散の扱いによる差やクラスタリング結果を可視化します。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(
    n_samples=600,
    centers=3,
    cluster_std=[1.0, 1.5, 0.8],
    random_state=7,
)

for cov_type in ["full", "tied", "diag"]:
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=0)
    gmm.fit(X)
    print(cov_type, "対数尤度:", gmm.score(X))

best = GaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(X)
labels = best.predict(X)
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20)
plt.scatter(best.means_[:, 0], best.means_[:, 1], marker="x", color="red", s=100, label="平均")
plt.legend()
plt.tight_layout()
plt.show()
```

![gaussian-mixture block 1](/images/basic/clustering/gaussian-mixture_block01.svg)

## 参考文献
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Dempster, A. P., Laird, N. M., &amp; Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. <i>Journal of the Royal Statistical Society, Series B</i>.</li>
<li>scikit-learn developers. (2024). <i>Gaussian Mixture Models</i>. https://scikit-learn.org/stable/modules/mixture.html</li>
{{% /references %}}
