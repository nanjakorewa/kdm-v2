---
title: "ガウス混合モデル (GMM)"
pre: "2.3.5 "
weight: 5
title_suffix: "確率的クラスタリングとソフトな所属"
---

{{< lead >}}
ガウス混合モデルは複数の正規分布を足し合わせてデータを表現する生成モデルです。クラスタ所属確率を出力でき、EM アルゴリズムでパラメータを推定します。
{{< /lead >}}

---

## 1. 数理モデル

確率密度は

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \; \mathcal{N}(\mathbf{x} \mid oldsymbol{\mu}_k, oldsymbol{\Sigma}_k)
$$

で表されます。\\(\pi_k\\) は混合係数で \\(\sum_k \pi_k = 1\\)。

---

## 2. EM アルゴリズム概要

1. **Eステップ**: 各データがクラスタ \\(k\\) に属する事後確率（責務）を計算
2. **Mステップ**: 責務を重みにして \\(\pi_k, oldsymbol{\mu}_k, oldsymbol{\Sigma}_k\\) を更新
3. 対数尤度が収束するまで繰り返し

---

## 3. Python 実装

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=600, centers=3, cluster_std=[1.0, 1.5, 0.8], random_state=7)

for cov_type in ["full", "tied", "diag"]:
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=0)
    gmm.fit(X)
    print(cov_type, "対数尤度:", gmm.score(X))

best = GaussianMixture(n_components=3, covariance_type="full", random_state=0).fit(X)
labels = best.predict(X)
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20)
plt.scatter(best.means_[:, 0], best.means_[:, 1], marker="x", color="red", s=100, label="centers")
plt.legend()
plt.tight_layout()
plt.show()
```

![ダミー図: GMM のクラスタリング結果](/images/placeholder_regression.png)

---

## 4. クラスタ数の決め方

- AIC/BIC でモデル選択
- 交差検証で対数尤度を比較
- 事前知識がある場合は固定

---

## 5. 長所と短所

| 長所 | 短所 |
| ---- | ---- |
| クラスタ所属確率を出力できる | 初期値や局所解に依存 |
| タイプ別に共分散の形を調整できる | 高次元だと共分散行列が不安定 |
| ベイズ的拡張（Dirichlet Process）も可能 | クラスタが球状でないと精度低下 |

---

## 6. まとめ

- GMM は「各クラスタ=ガウス分布」という仮定で柔軟なクラスタリングを実現
- EM アルゴリズムの収束監視と複数初期化で安定性を高める
- クラスタ所属確率を下流タスクで活用できるのが大きな利点です

---
