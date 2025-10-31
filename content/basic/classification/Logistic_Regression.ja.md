---
title: "ロジスティック回帰"
pre: "2.2.1 "
weight: 1
title_suffix: "シグモイドで確率を推定する"
---

{{% summary %}}
- ロジスティック回帰は入力の線形結合をシグモイド関数に通し、クラス 1 である確率を直接推定する二値分類モデル。
- 出力が \\([0, 1]\\) の確率なので意思決定のしきい値を柔軟に設定でき、係数は対数オッズ比として解釈しやすい。
- 学習はクロスエントロピー損失（対数尤度の最大化）で行い、L1/L2 正則化を組み合わせると過学習を抑えられる。
- scikit-learn の `LogisticRegression` を使えば、前処理から決定境界の可視化まで一貫して実装できる。
{{% /summary %}}

## 直感
線形回帰の出力は実数全域に広がりますが、分類では「クラス 1 である確率」が欲しいことが多くあります。ロジスティック回帰は線形結合 \\(z = \mathbf{w}^\top \mathbf{x} + b\\) をシグモイド関数 \\(\sigma(z) = 1 / (1 + e^{-z})\\) に通し、確率として解釈できる値を得ます。確率が 0.5 を超えたらクラス 1 と判定する、といったルールを決めるだけで分類が可能です。

## 具体的な数式
入力 \\(\mathbf{x}\\) に対するクラス 1 の確率は

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + \exp\left(-(\mathbf{w}^\top \mathbf{x} + b)\right)}
$$

で表されます。学習では対数尤度

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^{n} \Bigl[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Bigr], \quad p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b)
$$

を最大化（すなわち負のクロスエントロピー損失を最小化）します。L2 正則化を加えると係数が大きく振れるのを防ぎ、L1 正則化を加えると不要な特徴量を自動で 0 にできます。

## Pythonを用いた実験や説明
以下は人工的に作成した 2 次元データにロジスティック回帰を適用し、決定境界を可視化した例です。`LogisticRegression` を利用するだけで学習・予測・境界の描画まで完結します。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 2 次元の分類データを生成
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=1,
    random_state=2,
    n_clusters_per_class=1,
)

# モデルを学習
clf = LogisticRegression()
clf.fit(X, y)

# 決定境界を算出
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
slope = -w1 / w2
intercept = -b / w2

xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
xd = np.array([xmin, xmax])
yd = slope * xd + intercept

# 可視化
plt.figure(figsize=(8, 8))
plt.plot(xd, yd, "k-", lw=1, label="決定境界")
plt.scatter(*X[y == 0].T, marker="o", label="クラス 0")
plt.scatter(*X[y == 1].T, marker="x", label="クラス 1")
plt.legend()
plt.title("ロジスティック回帰による分類境界")
plt.show()
```

![logistic-regression block 2](/images/basic/classification/logistic-regression_block02.svg)

## 参考文献
{{% references %}}
<li>Agresti, A. (2015). <i>Foundations of Linear and Generalized Linear Models</i>. Wiley.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
