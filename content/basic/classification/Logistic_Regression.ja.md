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
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run_logistic_regression_demo(
    n_samples: int = 300,
    random_state: int = 2,
    label_class0: str = "クラス0",
    label_class1: str = "クラス1",
    label_boundary: str = "決定境界",
    title: str = "ロジスティック回帰による決定境界",
) -> dict[str, float]:
    """Train logistic regression on a synthetic 2D dataset and visualise the boundary.

    Args:
        n_samples: Number of samples to generate.
        random_state: Seed for reproducible sampling.
        label_class0: Legend label for class 0.
        label_class1: Legend label for class 1.
        label_boundary: Legend label for the separating line.
        title: Title for the plot.

    Returns:
        Dictionary containing training accuracy and coefficients.
    """
    japanize_matplotlib.japanize()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1,
    )

    clf = LogisticRegression()
    clf.fit(X, y)

    accuracy = float(accuracy_score(y, clf.predict(X)))
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    x1, x2 = X[:, 0], X[:, 1]
    grid_x1, grid_x2 = np.meshgrid(
        np.linspace(x1.min() - 1.0, x1.max() + 1.0, 200),
        np.linspace(x2.min() - 1.0, x2.max() + 1.0, 200),
    )
    grid = np.c_[grid_x1.ravel(), grid_x2.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(grid_x1.shape)

    cmap = ListedColormap(["#aec7e8", "#ffbb78"])
    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(grid_x1, grid_x2, probs, levels=20, cmap=cmap, alpha=0.4)
    ax.contour(grid_x1, grid_x2, probs, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(x1[y == 0], x2[y == 0], marker="o", edgecolor="k", label=label_class0)
    ax.scatter(x1[y == 1], x2[y == 1], marker="x", color="k", label=label_class1)
    ax.set_xlabel("特徴量1")
    ax.set_ylabel("特徴量2")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.colorbar(contour, ax=ax, label="P(class = 1)")
    fig.tight_layout()
    plt.show()

    return {
        "accuracy": accuracy,
        "coef_0": float(coef[0]),
        "coef_1": float(coef[1]),
        "intercept": intercept,
    }


metrics = run_logistic_regression_demo(
    label_class0="クラス0",
    label_class1="クラス1",
    label_boundary="決定境界",
    title="ロジスティック回帰による決定境界",
)
print(f"訓練精度: {metrics['accuracy']:.3f}")
print(f"特徴量1の係数: {metrics['coef_0']:.3f}")
print(f"特徴量2の係数: {metrics['coef_1']:.3f}")
print(f"切片: {metrics['intercept']:.3f}")

```


![logistic-regression demo](/images/basic/classification/logistic-regression_block01_ja.png)

## 参考文献
{{% references %}}
<li>Agresti, A. (2015). <i>Foundations of Linear and Generalized Linear Models</i>. Wiley.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
