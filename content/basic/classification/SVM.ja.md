---
title: "サポートベクターマシン (SVM) | マージン最大化で汎化性能を高める"
linkTitle: "サポートベクターマシン (SVM)"
seo_title: "サポートベクターマシン (SVM) | マージン最大化で汎化性能を高める"
pre: "2.2.5 "
weight: 5
title_suffix: "マージン最大化で汎化性能を高める"
---

{{% summary %}}
- SVM はクラス間のマージンを最大化する決定境界を学習し、汎化性能を重視した分類器を構築します。
- ソフトマージンではスラック変数を導入して誤分類を許容し、罰則係数 \\(C\\) でマージン幅とのバランスを制御します。
- カーネルトリックを使えば内積をカーネル関数に置き換え、明示的に特徴量を増やさなくても非線形境界を扱えます。
- 前処理としての標準化と、\\(C\\) や \\(\gamma\\) といったハイパーパラメータ探索が性能向上の鍵になります。
{{% /summary %}}

## 直感
分離超平面が複数存在するとき、SVM はサンプルから最も離れたマージンが最大のものを選びます。マージンに接するサンプルはサポートベクターと呼ばれ、彼らだけが最終的な境界を決定します。その結果、多少のノイズに強い滑らかな決定境界になります。

## 数式で見る
線形に分離可能な場合は次の最適化問題を解きます。

$$
\min_{\mathbf{w}, b} \ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1.
$$

実際のデータでは完全に分離できないことが多いため、スラック変数 \\(\xi_i \ge 0\\) を導入したソフトマージン SVM を用います。

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}}
\ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 + C \sum_{i=1}^{n} \xi_i
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i.
$$

内積 \\(\mathbf{x}_i^\top \mathbf{x}_j\\) をカーネル \\(K(\mathbf{x}_i, \mathbf{x}_j)\\) に置き換えれば、非線形な決定境界も表現できます。

## Pythonによる実験
次のコードは `make_moons` で生成した非線形データに線形カーネル SVM と RBF カーネル SVM を適用し、決定境界を比較する例です。RBF カーネルの方が曲がった境界を適切に表現できることがわかります。

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def run_svm_demo(
    n_samples: int = 400,
    noise: float = 0.25,
    random_state: int = 42,
    title: str = "RBF カーネル SVM の決定境界",
    xlabel: str = "特徴量1",
    ylabel: str = "特徴量2",
) -> dict[str, float]:
    """Train linear and RBF SVMs and plot the RBF decision boundary."""
    japanize_matplotlib.japanize()
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    linear_clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
    linear_clf.fit(X, y)

    rbf_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5.0, gamma=0.5))
    rbf_clf.fit(X, y)

    linear_acc = float(accuracy_score(y, linear_clf.predict(X)))
    rbf_acc = float(accuracy_score(y, rbf_clf.predict(X)))

    grid_x, grid_y = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 400),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 400),
    )
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    rbf_scores = rbf_clf.predict(grid).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(grid_x, grid_y, rbf_scores, alpha=0.2, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()

    return {"linear_accuracy": linear_acc, "rbf_accuracy": rbf_acc}


metrics = run_svm_demo(
    title="RBF カーネル SVM の決定境界",
    xlabel="特徴量1",
    ylabel="特徴量2",
)
print(f"線形カーネルの精度: {metrics['linear_accuracy']:.3f}")
print(f"RBF カーネルの精度: {metrics['rbf_accuracy']:.3f}")

```


![RBF カーネル SVM の決定境界](/images/basic/classification/svm_block01_ja.png)

## 参考文献
{{% references %}}
<li>Vapnik, V. (1998). <i>Statistical Learning Theory</i>. Wiley.</li>
<li>Smola, A. J., &amp; Schölkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199 E22.</li>
{{% /references %}}
