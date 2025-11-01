---
title: "ナイーブベイズ | 条件付き独立性で高速に推論する"
linkTitle: "ナイーブベイズ"
seo_title: "ナイーブベイズ | 条件付き独立性で高速に推論する"
pre: "2.2.6 "
weight: 6
title_suffix: "条件付き独立性で高速に推論する"
---

{{% summary %}}
- ナイーブベイズは特徴量が条件付き独立であると仮定し、ベイズの定理で尤度と事前確率を組み合わせて分類します。
- 学習も推論も高速で、テキスト分類やスパム検知のような高次元・疎なデータの強力なベースラインになります。
- ラプラス平滑化や TF-IDF 特徴量と組み合わせると未知語や頻度差に頑健になります。
- 独立性の仮定が強過ぎる場合は、特徴選択や他モデルとのアンサンブルを検討します。
{{% /summary %}}

## 直感
ベイズの定理は「事前確率 × 尤度 ∝ 事後確率」という関係を与えます。特徴量が条件付き独立であれば、尤度は各特徴の確率の積で近似できます。ナイーブベイズはこの近似を用いて、少量の学習データでも安定した分類が可能です。

## 数式で見る
クラス \\(y\\) と特徴ベクトル \\(\mathbf{x} = (x_1, \ldots, x_d)\\) に対して

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y)
$$

と近似します。テキスト分類では出現回数を扱う多項ナイーブベイズ、出現/非出現を扱うベルヌーイ型、連続値にはガウス型など、課題に合わせて尤度の形を選べます。

## Pythonによる実験
以下は人工データにガウシアンナイーブベイズを適用し、決定領域と混同行列を確認する例です。高次元でも高速に学習でき、シンプルな評価指標で性能を把握できます。

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB


def run_naive_bayes_demo(
    n_samples: int = 600,
    n_classes: int = 3,
    random_state: int = 0,
    title: str = "ガウシアン Naive Bayes の決定領域",
    xlabel: str = "特徴量1",
    ylabel: str = "特徴量2",
) -> dict[str, float | np.ndarray]:
    """Train Gaussian Naive Bayes on synthetic data and plot decision regions."""
    japanize_matplotlib.japanize()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state,
    )

    clf = GaussianNB()
    clf.fit(X, y)

    accuracy = float(accuracy_score(y, clf.predict(X)))
    conf = confusion_matrix(y, clf.predict(X))

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    preds = clf.predict(grid).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(
        grid_x,
        grid_y,
        preds,
        alpha=0.25,
        cmap="coolwarm",
        levels=np.arange(-0.5, n_classes + 0.5, 1),
    )
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()

    return {"accuracy": accuracy, "confusion": conf}


metrics = run_naive_bayes_demo(
    title="ガウシアン Naive Bayes の決定領域",
    xlabel="特徴量1",
    ylabel="特徴量2",
)
print(f"訓練精度: {metrics['accuracy']:.3f}")
print("混同行列:")
print(metrics['confusion'])

```


![ガウシアン Naive Bayes の決定領域](/images/basic/classification/naive-bayes_block01_ja.png)

## 参考文献
{{% references %}}
<li>Manning, C. D., Raghavan, P., &amp; Schütze, H. (2008). <i>Introduction to Information Retrieval</i>. Cambridge University Press.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
