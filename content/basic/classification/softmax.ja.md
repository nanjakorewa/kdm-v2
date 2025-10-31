---
title: "ソフトマックス回帰"
pre: "2.2.2 "
weight: 2
title_suffix: "多クラスの確率を同時に推定する"
---

{{% summary %}}
- ソフトマックス回帰はロジスティック回帰を多クラスへ拡張し、すべてのクラスの出現確率を同時に推定します。
- 出力は 0 以上 1 以下で総和が 1 になるため、しきい値設定やコスト計算にそのまま利用できます。
- 学習はクロスエントロピー損失を最小化することで行い、予測確率と真の分布のずれを直接補正します。
- scikit-learn では `LogisticRegression(multi_class="multinomial")` がソフトマックス回帰を実装し、L1/L2 正則化にも対応しています。
{{% /summary %}}

## 直感
二値分類ではシグモイド関数がクラス1の確率を返しますが、多クラス問題では「すべてのクラスの確率を同時に知りたい」ときが多くあります。ソフトマックス回帰はクラスごとの線形スコアを指数関数で変換し、それらを正規化して確率分布にします。スコアの高いクラスが強調され、低いクラスは抑えられます。

## 数式で見る
クラス数を \\(K\\)、クラス \\(k\\) の重みベクトルとバイアスをそれぞれ \\(\mathbf{w}_k\\)、\\(b_k\\) とすると

$$
P(y = k \mid \mathbf{x}) =
\frac{\exp\left(\mathbf{w}_k^\top \mathbf{x} + b_k\right)}
{\sum_{j=1}^{K} \exp\left(\mathbf{w}_j^\top \mathbf{x} + b_j\right)}
$$

で確率が得られます。目的関数はクロスエントロピー損失

$$
L = - \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}(y_i = k) \log P(y = k \mid \mathbf{x}_i)
$$

です。重みに正則化項を加えると過学習を抑えられます。

## Pythonによる実験
下記は3クラスの人工データにソフトマックス回帰を適用し、決定領域を描画した例です。`multi_class="multinomial"` を指定するとソフトマックス学習が有効になります。

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run_softmax_regression_demo(
    n_samples: int = 300,
    n_classes: int = 3,
    random_state: int = 42,
    label_title: str = "ソフトマックス回帰の決定領域",
    xlabel: str = "特徴量1",
    ylabel: str = "特徴量2",
) -> dict[str, float]:
    """Train a softmax regression model and visualise decision regions."""
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

    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf.fit(X, y)

    accuracy = float(accuracy_score(y, clf.predict(X)))

    x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    grid_x1, grid_x2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 400),
        np.linspace(x2_min, x2_max, 400),
    )
    grid_points = np.c_[grid_x1.ravel(), grid_x2.ravel()]
    preds = clf.predict(grid_points).reshape(grid_x1.shape)

    cmap = ListedColormap(["#ff9896", "#98df8a", "#aec7e8", "#f7b6d2", "#c5b0d5"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(
        grid_x1,
        grid_x2,
        preds,
        alpha=0.3,
        cmap=cmap,
        levels=np.arange(-0.5, n_classes + 0.5, 1),
    )
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(label_title)
    legend = ax.legend(*scatter.legend_elements(), title="class", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    plt.show()

    return {"accuracy": accuracy}


metrics = run_softmax_regression_demo(
    label_title="ソフトマックス回帰の決定領域",
    xlabel="特徴量1",
    ylabel="特徴量2",
)
print(f"訓練精度: {metrics['accuracy']:.3f}")

```


![ソフトマックス回帰の決定領域](/images/basic/classification/softmax_block01_ja.png)

## 参考文献
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
