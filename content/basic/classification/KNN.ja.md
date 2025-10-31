---
title: "k近傍法 (k-NN)"
pre: "2.2.7 "
weight: 7
title_suffix: "距離に基づく怠惰学習"
---

{{% summary %}}
- k-NN は学習時にモデルを構築せず、推論時に近傍サンプルの多数決でラベルを決めるシンプルな手法です。
- 主なハイパーパラメータは近傍数 \\(k\\) と距離の重み付けで、探索が比較的容易です。
- 非線形な決定境界を自然に表現できますが、高次元では距離の差が縮み「次元の呪い」が課題になります。
- 前処理として標準化や特徴選択を行うと距離計算が安定し、性能が向上しやすくなります。
{{% /summary %}}

## 直感
「近くのサンプルは同じラベルを持つ」という仮定のもと、k-NN は予測したい点から最も近い \\(k\\) 個の訓練サンプルを探し、多数決（または距離に応じた重み付き投票）でラベルを決めます。あらかじめパラメータを学習しないため、しばしば「怠惰学習（lazy learning）」と呼ばれます。

## 数式による定義
テスト点 \\(\mathbf{x}\\) に対して \\(\mathcal{N}_k(\mathbf{x})\\) を訓練データのうち距離が近い \\(k\\) 個の集合とすると、クラス \\(c\\) に対する票数は

$$
v_c = \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \,\mathbb{1}(y_i = c)
$$

で表されます。ここで \\(w_i\\) は各サンプルの重み（例：距離の逆数）です。最も票の多いクラスが予測ラベルとなります。

## Pythonによる実験
以下のコードは複数の近傍数を検証用データで評価し、最も良かったモデルの決定領域を描画します。

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_knn_demo(
    n_samples: int = 600,
    random_state: int = 7,
    weights: str = "distance",
    k_values: tuple[int, ...] = (1, 3, 5, 7, 11),
    validation_ratio: float = 0.3,
    title: str = "k-NN の判別領域",
    xlabel: str = "特徴量1",
    ylabel: str = "特徴量2",
    class_label_prefix: str = "クラス",
) -> dict[str, object]:
    """Evaluate k-NN for several neighbour counts and plot decision regions.

    Args:
        n_samples: Number of synthetic samples to draw.
        random_state: Seed for reproducible sampling.
        weights: Weighting scheme handed to KNeighborsClassifier.
        k_values: Candidate neighbour counts to evaluate.
        validation_ratio: Fraction of the data reserved for validation.
        title: Title for the generated figure.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        class_label_prefix: Prefix used when labelling the classes.

    Returns:
        Dictionary with validation scores per k and the best-performing k.
    """
    japanize_matplotlib.japanize()
    X, y = make_blobs(
        n_samples=n_samples,
        centers=3,
        cluster_std=[1.1, 1.0, 1.2],
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1.0 - validation_ratio))
    train_idx, valid_idx = indices[:split], indices[split:]
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    scores: dict[int, float] = {}
    for k in k_values:
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k, weights=weights),
        )
        model.fit(X_train, y_train)
        scores[k] = float(model.score(X_valid, y_valid))

    best_k = max(scores, key=scores.get)
    best_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=best_k, weights=weights),
    )
    best_model.fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1.5, X[:, 0].max() + 1.5, 300),
        np.linspace(X[:, 1].min() - 1.5, X[:, 1].max() + 1.5, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    predictions = best_model.predict(grid).reshape(xx.shape)

    unique_classes = np.unique(y)
    levels = np.arange(unique_classes.min(), unique_classes.max() + 2) - 0.5
    cmap = ListedColormap(["#fee0d2", "#deebf7", "#c7e9c0"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    contour = ax.contourf(xx, yy, predictions, levels=levels, cmap=cmap, alpha=0.85)
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="Set1",
        edgecolor="#1f2937",
        linewidth=0.6,
    )
    ax.set_title(f"{title} (k={best_k}, weights={weights})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(alpha=0.15)

    legend = ax.legend(
        handles=scatter.legend_elements()[0],
        labels=[f"{class_label_prefix} {cls}" for cls in unique_classes],
        loc="upper right",
        frameon=True,
    )
    legend.get_frame().set_alpha(0.9)
    fig.colorbar(contour, ax=ax, label="予測クラス")
    fig.tight_layout()
    plt.show()

    return {"scores": scores, "best_k": int(best_k), "validation_accuracy": scores[best_k]}


metrics = run_knn_demo(
    title="k-NN の判別領域",
    xlabel="特徴量1",
    ylabel="特徴量2",
    class_label_prefix="クラス",
)
print(f"最良の k: {metrics['best_k']}")
print(f"検証精度 (最良の k): {metrics['validation_accuracy']:.3f}")
for candidate_k, score in metrics["scores"].items():
    print(f"k={candidate_k}: 検証精度={score:.3f}")

```


![k-NN の判別領域](/images/basic/classification/knn_block01_ja.png)

## 参考文献
{{% references %}}
<li>Cover, T. M., &amp; Hart, P. E. (1967). Nearest Neighbor Pattern Classification. <i>IEEE Transactions on Information Theory</i>, 13(1), 21 E7.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
