---
title: "k-means"
weight: 1
pre: "2.5.1 "
title_suffix: "クラスタの“重心”を反復的に洗い出す"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means は「距離が近い点同士をまとめる」という素朴な直感をもとに、クラスタの代表点（セントロイド）を更新しながらデータを \\(k\\) 個に分割します。
- 目的関数は各サンプルと所属クラスタのセントロイドとの距離二乗和（WCSS）で、これを最小にする配置を探します。
- `scikit-learn` の `KMeans` を使えば、初期値や反復回数を変えつつ収束過程やクラスタ割り当ての変化を可視化できます。
- クラスタ数 \\(k\\) の選択にはエルボー法やシルエット係数などの指標が使われ、データ構造とビジネス要件を踏まえて判断します。
{{% /summary %}}

## 直感
k-means はセントロイドと呼ばれる代表点を動かしながら、次の手順を収束するまで繰り返します。

1. クラスタ数 \\(k\\) を決め、初期セントロイドを設定する。
2. 各サンプルを「最も近いセントロイド」に割り当てる。
3. 割り当てられたサンプルの平均を取り、新しいセントロイドを計算する。
4. セントロイドの移動が十分に小さくなるまで 2 E3 を繰り返す。

セントロイドはクラスタの“重心”と解釈できますが、初期値や外れ値に敏感なので複数回初期化したり前処理を丁寧に行ったりすることが大切です。

## 数式で理解する
データ集合 \\(\mathcal{X} = \{x_1, \dots, x_n\}\\) を \\(k\\) 個のクラスタ \\(\{C_1, \dots, C_k\}\\) に分けるとき、k-means は次の目的関数を最小化します。

$$
\min_{C_1, \dots, C_k} \sum_{j=1}^k \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2,
$$

ここで \\(\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i\\) はクラスタ \\(j\\) のセントロイドです。つまり「各サンプルから重心までの距離二乗をできるだけ小さくする」配置を探していることになります。

## Pythonで確かめる
以下では `scikit-learn` を使って収束の様子やクラスタ数の選び方を可視化します。

### 1. データ生成と初期配置の確認
```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def generate_dataset(
    n_samples: int = 1000,
    random_state: int = 117_117,
    cluster_std: float = 1.5,
    n_centers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset for k-means demonstrations.

    Args:
        n_samples: Number of samples to draw.
        random_state: Seed for reproducibility.
        cluster_std: Standard deviation of each blob.
        n_centers: Number of latent blobs to create.

    Returns:
        Tuple containing features and their latent labels.
    """
    return make_blobs(
        n_samples=n_samples,
        random_state=random_state,
        cluster_std=cluster_std,
        centers=n_centers,
    )


def choose_initial_centroids(
    data: np.ndarray,
    n_clusters: int,
    threshold: float = -8.0,
) -> np.ndarray:
    """Pick deterministic initial centroids from the lower region of the space.

    Args:
        data: Feature matrix.
        n_clusters: Number of centroids to select.
        threshold: Y-axis threshold used to filter candidates.

    Returns:
        Array of initial centroid locations.
    """
    mask = data[:, 1] < threshold
    candidates = data[mask]
    if len(candidates) < n_clusters:
        raise ValueError("Not enough candidates below the threshold.")
    return candidates[:n_clusters]


def plot_initial_configuration(
    data: np.ndarray,
    centroids: np.ndarray,
    figsize: tuple[float, float] = (7.5, 7.5),
) -> None:
    """Visualise the raw dataset and highlight the chosen initial centroids."""
    japanize_matplotlib.japanize()
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(data[:, 0], data[:, 1], c="#4b5563", marker="x", label="データ点")
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="#ef4444",
        marker="o",
        s=80,
        label="初期セントロイド",
    )
    ax.set_title("初期データとセントロイドの配置")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


DATASET_X, DATASET_Y = generate_dataset()
INITIAL_CENTROIDS = choose_initial_centroids(DATASET_X, n_clusters=4)
plot_initial_configuration(DATASET_X, INITIAL_CENTROIDS)
```


![初期配置の可視化](/images/basic/clustering/k-means1_block01_ja.png)

### 2. 反復回数と収束の様子
```python
from typing import Sequence

from sklearn.cluster import KMeans


def plot_kmeans_convergence(
    data: np.ndarray,
    init_centroids: np.ndarray,
    max_iters: Sequence[int] = (1, 2, 3, 10),
    random_state: int = 1,
) -> dict[int, float]:
    """Run k-means with varying max_iter and plot the centroids after each run.

    Args:
        data: Feature matrix.
        init_centroids: Initial centroids reused across runs.
        max_iters: Sequence of iteration counts to visualise.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary mapping max_iter to the resulting inertia value.
    """
    japanize_matplotlib.japanize()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    inertia_by_iter: dict[int, float] = {}

    for ax, max_iter in zip(axes.ravel(), max_iters, strict=False):
        model = KMeans(
            n_clusters=len(init_centroids),
            init=init_centroids,
            max_iter=max_iter,
            n_init=1,
            random_state=random_state,
        )
        model.fit(data)
        labels = model.predict(data)
        ax.scatter(data[:, 0], data[:, 1], c=labels, cmap="tab10", s=10)
        ax.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            c="#dc2626",
            marker="o",
            s=80,
            label="セントロイド",
        )
        ax.set_title(f"max_iter = {max_iter}")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        inertia_by_iter[max_iter] = float(model.inertia_)

    fig.suptitle("反復回数によるセントロイドの収束")
    fig.tight_layout()
    plt.show()
    return inertia_by_iter


CONVERGENCE_STATS = plot_kmeans_convergence(DATASET_X, INITIAL_CENTROIDS)
for iteration, inertia in CONVERGENCE_STATS.items():
    print(f"max_iter={iteration}: inertia={inertia:,.1f}")
```


![収束の比較](/images/basic/clustering/k-means1_block02_ja.png)

### 3. クラスタの重なり具合と割り当て
```python
def plot_cluster_overlap_effect(
    base_random_state: int = 117_117,
    cluster_stds: Sequence[float] = (1.0, 2.0, 3.0, 4.5),
) -> None:
    """Show how increased overlap makes assignments harder.

    Args:
        base_random_state: Seed used for blob generation.
        cluster_stds: Standard deviations tested for the blobs.
    """
    japanize_matplotlib.japanize()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for ax, std in zip(axes.ravel(), cluster_stds, strict=False):
        features, _ = make_blobs(
            n_samples=1_000,
            random_state=base_random_state,
            cluster_std=std,
        )
        assignments = KMeans(n_clusters=2, random_state=base_random_state).fit_predict(
            features
        )
        ax.scatter(features[:, 0], features[:, 1], c=assignments, cmap="tab10", s=10)
        ax.set_title(f"cluster_std = {std}")
        ax.grid(alpha=0.2)

    fig.suptitle("クラスタの重なり具合と割り当て")
    fig.tight_layout()
    plt.show()


plot_cluster_overlap_effect()
```


![重なり具合の比較](/images/basic/clustering/k-means1_block03_ja.png)

### 4. \\(k\\) の選択を指標で比較する
```python
from sklearn.metrics import silhouette_score


def analyse_cluster_counts(
    data: np.ndarray,
    k_range: Sequence[int] = range(2, 11),
) -> dict[str, list[float]]:
    """Compute WCSS and silhouette scores for different cluster counts.

    Args:
        data: Feature matrix.
        k_range: Iterable of cluster counts to evaluate.

    Returns:
        Dictionary containing the evaluated metrics.
    """
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=117_117).fit(data)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(data, model.labels_)))

    return {"inertia": inertias, "silhouette": silhouettes}


def plot_cluster_count_metrics(
    metrics: dict[str, list[float]],
    k_range: Sequence[int],
) -> None:
    """Plot the elbow curve and silhouette scores side by side."""
    japanize_matplotlib.japanize()
    ks = list(k_range)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, metrics["inertia"], marker="o")
    axes[0].set_title("エルボー法 (WCSS)")
    axes[0].set_xlabel("クラスタ数 k")
    axes[0].set_ylabel("WCSS")
    axes[0].grid(alpha=0.2)

    axes[1].plot(ks, metrics["silhouette"], marker="o", color="#ea580c")
    axes[1].set_title("シルエット係数")
    axes[1].set_xlabel("クラスタ数 k")
    axes[1].set_ylabel("スコア")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    plt.show()


ELBOW_METRICS = analyse_cluster_counts(DATASET_X, range(2, 11))
plot_cluster_count_metrics(ELBOW_METRICS, range(2, 11))

best_k = int(
    range(2, 11)[
        max(
            range(len(ELBOW_METRICS["silhouette"])),
            key=ELBOW_METRICS["silhouette"].__getitem__,
        )
    ]
)
print(f"シルエット係数が最大となるクラスタ数: k={best_k}")
```


![エルボー法とシルエット係数](/images/basic/clustering/k-means1_block04_ja.png)

## 参考文献
{{% references %}}
<li>MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. <i>Proceedings of the Fifth Berkeley Symposium</i>.</li>
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
