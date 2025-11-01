---
title: "k-means Clustering | Iterative Centroid Partitioning"
linkTitle: "k-means"
seo_title: "k-means Clustering | Iterative Centroid Partitioning"
weight: 1
pre: "2.5.1 "
title_suffix: "Iteratively refine cluster centroids"
searchtitle: "k-means clustering illustrated in Python"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means follows an intuitive rule—group nearby points together—by repeatedly updating cluster representatives (centroids) until assignments stabilise.
- The objective function is the within-cluster sum of squares (WCSS), i.e. the squared distance between each sample and its assigned centroid.
- With `scikit-learn`’s `KMeans` you can visualise convergence, experiment with initialisation schemes, and inspect how assignments change.
- Choosing \\(k\\) typically involves diagnostics such as the elbow method or silhouette scores, balanced with domain knowledge.
{{% /summary %}}

## Intuition
k-means is a coordinate-descent style algorithm. After you choose the number of clusters \\(k\\), it alternates two simple steps:

1. Assign each sample to the closest centroid.
2. Recompute every centroid as the mean of the samples assigned to that cluster.

The assignments and centroids feed into one another; once neither step changes appreciably, the algorithm has converged. Because outliers and initialisation can sway the result, it is common to repeat k-means with several random seeds or better seeding strategies such as k-means++.

## Objective in mathematical form
Given data \\(\mathcal{X} = \{x_1, \dots, x_n\}\\) and clusters \\(\{C_1, \dots, C_k\}\\), k-means minimises

$$
\min_{C_1, \dots, C_k} \sum_{j=1}^k \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2,
$$

where \\(\mu_j = |C_j|^{-1} \sum_{x_i \in C_j} x_i\\) is the centroid of cluster \\(j\\). Intuitively, it finds centroids that minimise the average squared distance from points to their assigned “centre of mass”.

## Experimenting in Python
The following sections reproduce the Japanese walkthrough with English commentary and labels.

### 1. Generate data and inspect the initial placement
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
    """Create a synthetic dataset suitable for k-means demos."""
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
    """Pick deterministic initial centroids from points below a threshold."""
    candidates = data[data[:, 1] < threshold]
    if len(candidates) < n_clusters:
        raise ValueError("Not enough candidates for the requested centroids.")
    return candidates[:n_clusters]


def plot_initial_configuration(
    data: np.ndarray,
    centroids: np.ndarray,
    figsize: tuple[float, float] = (7.5, 7.5),
) -> None:
    """Show the raw dataset and highlight the initial centroids."""
    japanize_matplotlib.japanize()
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(data[:, 0], data[:, 1], c="#4b5563", marker="x", label="samples")
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="#ef4444",
        marker="o",
        s=80,
        label="initial centroids",
    )
    ax.set_title("Initial dataset and centroid seeds")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


DATASET_X, DATASET_Y = generate_dataset()
INITIAL_CENTROIDS = choose_initial_centroids(DATASET_X, n_clusters=4)
plot_initial_configuration(DATASET_X, INITIAL_CENTROIDS)
```


![Initial configuration](/images/basic/clustering/k-means1_block01_en.png)

### 2. Watch centroids converge
```python
from typing import Sequence

from sklearn.cluster import KMeans


def plot_kmeans_convergence(
    data: np.ndarray,
    init_centroids: np.ndarray,
    max_iters: Sequence[int] = (1, 2, 3, 10),
    random_state: int = 1,
) -> dict[int, float]:
    """Run k-means with different iteration caps and plot the outcomes."""
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
            label="centroids",
        )
        ax.set_title(f"max_iter = {max_iter}")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        inertia_by_iter[max_iter] = float(model.inertia_)

    fig.suptitle("Convergence behaviour versus iteration cap")
    fig.tight_layout()
    plt.show()
    return inertia_by_iter


CONVERGENCE_STATS = plot_kmeans_convergence(DATASET_X, INITIAL_CENTROIDS)
for iteration, inertia in CONVERGENCE_STATS.items():
    print(f"max_iter={iteration}: inertia={inertia:,.1f}")
```


![Convergence comparison](/images/basic/clustering/k-means1_block02_en.png)

### 3. Increase overlap and inspect assignments
```python
def plot_cluster_overlap_effect(
    base_random_state: int = 117_117,
    cluster_stds: Sequence[float] = (1.0, 2.0, 3.0, 4.5),
) -> None:
    """Show how heavier overlap complicates assignments."""
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

    fig.suptitle("Effect of cluster overlap on assignments")
    fig.tight_layout()
    plt.show()


plot_cluster_overlap_effect()
```


![Overlap comparison](/images/basic/clustering/k-means1_block03_en.png)

### 4. Compare diagnostics for choosing \\(k\\)
```python
from sklearn.metrics import silhouette_score


def analyse_cluster_counts(
    data: np.ndarray,
    k_range: Sequence[int] = range(2, 11),
) -> dict[str, list[float]]:
    """Evaluate WCSS and silhouette scores across cluster counts."""
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
    """Plot the elbow curve and silhouette scores."""
    japanize_matplotlib.japanize()
    ks = list(k_range)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, metrics["inertia"], marker="o")
    axes[0].set_title("Elbow method (WCSS)")
    axes[0].set_xlabel("cluster count k")
    axes[0].set_ylabel("WCSS")
    axes[0].grid(alpha=0.2)

    axes[1].plot(ks, metrics["silhouette"], marker="o", color="#ea580c")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("cluster count k")
    axes[1].set_ylabel("score")
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
print(f"Silhouette score peaks at k={best_k}")
```


![Diagnostics for k](/images/basic/clustering/k-means1_block04_en.png)

## References
{{% references %}}
<li>MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. <i>Proceedings of the Fifth Berkeley Symposium</i>.</li>
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. Retrieved from https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
