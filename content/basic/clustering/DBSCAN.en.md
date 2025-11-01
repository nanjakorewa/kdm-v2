---
title: "DBSCAN | Extracting density-based clusters"
linkTitle: "DBSCAN"
seo_title: "DBSCAN | Extracting density-based clusters"
pre: "2.5.4 "
weight: 4
title_suffix: "Extracting density-based clusters"
searchtitle: "DBSCAN clustering illustrated in Python"
---

{{% summary %}}
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) labels dense regions as clusters and treats sparse regions as noise.
- Two hyperparameters, `eps` (the search radius) and `min_samples` (the minimum neighbourhood size), classify points as core, border, or noise.
- Unlike k-means, DBSCAN does not require a preset cluster count and copes well with non-convex shapes and noisy data.
- A k-distance plot helps tune `eps`, and standardising features keeps distance measures comparable.
{{% /summary %}}

## Intuition
DBSCAN builds clusters from points that have enough neighbours within distance `eps`.

- **Core point**: has at least `min_samples` neighbours within the `eps` radius.
- **Border point**: lies within the neighbourhood of a core point but does not itself have enough neighbours.
- **Noise point**: belongs to neither category.

Core points that are density-reachable from one another form a cluster, while border points attach to their closest dense region. Any remaining isolated point is labelled as noise.

## Mathematics
For data \\(\mathcal{X}\\) and point \\(x_i\\), the \\(\varepsilon\\)-neighbourhood is

$$
\mathcal{N}_\varepsilon(x_i) = \{\, x_j \in \mathcal{X} \mid \lVert x_i - x_j \rVert \le \varepsilon \,\}.
$$

A point becomes core when

$$
|\mathcal{N}_\varepsilon(x_i)| \ge \texttt{min\_samples}.
$$

Clusters emerge by connecting core points that are density-reachable; border points join the nearest reachable cluster, and all others remain noise.

## Python walkthrough
We cluster a two-moons dataset (after standardisation) and report the number of clusters and noise points.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def run_dbscan_demo(
    n_samples: int = 600,
    noise: float = 0.08,
    eps: float = 0.3,
    min_samples: int = 10,
    random_state: int = 0,
) -> dict[str, int]:
    """Cluster two-moons data with DBSCAN and report cluster and noise counts."""
    features, _ = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )
    features = StandardScaler().fit_transform(features)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)

    unique_labels = sorted(np.unique(labels))
    cluster_ids = [label for label in unique_labels if label != -1]
    noise_count = int(np.sum(labels == -1))

    core_mask = np.zeros(labels.shape[0], dtype=bool)
    if hasattr(model, "core_sample_indices_"):
        core_mask[model.core_sample_indices_] = True

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    palette = plt.cm.get_cmap("tab10", max(len(cluster_ids), 1))

    for order, cluster_id in enumerate(cluster_ids):
        mask = labels == cluster_id
        colour = palette(order)
        if np.any(mask & core_mask):
            ax.scatter(
                features[mask & core_mask, 0],
                features[mask & core_mask, 1],
                c=[colour],
                s=36,
                edgecolor="white",
                linewidth=0.2,
                label=f"cluster {cluster_id} (core)",
            )
        if np.any(mask & ~core_mask):
            ax.scatter(
                features[mask & ~core_mask, 0],
                features[mask & ~core_mask, 1],
                c=[colour],
                s=24,
                edgecolor="white",
                linewidth=0.2,
                marker="o",
                label=f"cluster {cluster_id} (border)",
            )

    if noise_count:
        noise_mask = labels == -1
        ax.scatter(
            features[noise_mask, 0],
            features[noise_mask, 1],
            c="#9ca3af",
            marker="x",
            s=28,
            linewidth=0.8,
            label="noise",
        )

    ax.set_title("DBSCAN clustering")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    plt.show()

    return {"n_clusters": len(cluster_ids), "n_noise": noise_count}


result = run_dbscan_demo()
print(f"Number of clusters found: {result['n_clusters']}")
print(f"Number of noise points: {result['n_noise']}")
```


![DBSCAN clustering result](/images/basic/clustering/dbscan_block01_en.png)

## References
{{% references %}}
<li>Ester, M., Kriegel, H.-P., Sander, J., &amp; Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. <i>KDD</i>.</li>
<li>Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., &amp; Xu, X. (2017). DBSCAN Revisited, Revisited. <i>ACM Transactions on Database Systems</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
