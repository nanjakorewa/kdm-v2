---
title: "k-means++ | Smarter Seeding Strategies for k-means Clustering"
linkTitle: "k-means++"
seo_title: "k-means++ | Smarter Seeding Strategies for k-means Clustering"
weight: 2
pre: "2.5.2 "
title_suffix: "Smarter centroid seeding for k-means"
searchtitle: "Understanding k-means++ initialisation in Python"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means++ spreads the initial centroids apart, reducing the chance that vanilla k-means converges to a poor local optimum.
- Additional centroids are sampled with probability proportional to the squared distance from the existing centroids, discouraging tight clusters of seeds.
- In `scikit-learn`, `KMeans(init="k-means++")` activates the method, making it easy to compare with purely random initialisation.
- Large-scale variants such as mini-batch k-means build on k-means++ and are common in streaming or big-data settings.
{{% /summary %}}

## Intuition
k-means is notoriously sensitive to the initial centroids. If they all fall inside a single dense region, the algorithm may converge quickly but produce a clustering with large WCSS. k-means++ mitigates this by sampling the first centroid uniformly and subsequent ones with higher probability the farther they are from the already selected centroids.

## Mathematical formulation
Given the set of chosen centroids \\(\{\mu_1, \dots, \mu_m\}\\), a candidate point \\(x\\) is selected as the next centroid with probability

$$
P(x) = \frac{D(x)^2}{\sum_{x' \in \mathcal{X}} D(x')^2}, \qquad
D(x) = \min_{1 \le j \le m} \lVert x - \mu_j \rVert.
$$

Points that are far from every existing centroid (large \\(D(x)\\)) therefore receive higher probability mass, yielding a well-scattered initial configuration (Arthur & Vassilvitskii, 2007).

## Experiment in Python
The snippet below compares random initialisation and k-means++ across three trials, collecting the WCSS after a single k-means iteration.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def create_blobs_dataset(
    n_samples: int = 3000,
    n_centers: int = 8,
    cluster_std: float = 1.5,
    random_state: int = 11711,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for initialisation experiments."""
    return make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )


def compare_initialisation_strategies(
    data: np.ndarray,
    n_clusters: int = 5,
    subset_size: int = 1000,
    n_trials: int = 3,
    random_state: int = 11711,
) -> dict[str, list[float]]:
    """Compare random versus k-means++ seeding and plot the assignments.

    Args:
        data: Feature matrix to cluster.
        n_clusters: Number of clusters to form.
        subset_size: Number of samples drawn per trial.
        n_trials: Number of comparisons to perform.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing WCSS values for both initialisation modes.
    """
    rng = np.random.default_rng(random_state)
    inertia_random: list[float] = []
    inertia_kpp: list[float] = []

    fig, axes = plt.subplots(
        n_trials,
        2,
        figsize=(10, 3.2 * n_trials),
        sharex=True,
        sharey=True,
    )

    for trial in range(n_trials):
        indices = rng.choice(len(data), size=subset_size, replace=False)
        subset = data[indices]

        random_model = KMeans(
            n_clusters=n_clusters,
            init="random",
            n_init=1,
            max_iter=1,
            random_state=random_state + trial,
        ).fit(subset)
        kpp_model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=1,
            random_state=random_state + trial,
        ).fit(subset)

        inertia_random.append(float(random_model.inertia_))
        inertia_kpp.append(float(kpp_model.inertia_))

        ax_random = axes[trial, 0] if n_trials > 1 else axes[0]
        ax_kpp = axes[trial, 1] if n_trials > 1 else axes[1]

        ax_random.scatter(subset[:, 0], subset[:, 1], c=random_model.labels_, s=10)
        ax_random.set_title(f"Random init (trial {trial + 1})")
        ax_random.grid(alpha=0.2)

        ax_kpp.scatter(subset[:, 0], subset[:, 1], c=kpp_model.labels_, s=10)
        ax_kpp.set_title(f"k-means++ init (trial {trial + 1})")
        ax_kpp.grid(alpha=0.2)

    fig.suptitle("Label assignments after one iteration (random vs k-means++)")
    fig.tight_layout()
    plt.show()

    return {"random": inertia_random, "k-means++": inertia_kpp}


FEATURES, _ = create_blobs_dataset()
results = compare_initialisation_strategies(
    data=FEATURES,
    n_clusters=5,
    subset_size=1000,
    n_trials=3,
    random_state=2024,
)
for method, values in results.items():
    print(f"{method} mean WCSS: {np.mean(values):.1f}")
```


![Initialisation comparison](/images/basic/clustering/k-means2_block01_en.png)

## References
{{% references %}}
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>Bahmani, B., Moseley, B., Vattani, A., Kumar, R., &amp; Vassilvitskii, S. (2012). Scalable k-means++. <i>VLDB</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. Retrieved from https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
