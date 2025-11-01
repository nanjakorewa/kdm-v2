---
title: "X-means | Automatically estimating the number of clusters"
linkTitle: "X-means"
seo_title: "X-means | Automatically estimating the number of clusters"
weight: 3
pre: "2.5.3 "
title_suffix: "Automatically estimating the number of clusters"
searchtitle: "X-means clustering with Python"
---

{{< katex />}}
{{% youtube "2hkyJcWctUA" %}}

{{% summary %}}
- X-means extends k-means by choosing well-separated seeds and splitting clusters only when a model-selection criterion says it is worthwhile.
- Each candidate split compares Bayesian Information Criterion (BIC) values; only splits that raise the score are accepted.
- A lightweight implementation can be built entirely on top of scikit-learn by combining k-means with a small BIC utility.
- The method is handy when you cannot guess \\(k\\) beforehand: the algorithm grows the number of clusters only when the data justify it.
{{% /summary %}}

## Intuition
Plain k-means requires a user-specified \\(k\\). If \\(k\\) is too small, clusters merge; if too large, clusters fragment. X-means tackles this by starting with a modest \\(k\\), running k-means, and then attempting to split each cluster. For every candidate split we compare BIC scores: if the split explains the data better, we keep it; otherwise we leave the cluster untouched. Iterating this procedure yields the number of clusters that best balances fit and complexity.

## Mathematical reminder
For clusters \\(\{C_1, \dots, C_k\}\\) with centres \\(\{\mu_1, \dots, \mu_k\}\\), the BIC is

$$
\mathrm{BIC} = \ln L - \tfrac{p}{2} \ln n,
$$

where \\(L\\) is the model likelihood, \\(p\\) the number of free parameters, and \\(n\\) the number of samples. When a cluster is split into two, we compute the BIC for both models and keep whichever yields the higher value.

## Python walkthrough
Below we implement a compact X-means-style splitter and compare it with a fixed \\(k=4\\) k-means run.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def generate_dataset(
    n_samples: int = 1200,
    n_centers: int = 9,
    cluster_std: float = 1.1,
    random_state: int = 1707,
) -> NDArray[np.float64]:
    """Generate synthetic blobs for clustering experiments."""
    features, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return features


def compute_bic(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    centers: NDArray[np.float64],
) -> float:
    """Compute a BIC score tailored to k-means clusters."""
    n_samples, n_features = data.shape
    n_clusters = centers.shape[0]
    if n_clusters <= 0:
        raise ValueError("Number of clusters must be positive.")

    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    sse = 0.0
    for idx, center in enumerate(centers):
        if cluster_sizes[idx] == 0:
            continue
        diffs = data[labels == idx] - center
        sse += float((diffs**2).sum())

    variance = sse / max(n_samples - n_clusters, 1)
    if variance <= 0:
        variance = 1e-6

    log_likelihood = 0.0
    norm_const = -0.5 * n_features * np.log(2 * np.pi * variance)
    for size in cluster_sizes:
        if size > 0:
            log_likelihood += size * norm_const - 0.5 * (size - 1)

    n_params = n_clusters * (n_features + 1)
    bic = log_likelihood - 0.5 * n_params * np.log(n_samples)
    return float(bic)


def xmeans_split(
    data: NDArray[np.float64],
    k_max: int = 20,
    random_state: int = 42,
) -> NDArray[np.int64]:
    """Approximate X-means by recursively splitting clusters when BIC improves."""
    rng = np.random.default_rng(random_state)
    pending = [np.arange(len(data))]
    accepted: list[NDArray[np.int64]] = []

    while pending:
        indices = pending.pop()
        subset = data[indices]

        if len(indices) <= 2:
            accepted.append(indices)
            continue

        parent_labels = np.zeros(len(indices), dtype=int)
        parent_center = subset.mean(axis=0, keepdims=True)
        bic_parent = compute_bic(subset, parent_labels, parent_center)

        split_model = KMeans(
            n_clusters=2,
            n_init=10,
            max_iter=200,
            random_state=rng.integers(0, 1_000_000),
        ).fit(subset)
        bic_children = compute_bic(
            subset,
            split_model.labels_,
            split_model.cluster_centers_,
        )

        if bic_children > bic_parent and (len(accepted) + len(pending) + 1) < k_max:
            pending.append(indices[split_model.labels_ == 0])
            pending.append(indices[split_model.labels_ == 1])
        else:
            accepted.append(indices)

    labels = np.empty(len(data), dtype=int)
    for cluster_id, cluster_indices in enumerate(accepted):
        labels[cluster_indices] = cluster_id
    return labels


def run_xmeans_demo(
    random_state: int = 2024,
    initial_k: int = 4,
    k_max: int = 18,
) -> dict[str, object]:
    """Compare vanilla k-means and the simple X-means splitter."""
    data = generate_dataset(random_state=random_state)

    kmeans_labels = KMeans(
        n_clusters=initial_k,
        n_init=10,
        random_state=random_state,
    ).fit_predict(data)
    xmeans_labels = xmeans_split(data, k_max=k_max, random_state=random_state + 99)

    unique_xmeans = np.unique(xmeans_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(data[:, 0], data[:, 1], c=kmeans_labels, cmap="tab20", s=10)
    axes[0].set_title(f"k-means (k={initial_k})")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(data[:, 0], data[:, 1], c=xmeans_labels, cmap="tab20", s=10)
    axes[1].set_title(f"X-means (k={len(unique_xmeans)})")
    axes[1].grid(alpha=0.2)

    fig.suptitle("k-means versus X-means cluster assignments")
    fig.tight_layout()
    plt.show()

    return {
        "kmeans_clusters": int(initial_k),
        "xmeans_clusters": int(len(unique_xmeans)),
        "cluster_sizes": np.bincount(xmeans_labels).tolist(),
    }


metrics = run_xmeans_demo()
print(f"Clusters requested by k-means: {metrics['kmeans_clusters']}")
print(f"Clusters discovered by X-means: {metrics['xmeans_clusters']}")
print(f"X-means cluster sizes: {metrics['cluster_sizes']}")
```


![k-means vs X-means](/images/basic/clustering/x-means_block01_en.png)

## References
{{% references %}}
<li>Pelleg, D., &amp; Moore, A. W. (2000). X-means: Extending k-means with Efficient Estimation of the Number of Clusters. <i>ICML</i>.</li>
<li>Bahmani, B., Moseley, B., Vattani, A., Kumar, R., &amp; Vassilvitskii, S. (2012). Scalable k-means++. <i>VLDB</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
