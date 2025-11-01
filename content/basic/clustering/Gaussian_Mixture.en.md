---
title: "Gaussian Mixture Models (GMM)"
pre: "2.5.5 "
weight: 5
title_suffix: "Soft clustering with probabilistic assignments"
searchtitle: "Gaussian mixture clustering in Python"
---

{{% summary %}}
- A Gaussian Mixture Model represents data as a weighted sum of multivariate normal components.
- It outputs a responsibility matrix that quantifies how strongly each component explains every sample.
- Parameters are estimated with the EM algorithm; covariance structures can be `full`, `tied`, `diag`, or `spherical`.
- Model selection typically combines information criteria (BIC/AIC) with multiple random initialisations for stability.
{{% /summary %}}

## Intuition
Assume the data arise from \\(K\\) Gaussian sources. Each component contributes a mean vector and covariance matrix, forming elliptical clusters. Unlike k-means, which makes a hard decision, GMMs provide soft assignments: for every sample \\(x_i\\) and component \\(k\\) we obtain \\(\gamma_{ik}\\), the probability that component \\(k\\) generated \\(x_i\\).

## Mathematics
The density of \\(\mathbf{x}\\) is

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
$$

with mixture weights \\(\pi_k\\) (non-negative and summing to 1). EM alternates:

- **E-step**: compute responsibilities \\(\gamma_{ik}\\).
  $$
  \gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
  {\sum_{j=1}^K \pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.
  $$
- **M-step**: re-estimate \\(\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\\) using \\(\gamma_{ik}\\) as weights.

The log-likelihood increases monotonically and converges to a local optimum.

## Python walkthrough
We fit a GMM to synthetic 2D blobs, plot the hard assignments, and report mixture weights and the responsibility matrix shape.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


def run_gmm_demo(
    n_samples: int = 600,
    n_components: int = 3,
    cluster_std: list[float] | tuple[float, ...] = (1.0, 1.4, 0.8),
    covariance_type: str = "full",
    random_state: int = 7,
    n_init: int = 8,
) -> dict[str, object]:
    """Fit a Gaussian mixture model and visualise hard labels with component centres."""
    features, _ = make_blobs(
        n_samples=n_samples,
        centers=n_components,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    )
    gmm.fit(features)

    hard_labels = gmm.predict(features)
    responsibilities = gmm.predict_proba(features)
    log_likelihood = float(gmm.score(features))
    weights = gmm.weights_

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    scatter = ax.scatter(
        features[:, 0],
        features[:, 1],
        c=hard_labels,
        cmap="viridis",
        s=30,
        edgecolor="white",
        linewidth=0.2,
        alpha=0.85,
    )
    ax.scatter(
        gmm.means_[:, 0],
        gmm.means_[:, 1],
        marker="x",
        c="red",
        s=140,
        linewidth=2.0,
        label="Component centre",
    )
    ax.set_title("Gaussian mixture clustering (hard labels shown)")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.grid(alpha=0.2)
    handles, _ = scatter.legend_elements()
    labels = [f"cluster {idx}" for idx in range(n_components)]
    ax.legend(handles, labels, title="predicted label", loc="upper right")
    fig.tight_layout()
    plt.show()

    return {
        "log_likelihood": log_likelihood,
        "weights": weights.tolist(),
        "responsibilities_shape": responsibilities.shape,
    }


metrics = run_gmm_demo()
print(f"log-likelihood: {metrics['log_likelihood']:.3f}")
print("mixture weights:", metrics["weights"])
print("responsibility matrix shape:", metrics["responsibilities_shape"])
```


![Gaussian mixture clustering result](/images/basic/clustering/gaussian-mixture_block01_en.png)

## References
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Dempster, A. P., Laird, N. M., &amp; Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. <i>Journal of the Royal Statistical Society, Series B</i>.</li>
<li>scikit-learn developers. (2024). <i>Gaussian Mixture Models</i>. https://scikit-learn.org/stable/modules/mixture.html</li>
{{% /references %}}
