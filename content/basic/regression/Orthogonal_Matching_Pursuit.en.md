---
title: "Orthogonal Matching Pursuit (OMP)"
pre: "2.1.12 "
weight: 12
title_suffix: "Greedy selection for sparse coefficients"
---

{{% summary %}}
- Orthogonal Matching Pursuit (OMP) selects the feature most correlated with the current residual at each step to build a sparse linear model.
- The algorithm solves a least-squares problem restricted to the selected features, keeping the coefficients interpretable.
- Instead of tuning a regularization strength, you directly control sparsity by specifying the number of features to keep.
- Standardizing features and checking for multicollinearity help the method remain stable when recovering sparse signals.
{{% /summary %}}

## Intuition
When only a handful of features truly matter, OMP adds the one that most reduces the residual error at each iteration. It is a cornerstone of dictionary learning and sparse coding, producing compact coefficient vectors that highlight the most relevant predictors.

## Mathematical formulation
Initialize the residual as \(\mathbf{r}^{(0)} = \mathbf{y}\). For each iteration \(t\):

1. Compute the inner product between every feature \(\mathbf{x}_j\) and the residual \(\mathbf{r}^{(t-1)}\), and pick the feature \(j\) with the largest absolute value.
2. Add \(j\) to the active set \(\mathcal{A}_t\).
3. Solve the least-squares problem restricted to \(\mathcal{A}_t\) to obtain \(\hat{\boldsymbol\beta}_{\mathcal{A}_t}\).
4. Update the residual \(\mathbf{r}^{(t)} = \mathbf{y} - \mathbf{X}_{\mathcal{A}_t} \hat{\boldsymbol\beta}_{\mathcal{A}_t}\).

Stop once you reach the desired sparsity or the residual is sufficiently small.

## Experiments with Python
The snippet below compares OMP and lasso on data with a sparse ground-truth coefficient vector.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error


def run_omp_vs_lasso(
    n_samples: int = 200,
    n_features: int = 40,
    sparsity: int = 4,
    noise_scale: float = 0.5,
    xlabel: str = "feature index",
    ylabel: str = "coefficient",
    label_true: str = "true",
    label_omp: str = "OMP",
    label_lasso: str = "Lasso",
    title: str | None = None,
) -> dict[str, object]:
    """Compare OMP and lasso on synthetic sparse regression data.

    Args:
        n_samples: Number of training samples to generate.
        n_features: Total number of features in the dictionary.
        sparsity: Count of non-zero coefficients in the ground truth.
        noise_scale: Standard deviation of Gaussian noise added to targets.
        xlabel: Label for the coefficient plot x-axis.
        ylabel: Label for the coefficient plot y-axis.
        label_true: Legend label for the ground-truth bars.
        label_omp: Legend label for the OMP bars.
        label_lasso: Legend label for the lasso bars.
        title: Optional title for the bar chart.

    Returns:
        Dictionary containing recovered supports and MSE values.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n_samples, n_features))
    true_coef = np.zeros(n_features)
    true_support = rng.choice(n_features, size=sparsity, replace=False)
    true_coef[true_support] = rng.normal(loc=0.0, scale=3.0, size=sparsity)
    y = X @ true_coef + rng.normal(scale=noise_scale, size=n_samples)

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
    omp.fit(X, y)
    lasso = Lasso(alpha=0.05)
    lasso.fit(X, y)

    omp_pred = omp.predict(X)
    lasso_pred = lasso.predict(X)

    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(n_features)
    ax.bar(indices - 0.3, true_coef, width=0.2, label=label_true, color="#2ca02c")
    ax.bar(indices, omp.coef_, width=0.2, label=label_omp, color="#1f77b4")
    ax.bar(indices + 0.3, lasso.coef_, width=0.2, label=label_lasso, color="#d62728")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "true_support": np.flatnonzero(true_coef),
        "omp_support": np.flatnonzero(omp.coef_),
        "lasso_support": np.flatnonzero(np.abs(lasso.coef_) > 1e-6),
        "omp_mse": float(mean_squared_error(y, omp_pred)),
        "lasso_mse": float(mean_squared_error(y, lasso_pred)),
    }



metrics = run_omp_vs_lasso()
print("True support:", metrics['true_support'])
print("OMP support:", metrics['omp_support'])
print("Lasso support:", metrics['lasso_support'])
print(f"OMP MSE: {metrics['omp_mse']:.4f}")
print(f"Lasso MSE: {metrics['lasso_mse']:.4f}")

```

### Reading the results
- Setting `n_nonzero_coefs` to the true sparsity helps OMP recover the relevant features reliably.
- Compared with lasso, OMP produces exactly zero coefficients for unselected features.
- When features are highly correlated, the selection order can fluctuate, so preprocessing and feature engineering remain important.

## References
{{% references %}}
<li>Pati, Y. C., Rezaiifar, R., &amp; Krishnaprasad, P. S. (1993). Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition. In <i>Conference Record of the Twenty-Seventh Asilomar Conference on Signals, Systems and Computers</i>.</li>
<li>Tropp, J. A. (2004). Greed is Good: Algorithmic Results for Sparse Approximation. <i>IEEE Transactions on Information Theory</i>, 50(10), 2231–2242.</li>
{{% /references %}}
