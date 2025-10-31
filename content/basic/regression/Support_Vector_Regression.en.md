---
title: "Support Vector Regression (SVR)"
pre: "2.1.10 "
weight: 10
title_suffix: "Robust predictions with an ε-insensitive tube"
---

{{% summary %}}
- Support Vector Regression extends SVMs to regression, treating errors within an ε-insensitive tube as zero to reduce outlier impact.
- Kernel methods enable flexible non-linear relationships while keeping the model compact via support vectors.
- Hyperparameters `C`, `epsilon`, and `gamma` govern the balance between generalization and smoothness.
- Feature scaling is essential; wrapping preprocessing and learning in a pipeline ensures consistent transformations.
{{% /summary %}}

## Intuition
SVR fits a function surrounded by an ε-wide tube: points inside the tube incur no loss, whereas those outside pay a penalty. Only the points touching or leaving the tube—the support vectors—affect the final model. This yields a smooth approximation that resists noisy observations.

## Mathematical formulation
The optimization problem is

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
$$

subject to

$$
\begin{aligned}
y_i - (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) &\le \epsilon + \xi_i, \\
(\mathbf{w}^\top \phi(\mathbf{x}_i) + b) - y_i &\le \epsilon + \xi_i^*, \\
\xi_i, \xi_i^* &\ge 0,
\end{aligned}
$$

where \(\phi\) maps inputs into a feature space via the chosen kernel. Solving the dual yields the support vectors and coefficients.

## Experiments with Python
This example demonstrates SVR combined with `StandardScaler` in a pipeline.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def run_svr_demo(
    *,
    n_samples: int = 300,
    train_size: float = 0.75,
    xlabel: str = "input x",
    ylabel: str = "output y",
    label_train: str = "train samples",
    label_test: str = "test samples",
    label_pred: str = "SVR prediction",
    label_truth: str = "ground truth",
    title: str | None = None,
) -> dict[str, float]:
    """Train SVR on synthetic nonlinear data, plot fit, and report metrics.

    Args:
        n_samples: Number of data points sampled from the underlying function.
        train_size: Fraction of data used for training.
        xlabel: X-axis label for the plot.
        ylabel: Y-axis label for the plot.
        label_train: Legend label for training samples.
        label_test: Legend label for test samples.
        label_pred: Legend label for the SVR prediction line.
        label_truth: Legend label for the ground-truth curve.
        title: Optional title for the plot.

    Returns:
        Dictionary containing training and test RMSE values.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(seed=42)

    X = np.linspace(0.0, 6.0, n_samples, dtype=float)
    y_true = np.sin(X) * 1.5 + 0.3 * np.cos(2 * X)
    y_noisy = y_true + rng.normal(scale=0.2, size=X.shape)

    X_train, X_test, y_train, y_test, y_true_train, y_true_test = train_test_split(
        X[:, np.newaxis],
        y_noisy,
        y_true,
        train_size=train_size,
        random_state=42,
        shuffle=True,
    )

    svr = make_pipeline(
        StandardScaler(),
        SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
    )
    svr.fit(X_train, y_train)

    train_pred = svr.predict(X_train)
    test_pred = svr.predict(X_test)

    grid = np.linspace(0.0, 6.0, 400, dtype=float)[:, np.newaxis]
    grid_truth = np.sin(grid.ravel()) * 1.5 + 0.3 * np.cos(2 * grid.ravel())
    grid_pred = svr.predict(grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X_train, y_train, color="#1f77b4", alpha=0.6, label=label_train)
    ax.scatter(X_test, y_test, color="#ff7f0e", alpha=0.6, label=label_test)
    ax.plot(grid, grid_truth, color="#2ca02c", linewidth=2, label=label_truth)
    ax.plot(grid, grid_pred, color="#d62728", linewidth=2, linestyle="--", label=label_pred)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "train_rmse": float(mean_squared_error(y_train, train_pred, squared=False)),
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
    }



metrics = run_svr_demo()
print(f"Train RMSE: {metrics['train_rmse']:.3f}")
print(f"Test RMSE: {metrics['test_rmse']:.3f}")

```

### Reading the results
- The pipeline scales training data using its mean and variance, then applies the same transform to the test set.
- `pred` contains predictions for the test features; tuning `epsilon` and `C` adjusts the trade-off between overfitting and underfitting.
- Increasing the RBF kernel’s `gamma` focuses on local patterns, whereas smaller values produce smoother functions.

## References
{{% references %}}
<li>Smola, A. J., &amp; Schölkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199–222.</li>
<li>Vapnik, V. (1995). <i>The Nature of Statistical Learning Theory</i>. Springer.</li>
{{% /references %}}
