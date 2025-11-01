---
title: "Partial Least Squares Regression (PLS) | Learn latent factors aligned with the target"
linkTitle: "Partial Least Squares Regression (PLS)"
seo_title: "Partial Least Squares Regression (PLS) | Learn latent factors aligned with the target"
pre: "2.1.9 "
weight: 9
title_suffix: "Learn latent factors aligned with the target"
---

{{% summary %}}
- Partial Least Squares (PLS) extracts latent factors that maximize the covariance between predictors and the target before performing regression.
- Unlike PCA, the learned axes incorporate target information, preserving predictive performance while reducing dimensionality.
- Tuning the number of latent factors stabilizes models in the presence of strong multicollinearity.
- Inspecting loadings reveals which combinations of features are most related to the target.
{{% /summary %}}

## Intuition
Principal component regression relies only on feature variance, so directions that matter for the target might be discarded. PLS builds latent factors while simultaneously considering predictors and response, retaining the information most useful for prediction. As a result you can work with a compact representation without sacrificing accuracy.

## Mathematical formulation
Given a predictor matrix \\(\mathbf{X}\\) and response vector \\(\mathbf{y}\\), PLS alternates updates of latent scores \\(\mathbf{t} = \mathbf{X} \mathbf{w}\\) and \\(\mathbf{u} = \mathbf{y} c\\) so that their covariance \\(\mathbf{t}^\top \mathbf{u}\\) is maximized. Repeating this procedure yields a set of latent factors on which a linear regression model

$$
\hat{y} = \mathbf{t} \boldsymbol{b} + b_0
$$

is fitted. The number of factors \\(k\\) is typically chosen via cross-validation.

## Experiments with Python
We compare PLS performance for different numbers of latent factors on the Linnerud fitness dataset.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_linnerud
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_pls_latent_factors(
    cv_splits: int = 5,
    xlabel: str = "Number of latent factors",
    ylabel: str = "CV MSE (lower is better)",
    label_best: str = "best={k}",
    title: str | None = None,
) -> dict[str, object]:
    """Cross-validate PLS regression for different latent factor counts.

    Args:
        cv_splits: Number of folds for cross-validation.
        xlabel: Label for the number-of-factors axis.
        ylabel: Label for the cross-validation error axis.
        label_best: Format string for the best-factor annotation.
        title: Optional plot title.

    Returns:
        Dictionary with the selected factor count, CV score, and loadings.
    """
    japanize_matplotlib.japanize()
    data = load_linnerud()
    X = data["data"]
    y = data["target"][:, 0]

    max_components = min(X.shape[1], 6)
    components = np.arange(1, max_components + 1)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=0)

    scores = []
    pipelines = []
    for k in components:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("pls", PLSRegression(n_components=int(k))),
        ])
        cv_score = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
        ).mean()
        scores.append(cv_score)
        pipelines.append(model)

    scores_arr = np.array(scores)
    best_idx = int(np.argmax(scores_arr))
    best_k = int(components[best_idx])
    best_mse = float(-scores_arr[best_idx])

    best_model = pipelines[best_idx].fit(X, y)
    x_loadings = best_model["pls"].x_loadings_
    y_loadings = best_model["pls"].y_loadings_

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(components, -scores_arr, marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=label_best.format(k=best_k))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "best_k": best_k,
        "best_mse": best_mse,
        "x_loadings": x_loadings,
        "y_loadings": y_loadings,
    }



metrics = evaluate_pls_latent_factors()
print(f"Best number of latent factors: {metrics['best_k']}")
print(f"Best CV MSE: {metrics['best_mse']:.3f}")
print("X loadings:
", metrics['x_loadings'])
print("Y loadings:
", metrics['y_loadings'])

```

![partial-least-squares block 1](/images/basic/regression/partial-least-squares_block01_en.png)

### Reading the results
- Cross-validated MSE decreases as factors are added, reaches a minimum, and then worsens if you keep adding more.
- Inspecting `x_loadings_` and `y_loadings_` shows which features contribute most to each latent factor.
- Standardizing inputs ensures features measured on different scales contribute evenly.

## References
{{% references %}}
<li>Wold, H. (1975). Soft Modelling by Latent Variables: The Non-Linear Iterative Partial Least Squares (NIPALS) Approach. In <i>Perspectives in Probability and Statistics</i>. Academic Press.</li>
<li>Geladi, P., &amp; Kowalski, B. R. (1986). Partial Least-Squares Regression: A Tutorial. <i>Analytica Chimica Acta</i>, 185, 1–17.</li>
{{% /references %}}
