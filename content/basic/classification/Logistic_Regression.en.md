---
title: "Logistic Regression | Modeling Class Probabilities with the Sigmoid"
linkTitle: "Logistic Regression"
seo_title: "Logistic Regression | Modeling Class Probabilities with the Sigmoid"
pre: "2.2.1 "
weight: 1
title_suffix: "Estimate class probabilities with the sigmoid"
---

{{% summary %}}
- Logistic regression passes a linear combination of the inputs through the sigmoid function to predict the probability that the label is 1.
- The output lives in \\([0, 1]\\), so you can set decision thresholds flexibly and interpret coefficients as contributions to the log-odds.
- Training minimises the cross-entropy loss (equivalently maximises the log-likelihood); L1/L2 regularisation keeps the model from overfitting.
- scikit-learn's `LogisticRegression` handles preprocessing, training, and decision-boundary visualisation with a few lines of code.
{{% /summary %}}

## Intuition
Linear regression produces real-valued outputs, but classification often demands probabilities such as "how likely is class 1?". Logistic regression addresses this by feeding the linear score \\(z = \mathbf{w}^\top \mathbf{x} + b\\) into the sigmoid function \\(\sigma(z) = 1 / (1 + e^{-z})\\), yielding values that can be interpreted probabilistically. With that probability in hand, a simple rule like "predict class 1 if \\(P(y=1 \mid \mathbf{x}) > 0.5\\)" solves the classification task.

## Mathematical formulation
The probability of class 1 given \\(\mathbf{x}\\) is

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + \exp\left(-(\mathbf{w}^\top \mathbf{x} + b)\right)}.
$$

Learning maximises the log-likelihood

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^{n} \Bigl[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Bigr], \quad p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b),
$$

or equivalently minimises the negative cross-entropy loss. Adding L2 regularisation keeps coefficients from exploding, while L1 regularisation can drive irrelevant weights all the way to zero.

## Experiments with Python
The snippet below fits logistic regression to a synthetic two-dimensional data set and visualises the resulting decision boundary. Everything—from training to plotting—fits in a few lines thanks to scikit-learn.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run_logistic_regression_demo(
    n_samples: int = 300,
    random_state: int = 2,
    label_class0: str = "class 0",
    label_class1: str = "class 1",
    label_boundary: str = "decision boundary",
    title: str = "Logistic regression decision boundary",
) -> dict[str, float]:
    """Train logistic regression on a synthetic 2D dataset and visualise the boundary.

    Args:
        n_samples: Number of samples to generate.
        random_state: Seed for reproducible sampling.
        label_class0: Legend label for class 0.
        label_class1: Legend label for class 1.
        label_boundary: Legend label for the separating line.
        title: Title for the plot.

    Returns:
        Dictionary containing training accuracy and coefficients.
    """
    japanize_matplotlib.japanize()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1,
    )

    clf = LogisticRegression()
    clf.fit(X, y)

    accuracy = float(accuracy_score(y, clf.predict(X)))
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    x1, x2 = X[:, 0], X[:, 1]
    grid_x1, grid_x2 = np.meshgrid(
        np.linspace(x1.min() - 1.0, x1.max() + 1.0, 200),
        np.linspace(x2.min() - 1.0, x2.max() + 1.0, 200),
    )
    grid = np.c_[grid_x1.ravel(), grid_x2.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(grid_x1.shape)

    cmap = ListedColormap(["#aec7e8", "#ffbb78"])
    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(grid_x1, grid_x2, probs, levels=20, cmap=cmap, alpha=0.4)
    ax.contour(grid_x1, grid_x2, probs, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(x1[y == 0], x2[y == 0], marker="o", edgecolor="k", label=label_class0)
    ax.scatter(x1[y == 1], x2[y == 1], marker="x", color="k", label=label_class1)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.colorbar(contour, ax=ax, label="P(class = 1)")
    fig.tight_layout()
    plt.show()

    return {
        "accuracy": accuracy,
        "coef_0": float(coef[0]),
        "coef_1": float(coef[1]),
        "intercept": intercept,
    }


metrics = run_logistic_regression_demo(
    label_class0="class 0",
    label_class1="class 1",
    label_boundary="decision boundary",
    title="Logistic regression decision boundary",
)
print(f"Training accuracy: {metrics['accuracy']:.3f}")
print(f"Coefficient feature 1: {metrics['coef_0']:.3f}")
print(f"Coefficient feature 2: {metrics['coef_1']:.3f}")
print(f"Intercept: {metrics['intercept']:.3f}")

```


![logistic-regression demo](/images/basic/classification/logistic-regression_block01_en.png)

## References
{{% references %}}
<li>Agresti, A. (2015). <i>Foundations of Linear and Generalized Linear Models</i>. Wiley.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
