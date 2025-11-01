---
title: "Naive Bayes | Fast inference with conditional independence"
linkTitle: "Naive Bayes"
seo_title: "Naive Bayes | Fast inference with conditional independence"
pre: "2.2.6 "
weight: 6
title_suffix: "Fast inference with conditional independence"
---

{{% summary %}}
- Naive Bayes assumes conditional independence between features and combines prior probabilities with likelihoods via Bayes窶・rule.
- Training and inference are extremely fast, making it a strong baseline for high-dimensional sparse data such as text or spam filtering.
- Laplace smoothing and TF-IDF features mitigate issues with unseen words and frequency imbalance.
- When the independence assumption is too strong, consider feature selection or ensembling Naive Bayes with other models.
{{% /summary %}}

## Intuition
Bayes窶・rule states that 窶徘rior ﾃ・likelihood 竏・posterior窶・ If features are conditionally independent, the likelihood factorises into a product of per-feature probabilities. Naive Bayes leverages this approximation, delivering robust estimates even with small training sets.

## Mathematical formulation
For class \\(y\\) and features \\(\mathbf{x} = (x_1, \ldots, x_d)\\),

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y).
$$

Different likelihood models suit different data types: the multinomial model for word counts, the Bernoulli model for binary presence/absence, and Gaussian Naive Bayes for continuous values.

## Experiments with Python
The snippet below trains a multinomial Naive Bayes classifier on a subset of the 20 Newsgroups data set, using TF-IDF features. Even with thousands of features the model trains quickly, and the classification report summarises performance.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB


def run_naive_bayes_demo(
    n_samples: int = 600,
    n_classes: int = 3,
    random_state: int = 0,
    title: str = "Decision regions of Gaussian Naive Bayes",
    xlabel: str = "feature 1",
    ylabel: str = "feature 2",
) -> dict[str, float]:
    """Train Gaussian Naive Bayes on synthetic data and plot decision regions."""
    japanize_matplotlib.japanize()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state,
    )

    clf = GaussianNB()
    clf.fit(X, y)

    accuracy = float(accuracy_score(y, clf.predict(X)))
    conf = confusion_matrix(y, clf.predict(X))

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    preds = clf.predict(grid).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(grid_x, grid_y, preds, alpha=0.25, cmap="coolwarm", levels=np.arange(-0.5, n_classes + 0.5, 1))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()

    return {"accuracy": accuracy, "confusion": conf}


metrics = run_naive_bayes_demo(
    title="Decision regions of Gaussian Naive Bayes",
    xlabel="feature 1",
    ylabel="feature 2",
)
print(f"Training accuracy: {metrics['accuracy']:.3f}")
print("Confusion matrix:")
print(metrics['confusion'])

```


## References
{{% references %}}
<li>Manning, C. D., Raghavan, P., &amp; Schﾃｼtze, H. (2008). <i>Introduction to Information Retrieval</i>. Cambridge University Press.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}

