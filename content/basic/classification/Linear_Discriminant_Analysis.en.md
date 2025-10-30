---
title: "Linear Discriminant Analysis (LDA)"
pre: "2.2.4 "
weight: 4
title_suffix: "Learn directions that separate classes"
---

{{% summary %}}
- LDA finds directions that maximize the ratio of between-class variance to within-class variance, serving both classification and dimensionality reduction.
- The decision boundary takes the form \(\mathbf{w}^\top \mathbf{x} + b = 0\), which becomes a line in 2D or a plane in 3D, giving a clear geometric interpretation.
- Assuming each class is Gaussian with the same covariance matrix, LDA approaches the Bayes-optimal classifier.
- scikit-learn’s `LinearDiscriminantAnalysis` makes it easy to visualise decision boundaries and to inspect the projected features.
{{% /summary %}}

## Intuition
LDA searches for directions that keep samples of the same class close together while pushing samples from different classes far apart. Projecting the data onto that direction makes the classes easier to separate, so it can be used directly for classification or as a dimensionality-reduction step before another classifier.

## Mathematical formulation
For the two-class case, the projection direction \(\mathbf{w}\) maximizes

$$
J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}},
$$

where \(\mathbf{S}_B\) is the between-class scatter matrix and \(\mathbf{S}_W\) is the within-class scatter matrix. In the multi-class case we obtain up to \(K-1\) projection directions, which can be used for dimensionality reduction.

## Experiments with Python
Below we apply LDA to a synthetic two-class data set, draw the decision boundary, and plot the projected 1D features. Calling `transform` returns the projected data directly.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

rs = 42
X, y = make_blobs(
    n_samples=200, centers=2, n_features=2, cluster_std=2, random_state=rs
)

clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X, y)

w = clf.coef_[0]
b = clf.intercept_[0]

xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
ys_boundary = -(w[0] / w[1]) * xs - b / w[1]

scale = 3.0
origin = X.mean(axis=0)
arrow_end = origin + scale * (w / np.linalg.norm(w))

plt.figure(figsize=(7, 7))
plt.title("LDA decision boundary and projection direction", fontsize=16)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.8, label="samples")
plt.plot(xs, ys_boundary, "k--", lw=1.2, label=r"$\mathbf{w}^\top \mathbf{x} + b = 0$")
plt.arrow(
    origin[0],
    origin[1],
    (arrow_end - origin)[0],
    (arrow_end - origin)[1],
    head_width=0.5,
    length_includes_head=True,
    color="k",
    alpha=0.7,
    label="projection direction $\mathbf{w}$",
)
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()
plt.xlim(X[:, 0].min() - 2, X[:, 0].max() + 2)
plt.ylim(X[:, 1].min() - 2, X[:, 1].max() + 2)
plt.grid(alpha=0.25)
plt.show()

X_1d = clf.transform(X)[:, 0]

plt.figure(figsize=(8, 4))
plt.title("One-dimensional projection by LDA", fontsize=15)
plt.hist(X_1d[y == 0], bins=20, alpha=0.7, label="class 0")
plt.hist(X_1d[y == 1], bins=20, alpha=0.7, label="class 1")
plt.xlabel("projected feature")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

![linear-discriminant-analysis block 2](/images/basic/classification/linear-discriminant-analysis_block02.svg)

## References
{{% references %}}
<li>Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic Problems. <i>Annals of Eugenics</i>, 7(2), 179–188.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
