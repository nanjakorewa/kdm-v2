---
title: "k-Nearest Neighbours (k-NN)"
pre: "2.2.7 "
weight: 7
title_suffix: "Lazy learning based on distances"
---

{{% summary %}}
- k-NN stores the training data and predicts by majority vote among the \\(k\\) nearest neighbours of a query point.
- The main hyperparameters are the number of neighbours \\(k\\) and the distance weighting scheme, which are easy to tune.
- It naturally models non-linear decision boundaries, but distance metrics become less informative in high dimensions (“the curse of dimensionality”).
- Standardising features or selecting informative ones makes distance calculations more stable.
{{% /summary %}}

## Intuition
Under the assumption that “nearby samples share the same label”, k-NN finds the \\(k\\) closest training samples to a query point and predicts the label via majority vote (optionally weighted by distance). Because no explicit model is trained in advance, k-NN is known as a lazy learner.

## Mathematical formulation
For a test point \\(\mathbf{x}\\), let \\(\mathcal{N}_k(\mathbf{x})\\) be the set of \\(k\\) nearest neighbours in the training set. The vote for class \\(c\\) is

$$
v_c = \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \,\mathbb{1}(y_i = c),
$$

where the weights \\(w_i\\) can be uniform or a function of distance (e.g. inverse distance). The predicted class is the one with the highest vote.

## Experiments with Python
The following code compares cross-validation accuracy for different values of \\(k\\) and visualises the decision boundary on a two-dimensional data set.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Evaluate how accuracy changes with different k
X_full, y_full = make_blobs(
    n_samples=600,
    centers=3,
    cluster_std=[1.1, 1.0, 1.2],
    random_state=7,
)
ks = [1, 3, 5, 7, 11]
for k in ks:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
    scores = cross_val_score(model, X_full, y_full, cv=5)
    print(f"k={k}: CV accuracy={scores.mean():.3f} +/- {scores.std():.3f}")

# Visualise the decision boundary in 2D
X_vis, y_vis = make_blobs(
    n_samples=450,
    centers=[(-2, 3), (1.8, 2.2), (0.8, -2.5)],
    cluster_std=[1.0, 0.9, 1.1],
    random_state=42,
)
vis_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))
vis_model.fit(X_vis, y_vis)

fig, ax = plt.subplots(figsize=(6, 4.5))
xx, yy = np.meshgrid(
    np.linspace(X_vis[:, 0].min() - 1.5, X_vis[:, 0].max() + 1.5, 300),
    np.linspace(X_vis[:, 1].min() - 1.5, X_vis[:, 1].max() + 1.5, 300),
)
grid = np.column_stack([xx.ravel(), yy.ravel()])
pred = vis_model.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, pred, levels=np.arange(0, 4) - 0.5, cmap="Pastel1", alpha=0.9)

scatter = ax.scatter(
    X_vis[:, 0],
    X_vis[:, 1],
    c=y_vis,
    cmap="Set1",
    edgecolor="#1f2937",
    linewidth=0.6,
)
ax.set_title("k-NN (k=5, distance weights) decision boundary")
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(alpha=0.15)

legend = ax.legend(
    handles=scatter.legend_elements()[0],
    labels=[f"class {i}" for i in range(len(np.unique(y_vis)))],
    loc="upper right",
    frameon=True,
)
legend.get_frame().set_alpha(0.9)

fig.tight_layout()
```

![knn block 1](/images/basic/classification/knn_block01.svg)

## References
{{% references %}}
<li>Cover, T. M., &amp; Hart, P. E. (1967). Nearest Neighbor Pattern Classification. <i>IEEE Transactions on Information Theory</i>, 13(1), 21–27.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
