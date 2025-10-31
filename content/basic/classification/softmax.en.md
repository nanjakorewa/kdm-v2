---
title: "Softmax Regression"
pre: "2.2.2 "
weight: 2
title_suffix: "Predict all class probabilities at once"
---

{{% summary %}}
- Softmax regression generalises logistic regression to multiple classes, producing the probability of every class simultaneously.
- Outputs lie in \\([0, 1]\\) and sum to 1, so they plug directly into decision thresholds, cost-sensitive rules, or downstream pipelines.
- Training minimises the cross-entropy loss, directly correcting discrepancies between predicted and true probability distributions.
- In scikit-learn, `LogisticRegression(multi_class="multinomial")` implements softmax regression and supports L1/L2 regularisation.
{{% /summary %}}

## Intuition
In the binary case the sigmoid provides the probability of class 1. With multiple classes we want all probabilities together. Softmax regression takes a linear score for each class, exponentiates the scores, and normalises them so they form a valid probability distribution. Higher scores become emphasised while lower scores are suppressed.

## Mathematical formulation
Let \\(K\\) be the number of classes, \\(\mathbf{w}_k\\) and \\(b_k\\) the parameters for class \\(k\\). Then

$$
P(y = k \mid \mathbf{x}) =
\frac{\exp\left(\mathbf{w}_k^\top \mathbf{x} + b_k\right)}
{\sum_{j=1}^{K} \exp\left(\mathbf{w}_j^\top \mathbf{x} + b_j\right)}.
$$

The objective is the cross-entropy loss

$$
L = - \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}(y_i = k) \log P(y = k \mid \mathbf{x}_i),
$$

with optional regularisation on the weights to prevent overfitting.

## Experiments with Python
The script below applies softmax regression to a synthetic three-class data set and visualises the decision regions. Setting `multi_class="multinomial"` activates the softmax formulation.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a 3-class dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_classes=3,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

# Softmax regression (multinomial logistic regression)
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
clf.fit(X, y)

# Create a grid for visualisation
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200),
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.title("Decision regions of softmax regression")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.show()
```

![softmax block 1](/images/basic/classification/softmax_block01.svg)

## References
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
