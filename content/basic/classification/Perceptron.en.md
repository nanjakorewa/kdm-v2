---
title: "Perceptron"
pre: "2.2.3 "
weight: 3
title_suffix: "The simplest linear classifier"
---

{{% summary %}}
- The perceptron converges in a finite number of updates on linearly separable data, making it one of the oldest classification algorithms.
- Predictions use the sign of a weighted sum \(\mathbf{w}^\top \mathbf{x} + b\); if the sign is wrong, the corresponding sample updates the weights.
- The update rule—adding the misclassified sample scaled by the learning rate—provides an intuitive introduction to gradient-based methods.
- When data are not linearly separable, feature expansion or kernel tricks are needed.
{{% /summary %}}

## Intuition
The perceptron moves the decision boundary whenever it misclassifies a sample, nudging it toward the correct side. The weight vector \(\mathbf{w}\) is normal to the decision boundary, while the bias \(b\) controls the offset. A learning rate \(\eta\) determines how large each nudge should be.

## Mathematical formulation
Predictions are computed as

$$
\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b).
$$

If a sample \((\mathbf{x}_i, y_i)\) is misclassified, update the parameters via

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i\, \mathbf{x}_i,\qquad
b \leftarrow b + \eta\, y_i.
$$

When the data are linearly separable, this procedure is guaranteed to converge.

## Experiments with Python
The following example applies the perceptron to synthetic data, reporting the number of mistakes per epoch and plotting the resulting decision boundary.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=0)
y = np.where(y == 0, -1, 1)

w = np.zeros(X.shape[1])
b = 0.0
lr = 0.1
n_epochs = 20
history = []

for epoch in range(n_epochs):
    errors = 0
    for xi, target in zip(X, y):
        update = lr * target if target * (np.dot(w, xi) + b) <= 0 else 0.0
        if update != 0.0:
            w += update * xi
            b += update
            errors += 1
    history.append(errors)
    if errors == 0:
        break

print("Finished training w=", w, " b=", b)
print("Errors per epoch:", history)

# Plot the decision boundary
xx = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)
yy = -(w[0] * xx + b) / w[1]
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
plt.plot(xx, yy, color="black", linewidth=2, label="decision boundary")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()
plt.tight_layout()
plt.show()
```

![perceptron block 1](/images/basic/classification/perceptron_block01.svg)

## References
{{% references %}}
<li>Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. <i>Psychological Review</i>, 65(6), 386–408.</li>
<li>Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016). <i>Deep Learning</i>. MIT Press.</li>
{{% /references %}}
