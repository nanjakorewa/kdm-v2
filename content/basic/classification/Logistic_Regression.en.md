---
title: "Logistic Regression"
pre: "2.2.1 "
weight: 1
title_suffix: "Estimate class probabilities with the sigmoid"
---

{{% summary %}}
- Logistic regression passes a linear combination of the inputs through the sigmoid function to predict the probability that the label is 1.
- The output lives in \([0, 1]\), so you can set decision thresholds flexibly and interpret coefficients as contributions to the log-odds.
- Training minimises the cross-entropy loss (equivalently maximises the log-likelihood); L1/L2 regularisation keeps the model from overfitting.
- scikit-learn's `LogisticRegression` handles preprocessing, training, and decision-boundary visualisation with a few lines of code.
{{% /summary %}}

## Intuition
Linear regression produces real-valued outputs, but classification often demands probabilities such as "how likely is class 1?". Logistic regression addresses this by feeding the linear score \(z = \mathbf{w}^\top \mathbf{x} + b\) into the sigmoid function \(\sigma(z) = 1 / (1 + e^{-z})\), yielding values that can be interpreted probabilistically. With that probability in hand, a simple rule like "predict class 1 if \(P(y=1 \mid \mathbf{x}) > 0.5\)" solves the classification task.

## Mathematical formulation
The probability of class 1 given \(\mathbf{x}\) is

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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate a 2D classification dataset
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=1,
    random_state=2,
    n_clusters_per_class=1,
)

# Train the model
clf = LogisticRegression()
clf.fit(X, y)

# Compute the decision boundary
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
slope = -w1 / w2
intercept = -b / w2

xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
xd = np.array([xmin, xmax])
yd = slope * xd + intercept

# Plot
plt.figure(figsize=(8, 8))
plt.plot(xd, yd, "k-", lw=1, label="decision boundary")
plt.scatter(*X[y == 0].T, marker="o", label="class 0")
plt.scatter(*X[y == 1].T, marker="x", label="class 1")
plt.legend()
plt.title("Logistic regression decision boundary")
plt.show()
```

![logistic-regression block 2](/images/basic/classification/logistic-regression_block02.svg)

## References
{{% references %}}
<li>Agresti, A. (2015). <i>Foundations of Linear and Generalized Linear Models</i>. Wiley.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
