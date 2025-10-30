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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
)

svr.fit(X_train, y_train)
pred = svr.predict(X_test)
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
