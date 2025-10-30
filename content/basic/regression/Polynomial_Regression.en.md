---
title: "Polynomial Regression"
pre: "2.1.4 "
weight: 4
title_suffix: "Capture nonlinear patterns with a linear model"
---

{{% summary %}}
- Polynomial regression expands features with powers so that a linear model can fit nonlinear relationships.
- The model remains a linear combination of coefficients, preserving closed-form solutions and interpretability.
- Higher degrees increase expressiveness but also invite overfitting, making regularization and cross-validation important.
- Standardizing features and tuning the degree plus penalty strength leads to stable predictions.
{{% /summary %}}

## Intuition
A straight line cannot describe smooth curves or hill-shaped patterns. By expanding the input with polynomial terms—\(x, x^2, x^3, \dots\) for univariate regression, or powers and interaction terms for multivariate regression—we can express nonlinear behaviour while still using a linear model.

## Mathematical formulation
Given \(\mathbf{x} = (x_1, \dots, x_m)\), we build a polynomial feature vector \(\phi(\mathbf{x})\) up to degree \(d\) and fit a linear regression on top of it. For \(m = 2\) and \(d = 2\),

$$
\phi(\mathbf{x}) = (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2),
$$

and the model becomes

$$
y = \mathbf{w}^\top \phi(\mathbf{x}).
$$

As \(d\) grows, the number of terms increases rapidly, so in practice we start with degree 2 or 3 and pair it with regularization (e.g., Ridge) when necessary.

## Experiments with Python
Below we add third-degree polynomial features and fit a curve to data generated from a cubic function plus noise.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # optional
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 200)
y_true = 0.5 * X**3 - 1.2 * X**2 + 2.0 * X + 1.5
y = y_true + rng.normal(scale=2.0, size=X.shape)

X = X[:, None]

linear_model = LinearRegression().fit(X, y)
poly_model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression()
).fit(X, y)

grid = np.linspace(-3.5, 3.5, 300)[:, None]
linear_pred = linear_model.predict(grid)
poly_pred = poly_model.predict(grid)
true_curve = 0.5 * grid.ravel()**3 - 1.2 * grid.ravel()**2 + 2.0 * grid.ravel() + 1.5

print("Linear regression MSE:", mean_squared_error(y, linear_model.predict(X)))
print("Cubic polynomial regression MSE:", mean_squared_error(y, poly_model.predict(X)))

plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=20, color="#ff7f0e", alpha=0.6, label="observations")
plt.plot(grid, true_curve, color="#2ca02c", linewidth=2, label="true curve")
plt.plot(grid, linear_pred, color="#1f77b4", linestyle="--", linewidth=2, label="linear regression")
plt.plot(grid, poly_pred, color="#d62728", linewidth=2, label="degree-3 polynomial")
plt.xlabel("input $x$")
plt.ylabel("output $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![polynomial-regression block 1](/images/basic/regression/polynomial-regression_block01.svg)

### Reading the results
- Plain linear regression misses the curvature, especially near the center, while the cubic polynomial follows the true curve closely.
- Increasing the degree improves training fit but can make extrapolation unstable.
- Combining polynomial features with a regularized regression (e.g., Ridge) via a pipeline helps curb overfitting.

## References
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
