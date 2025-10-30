---
title: "Ridge and Lasso Regression"
pre: "2.1.2 "
weight: 2
title_suffix: "Improve generalization with regularization"
---

{{% summary %}}
- Ridge regression shrinks coefficients smoothly with an L2 penalty and remains stable even when features are highly correlated.
- Lasso regression applies an L1 penalty that can drive some coefficients exactly to zero, providing built-in feature selection and interpretability.
- Tuning the regularization strength \(\alpha\) controls the trade-off between fitting the training data and generalizing to unseen data.
- Combining standardization with cross-validation helps choose hyperparameters that prevent overfitting while keeping performance strong.
{{% /summary %}}

## Intuition
Least squares can produce extreme coefficients when outliers or multicollinearity are present. Ridge and Lasso add penalties so coefficients cannot grow arbitrarily large, balancing predictive accuracy and interpretability. Ridge “shrinks everything a little Ewhereas Lasso “turns some coefficients off Ein the sense that they become exactly zero.

## Mathematical formulation
Both methods minimize the usual squared-error loss plus a regularization term:

- **Ridge regression**
  $$
  \min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2 + \alpha \lVert \boldsymbol\beta \rVert_2^2
  $$
- **Lasso regression**
  $$
  \min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2 + \alpha \lVert \boldsymbol\beta \rVert_1
  $$

Larger \(\alpha\) enforces stronger shrinkage. In the case of Lasso, when \(\alpha\) exceeds a threshold, some coefficients become exactly zero, yielding sparse models.

## Experiments with Python
The example below applies ridge, lasso, and ordinary least squares to the same synthetic regression problem. We compare coefficient magnitudes and generalization scores.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Parameters
n_samples = 200
n_features = 10
n_informative = 3
rng = np.random.default_rng(42)

# True coefficients (only first three informative)
coef = np.zeros(n_features)
coef[:n_informative] = rng.normal(loc=3.0, scale=1.0, size=n_informative)

X = rng.normal(size=(n_samples, n_features))
y = X @ coef + rng.normal(scale=5.0, size=n_samples)

linear = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
ridge = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
).fit(X, y)
lasso = make_pipeline(
    StandardScaler(with_mean=False),
    LassoCV(alphas=np.logspace(-3, 1, 9), cv=5, max_iter=50_000)
).fit(X, y)

models = {
    "Linear": linear,
    "Ridge": ridge,
    "Lasso": lasso,
}

# Plot coefficient magnitudes
indices = np.arange(n_features)
width = 0.25
plt.figure(figsize=(10, 4))
plt.bar(indices - width, np.abs(linear[-1].coef_), width=width, label="Linear")
plt.bar(indices, np.abs(ridge[-1].coef_), width=width, label="Ridge")
plt.bar(indices + width, np.abs(lasso[-1].coef_), width=width, label="Lasso")
plt.xlabel("feature index")
plt.ylabel("|coefficient|")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Cross-validated R^2 scores
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name:>6}: R^2 = {scores.mean():.3f} ± {scores.std():.3f}")
```

### Reading the results
- Ridge shrinks all coefficients slightly and remains stable even with multicollinearity.
- Lasso pushes some coefficients to zero, keeping only the most important features.
- Select \(\alpha\) via cross-validation to balance bias and variance, and standardize features to ensure fair comparison across dimensions.

## References
{{% references %}}
<li>Hoerl, A. E., &amp; Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. <i>Technometrics</i>, 12(1), 55–67.</li>
<li>Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. <i>Journal of the Royal Statistical Society: Series B</i>, 58(1), 267–288.</li>
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
{{% /references %}}
