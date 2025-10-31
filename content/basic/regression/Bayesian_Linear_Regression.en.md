---
title: "Bayesian Linear Regression"
pre: "2.1.6 "
weight: 6
title_suffix: "Quantify predictive uncertainty"
---

{{% summary %}}
- Bayesian linear regression treats coefficients as random variables, estimating both predictions and their uncertainty.
- The posterior distribution is derived analytically from the prior and the likelihood, making the method robust for small or noisy datasets.
- The predictive distribution is Gaussian, so its mean and variance can be visualized and used for decision making.
- `BayesianRidge` in scikit-learn automatically tunes the noise variance, which simplifies practical adoption.
{{% /summary %}}

## Intuition
Ordinary least squares searches for a single “best” set of coefficients, yet in real data the estimate itself remains uncertain. Bayesian linear regression models that uncertainty by treating the coefficients probabilistically and combining prior knowledge with observed data. Even with limited observations, you obtain both the expected prediction and the spread around it.

## Mathematical formulation
Assume a multivariate Gaussian prior with mean 0 and variance \(\tau^{-1}\) for the coefficient vector \(\boldsymbol\beta\), and Gaussian noise \(\epsilon_i \sim \mathcal{N}(0, \alpha^{-1})\) on the observations. The posterior becomes

$$
p(\boldsymbol\beta \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\boldsymbol\beta \mid \boldsymbol\mu, \mathbf{\Sigma})
$$

with

$$
\mathbf{\Sigma} = (\alpha \mathbf{X}^\top \mathbf{X} + \tau \mathbf{I})^{-1}, \qquad
\boldsymbol\mu = \alpha \mathbf{\Sigma} \mathbf{X}^\top \mathbf{y}.
$$

The predictive distribution for a new input \(\mathbf{x}_*\) is also Gaussian, \(\mathcal{N}(\hat{y}_*, \sigma_*^2)\). `BayesianRidge` estimates \(\alpha\) and \(\tau\) from data, so you can use the model without hand-tuning them.

## Experiments with Python
The following example compares ordinary least squares with Bayesian linear regression on data containing outliers.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error

# Create a noisy linear trend and inject a few outliers
rng = np.random.default_rng(0)
X = np.linspace(-4, 4, 120)
y = 1.8 * X - 0.5 + rng.normal(scale=1.0, size=X.shape)
outlier_idx = rng.choice(len(X), size=6, replace=False)
y[outlier_idx] += rng.normal(scale=8.0, size=outlier_idx.shape)
X = X[:, None]

# Fit both models
ols = LinearRegression().fit(X, y)
bayes = BayesianRidge(compute_score=True).fit(X, y)

grid = np.linspace(-6, 6, 200)[:, None]
ols_mean = ols.predict(grid)
bayes_mean, bayes_std = bayes.predict(grid, return_std=True)

print("OLS MSE:", mean_squared_error(y, ols.predict(X)))
print("Bayesian regression MSE:", mean_squared_error(y, bayes.predict(X)))
print("Posterior mean of coefficients:", bayes.coef_)
print("Posterior variance diagonal:", np.diag(bayes.sigma_))

upper = bayes_mean + 1.96 * bayes_std
lower = bayes_mean - 1.96 * bayes_std

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="#ff7f0e", alpha=0.6, label="observations")
plt.plot(grid, ols_mean, color="#1f77b4", linestyle="--", label="OLS")
plt.plot(grid, bayes_mean, color="#2ca02c", linewidth=2, label="Bayesian mean")
plt.fill_between(grid.ravel(), lower, upper, color="#2ca02c", alpha=0.2, label="95% CI")
plt.xlabel("input $x$")
plt.ylabel("output $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![bayesian-linear-regression block 1](/images/basic/regression/bayesian-linear-regression_block01_en.png)

### Reading the results
- OLS is pulled toward the outliers, while Bayesian linear regression keeps the mean prediction more stable.
- Using `return_std=True` yields the predictive standard deviation, which makes it easy to plot credible intervals.
- Inspecting the posterior variance highlights which coefficients carry the most uncertainty.

## References
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
