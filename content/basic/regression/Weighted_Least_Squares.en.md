---
title: "Weighted Least Squares (WLS)"
pre: "2.1.11 "
weight: 11
title_suffix: "Handle observations with unequal variance"
---

{{% summary %}}
- Weighted least squares assigns observation-specific weights so trustworthy measurements influence the fitted line more strongly.
- Multiplying squared errors by weights downplays high-variance observations and keeps the estimate close to reliable data.
- You can run WLS with scikit-learn’s `LinearRegression` by providing `sample_weight`.
- Weights can stem from known variances, residual diagnostics, or domain knowledge; careful design is crucial.
{{% /summary %}}

## Intuition
Ordinary least squares assumes every observation is equally reliable, yet real-world sensors often vary in precision. WLS “listens” more to the data points you trust by emphasizing them during fitting, allowing the familiar linear regression framework to handle heteroscedastic data.

## Mathematical formulation
With positive weights \(w_i\), minimize

$$
L(\boldsymbol\beta, b) = \sum_{i=1}^{n} w_i \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2.
$$

The optimal choice \(w_i \propto 1/\sigma_i^2\) (inverse variance) gives more influence to precise observations.

## Experiments with Python
We compare OLS and WLS on data whose noise level differs across regions.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(7)
n_samples = 200
X = np.linspace(0, 10, n_samples)
true_y = 1.2 * X + 3

# Use different noise levels in separate regions (heteroscedastic errors)
noise_scale = np.where(X < 5, 0.5, 2.5)
y = true_y + rng.normal(scale=noise_scale)

weights = 1.0 / (noise_scale ** 2)  # Use inverse variance as weights
X = X[:, None]

ols = LinearRegression().fit(X, y)
wls = LinearRegression().fit(X, y, sample_weight=weights)

grid = np.linspace(0, 10, 200)[:, None]
ols_pred = ols.predict(grid)
wls_pred = wls.predict(grid)

print("OLS slope:", ols.coef_[0], " intercept:", ols.intercept_)
print("WLS slope:", wls.coef_[0], " intercept:", wls.intercept_)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, c=noise_scale, cmap="coolwarm", s=25, label="observations (color=noise)")
plt.plot(grid, 1.2 * grid.ravel() + 3, color="#2ca02c", label="true line")
plt.plot(grid, ols_pred, color="#1f77b4", linestyle="--", linewidth=2, label="OLS")
plt.plot(grid, wls_pred, color="#d62728", linewidth=2, label="WLS")
plt.xlabel("input $x$")
plt.ylabel("output $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![weighted-least-squares block 1](/images/basic/regression/weighted-least-squares_block01.svg)

### Reading the results
- Weighting draws the fit toward the low-noise region, producing estimates close to the true line.
- OLS is skewed by the noisy region and underestimates the slope.
- Performance hinges on choosing appropriate weights; diagnostics and domain intuition matter.

## References
{{% references %}}
<li>Carroll, R. J., &amp; Ruppert, D. (1988). <i>Transformation and Weighting in Regression</i>. Chapman &amp; Hall.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
