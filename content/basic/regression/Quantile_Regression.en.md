---
title: "Quantile Regression"
pre: "2.1.7 "
weight: 7
title_suffix: "Estimate conditional distribution contours"
---

{{% summary %}}
- Quantile regression directly estimates arbitrary quantiles—such as the median or the 10th percentile—instead of only the mean.
- Minimizing the pinball loss yields robustness to outliers and accommodates asymmetric noise.
- Independent models can be fit for different quantiles, and stacking them forms prediction intervals.
- Feature scaling and regularization help stabilize convergence and maintain generalization.
{{% /summary %}}

## Intuition
Whereas least squares focuses on average behavior, quantile regression asks “how often does the response fall below this value?” It is ideal for planning pessimistic, median, and optimistic scenarios in demand forecasting, or for estimating Value at Risk in finance—situations where the mean alone is insufficient.

## Mathematical formulation
With residual \(r = y - \hat{y}\) and quantile level \(\tau \in (0, 1)\), the pinball loss is defined as

$$
L_\tau(r) =
\begin{cases}
\tau r & (r \ge 0) \\
(\tau - 1) r & (r < 0)
\end{cases}
$$

Minimizing this loss yields a linear predictor for the \(\tau\)-quantile. Setting \(\tau = 0.5\) recovers the median and leads to the same solution as least absolute deviations regression.

## Experiments with Python
We use `QuantileRegressor` to estimate the 0.1, 0.5, and 0.9 quantiles and compare them with ordinary least squares.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(123)
n_samples = 400
X = np.linspace(0, 10, n_samples)
# Introduce asymmetric noise
noise = rng.gamma(shape=2.0, scale=1.0, size=n_samples) - 2.0
y = 1.5 * X + 5 + noise
X = X[:, None]

taus = [0.1, 0.5, 0.9]
models = {}
for tau in taus:
    model = make_pipeline(
        StandardScaler(with_mean=True),
        QuantileRegressor(alpha=0.001, quantile=tau, solver="highs"),
    )
    model.fit(X, y)
    models[tau] = model

ols = LinearRegression().fit(X, y)

grid = np.linspace(0, 10, 200)[:, None]
preds = {tau: m.predict(grid) for tau, m in models.items()}
ols_pred = ols.predict(grid)

for tau, pred in preds.items():
    print(f"tau={tau:.1f}, min prediction {pred.min():.2f}, max prediction {pred.max():.2f}")

plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=15, alpha=0.4, label="observations")
colors = {0.1: "#1f77b4", 0.5: "#2ca02c", 0.9: "#d62728"}
for tau, pred in preds.items():
    plt.plot(grid, pred, color=colors[tau], linewidth=2, label=f"quantile τ={tau}")
plt.plot(grid, ols_pred, color="#9467bd", linestyle="--", label="mean (OLS)")
plt.xlabel("input X")
plt.ylabel("output y")
plt.legend()
plt.tight_layout()
plt.show()
```

![quantile-regression block 1](/images/basic/regression/quantile-regression_block01.svg)

### Reading the results
- Each quantile produces a different line, capturing the vertical spread of the data.
- Compared with the mean-focused OLS model, quantile regression adapts to skewed noise.
- Combining multiple quantiles yields prediction intervals that communicate decision-relevant uncertainty.

## References
{{% references %}}
<li>Koenker, R., &amp; Bassett, G. (1978). Regression Quantiles. <i>Econometrica</i>, 46(1), 33–50.</li>
<li>Koenker, R. (2005). <i>Quantile Regression</i>. Cambridge University Press.</li>
{{% /references %}}
