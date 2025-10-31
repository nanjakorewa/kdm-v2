---
title: "Robust regression"
pre: "2.1.3 "
weight: 3
title_suffix: "Handle outliers with the Huber loss"
---

{{% summary %}}
- Ordinary least squares (OLS) reacts strongly to outliers because squared residuals explode, so a single erroneous measurement can distort the entire fit.
- The Huber loss keeps squared loss for small residuals but switches to a linear penalty for large ones, reducing the influence of extreme points.
- Tuning the threshold \\(\delta\\) (epsilon in scikit-learn) and the optional L2 penalty \\(\alpha\\) balances robustness against variance.
- Combining scaling with cross-validation yields stable models on real-world data sets that often mix nominal points and anomalies.
{{% /summary %}}

## Intuition
Outliers arise from sensor glitches, data entry mistakes, or regime changes. In OLS the squared residual of an outlier is enormous, so the fitted line is pulled toward it. Robust regression deliberately treats large residuals more gently so that the model follows the dominant trend while discounting dubious observations. The Huber loss is a classic choice: it behaves like squared loss near zero and like absolute loss in the tails.

## Mathematical formulation
Let the residual be \\(r = y - \hat{y}\\). For a chosen threshold \\(\delta > 0\\), the Huber loss is

$$
\ell_\delta(r) =
\begin{cases}
\dfrac{1}{2} r^2, & |r| \le \delta, \\
\delta \bigl(|r| - \dfrac{1}{2}\delta\bigr), & |r| > \delta.
\end{cases}
$$

Small residuals are squared exactly as in OLS, but large residuals grow only linearly. The influence function (the derivative) therefore saturates:

$$
\psi_\delta(r) =
\begin{cases}
r, & |r| \le \delta, \\
\delta\,\mathrm{sign}(r), & |r| > \delta.
\end{cases}
$$

In scikit-learn, the threshold corresponds to the parameter `epsilon`. Adding an L2 penalty \\(\alpha \lVert \boldsymbol\beta \rVert_2^2\\) further stabilizes the coefficients when features correlate.

## Experiments with Python
We visualize the loss shapes and compare OLS, Ridge, and Huber on a small synthetic data set that contains a single extreme outlier.

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

### Huber loss versus squared and absolute losses

```python
def huber_loss(r: np.ndarray, delta: float = 1.5):
    half_sq = 0.5 * np.square(r)
    lin = delta * (np.abs(r) - 0.5 * delta)
    return np.where(np.abs(r) <= delta, half_sq, lin)

delta = 1.5
r_vals = np.arange(-2, 2, 0.01)
h_vals = huber_loss(r_vals, delta=delta)

plt.figure(figsize=(8, 6))
plt.plot(r_vals, np.square(r_vals), "red",   label=r"squared $r^2$")
plt.plot(r_vals, np.abs(r_vals),    "orange",label=r"absolute $|r|$")
plt.plot(r_vals, h_vals,            "green", label=fr"Huber ($\delta={delta}$)")
plt.axhline(0, color="k", linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel("residual $r$")
plt.ylabel("loss")
plt.title("Squared, absolute, and Huber losses")
plt.show()
```

### A toy data set with an outlier

```python
np.random.seed(42)

N = 30
x1 = np.arange(N)
x2 = np.arange(N)
X = np.c_[x1, x2]
epsilon = np.random.rand(N)
y = 5 * x1 + 10 * x2 + epsilon * 10

y[5] = 500  # introduce one extreme outlier

plt.figure(figsize=(8, 6))
plt.plot(x1, y, "ko", label="data")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Data containing an outlier")
plt.show()
```

### Comparing OLS, Ridge, and Huber

```python
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression

plt.figure(figsize=(8, 6))

huber = HuberRegressor(alpha=0.0, epsilon=3.0)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="Huber")

ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="Ridge (α=1.0)")

ols = LinearRegression()
ols.fit(X, y)
plt.plot(x1, ols.predict(X), "r-", label="OLS")

plt.plot(x1, y, "kx", alpha=0.7)
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Influence of an outlier on different regressors")
plt.grid(alpha=0.3)
plt.show()
```

### Reading the results
- OLS (red) is heavily pulled by the outlier.
- Ridge (orange) is slightly more stable thanks to the L2 penalty but still deviates.
- Huber (green) limits the impact of the outlier and follows the main trend better.

## References
{{% references %}}
<li>Huber, P. J. (1964). Robust Estimation of a Location Parameter. <i>The Annals of Mathematical Statistics</i>, 35(1), 73–101.</li>
<li>Hampel, F. R. et al. (1986). <i>Robust Statistics: The Approach Based on Influence Functions</i>. Wiley.</li>
<li>Huber, P. J., &amp; Ronchetti, E. M. (2009). <i>Robust Statistics</i> (2nd ed.). Wiley.</li>
{{% /references %}}
