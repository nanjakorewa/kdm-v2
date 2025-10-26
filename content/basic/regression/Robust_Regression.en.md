---
title: "Outliers and Robustness"
pre: "2.1.3 "
weight: 3
title_suffix: "Handling with Huber Regression"
---

{{% youtube "CrN5Si0379g" %}}


<div class="pagetop-box">
  <p><b>Outliers</b> are observations that deviate strongly from most of the data. What counts as an outlier depends on the problem, the distribution, and the target scale. Here we contrast ordinary least squares (squared loss) with <b>Huber loss</b> on data that include an extreme point.</p>
  </div>

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

---

## 1. Why OLS is sensitive to outliers

OLS minimizes the sum of squared residuals
$$
\text{RSS} = \sum_{i=1}^n (y_i - \hat y_i)^2.
$$
Because residuals are <b>squared</b>, even a single extreme point can dominate the loss and <b>drag the fitted line</b> toward the outlier.

---

## 2. Huber loss: a compromise between squared and absolute

The <b>Huber loss</b> uses squared loss for small residuals and absolute loss for large residuals. For residual <code>r = y - \hat y</code> and threshold <code>\\(\delta > 0\\)</code>:

$$
\ell_\delta(r) = \begin{cases}
\dfrac{1}{2}r^2, & |r| \le \delta \\
\delta\left(|r| - \dfrac{1}{2}\delta\right), & |r| > \delta.
\end{cases}
$$

The derivative (influence) is
$$
\psi_\delta(r) = \frac{d}{dr}\ell_\delta(r) = \begin{cases}
r, & |r| \le \delta \\
\delta\,\mathrm{sign}(r), & |r| > \delta,
\end{cases}
$$
so the gradient is <b>clipped</b> for large residuals (outliers).

> Note: in scikit-learn’s <code>HuberRegressor</code>, the threshold is parameter <code>epsilon</code> (corresponds to <code>\\(\delta\\)</code> above).

---

## 3. Visualizing the loss shapes

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
plt.title("Squared vs absolute vs Huber")
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_5_0.png)

---

## 4. What happens with an outlier? (data)

Create a simple 2-feature linear problem and inject <b>one extreme outlier</b> in <code>y</code>.

```python
np.random.seed(42)

N = 30
x1 = np.arange(N)
x2 = np.arange(N)
X = np.c_[x1, x2]                      # shape (N, 2)
epsilon = np.random.rand(N)            # noise in [0, 1)
y = 5 * x1 + 10 * x2 + epsilon * 10    # true relation + noise

y[5] = 500  # one very large outlier

plt.figure(figsize=(8, 6))
plt.plot(x1, y, "ko", label="data")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Dataset with an outlier")
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_7_0.png)

---

## 5. Compare OLS vs Ridge vs Huber

- <b>OLS</b> (squared loss): very sensitive to outliers.  
- <b>Ridge</b> (L2): shrinks coefficients; slightly more stable, but still affected.  
- <b>Huber</b>: clips the influence of outliers; the line is less dragged.

{{% notice document %}}
- [HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
{{% /notice %}}

{{% notice seealso %}}
[Preprocessing approach: label anomalies (JP)](https://k-dm.work/ja/prep/numerical/add_label_to_anomaly/)
{{% /notice %}}

```python
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression

plt.figure(figsize=(8, 6))

# Huber: use epsilon=3 to reduce outlier influence
huber = HuberRegressor(alpha=0.0, epsilon=3.0)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="Huber")

# Ridge (L2). With alpha≈0, behaves like OLS.
ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="Ridge (α=1.0)")

# OLS
lr = LinearRegression()
lr.fit(X, y)
plt.plot(x1, lr.predict(X), "r-", label="OLS")

# raw data
plt.plot(x1, y, "kx", alpha=0.7)

plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Effect of an outlier on fitted lines")
plt.grid(alpha=0.3)
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_9_0.png)

Interpretation:
- OLS (red) is strongly pulled by the outlier.
- Ridge (orange) mitigates slightly but remains affected.
- Huber (green) reduces outlier influence and follows the overall trend better.

---

## 6. Parameters: epsilon and alpha

- <code>epsilon</code> (threshold <code>\\(\delta\\)</code>):
  - Larger → closer to OLS; smaller → closer to absolute loss.
  - Depends on residual scale; standardization or robust scaling helps.
- <code>alpha</code> (L2 penalty):
  - Stabilizes coefficients, useful under collinearity.

Sensitivity to <code>epsilon</code>:

```python
from sklearn.metrics import mean_squared_error

for eps in [1.2, 1.5, 2.0, 3.0]:
    h = HuberRegressor(alpha=0.0, epsilon=eps).fit(X, y)
    mse = mean_squared_error(y, h.predict(X))
    print(f"epsilon={eps:>3}: MSE={mse:.3f}")
```

---

## 7. Practical notes

- <b>Scaling</b>: if feature/target scales differ, the meaning of <code>epsilon</code> changes; standardize or use robust scaling.
- <b>High leverage points</b>: Huber is robust to vertical outliers in <code>y</code>, but not necessarily to extreme points in <code>X</code>.
- <b>Choosing thresholds</b>: tune <code>epsilon</code> and <code>alpha</code> (e.g., via <code>GridSearchCV</code>).
- <b>Evaluate with CV</b>: don’t judge by training fit only.

---

## 8. Summary

- OLS is sensitive to outliers; the fit can be dragged.
- Huber uses squared loss for small errors and absolute for large errors, effectively <b>clipping gradients</b> for outliers.
- Tuning <code>epsilon</code> and <code>alpha</code> balances robustness and fit.
- Beware of leverage points; combine with inspection and preprocessing if needed.

---

