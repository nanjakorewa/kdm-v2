---
title: "Least Squares"
pre: "2.1.1 "
weight: 1
title_suffix: "Concepts and Implementation"
---

{{% youtube "KKuAxQbuJpk" %}}


<div class="pagetop-box">
  <p><b>Least squares</b> finds the coefficients of a function that best fits pairs of observations <code>(x_i, y_i)</code> by minimizing the sum of squared residuals. We focus on the simplest case, a straight line <code>y = wx + b</code>, and walk through the intuition and a practical implementation.</p>
  </div>

{{% notice tip %}}
Math is rendered with KaTeX. <code>\(\hat y\)</code> denotes the model prediction and <code>\(\epsilon\)</code> denotes noise.
{{% /notice %}}

## Goal
- Learn the line <code>\(\hat y = wx + b\)</code> that best fits the data.
- “Best” means minimizing the sum of squared errors (SSE):
  <code>\(\displaystyle L(w,b) = \sum_{i=1}^n (y_i - (w x_i + b))^2\)</code>

## Create a simple dataset
We generate a noisy straight line and fix the random seed for reproducibility.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # optional for Japanese labels

rng = np.random.RandomState(42)
n_samples = 200

# True line (slope 0.8, intercept 0.5) with noise
X = np.linspace(-10, 10, n_samples)
epsilon = rng.normal(loc=0.0, scale=1.0, size=n_samples)
y = 0.8 * X + 0.5 + epsilon

# Reshape to 2D for scikit-learn: (n_samples, 1)
X_2d = X.reshape(-1, 1)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="observations", c="orange")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_6_0.png)

{{% notice info %}}
In scikit-learn, features are always a 2D array: rows are samples and columns are features. Use <code>X.reshape(-1, 1)</code> for a single feature.
{{% /notice %}}

## Inspect the noise
Let’s check the distribution of <code>epsilon</code>.

```python
plt.figure(figsize=(10, 5))
plt.hist(epsilon, bins=30)
plt.xlabel("$\\epsilon$")
plt.ylabel("count")
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_8_0.png)

## Linear regression (least squares) with scikit-learn
We use <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html" target="_blank" rel="noopener">sklearn.linear_model.LinearRegression</a>.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()  # fit_intercept=True by default
model.fit(X_2d, y)

print("slope w:", model.coef_[0])
print("intercept b:", model.intercept_)

y_pred = model.predict(X_2d)

# Metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("MSE:", mse)
print("R^2:", r2)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="observations", c="orange")
plt.plot(X, y_pred, label="fitted line", c="C0")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_10_0.png)

{{% notice tip %}}
Scaling is not required to solve ordinary least squares, but it helps with multivariate problems and regularization.
{{% /notice %}}

## Closed-form solution (reference)
For <code>\(\hat y = wx + b\)</code>, the minimizers are

- <code>\(\displaystyle w = \frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(x)}\)</code>
- <code>\(\displaystyle b = \bar y - w\,\bar x\)</code>

Verify with NumPy:

```python
x_mean, y_mean = X.mean(), y.mean()
w_hat = ((X - x_mean) * (y - y_mean)).sum() / ((X - x_mean) ** 2).sum()
b_hat = y_mean - w_hat * x_mean
print(w_hat, b_hat)
```

## Common pitfalls
- Array shapes: <code>X</code> should be <code>(n_samples, n_features)</code>. Even for 1 feature, use <code>reshape(-1, 1)</code>.
- Target shape: <code>y</code> can be <code>(n_samples,)</code>. <code>(n,1)</code> also works but mind broadcasting.
- Intercept: default <code>fit_intercept=True</code>. If you centered features and target, <code>False</code> is fine.
- Reproducibility: use a fixed seed via <code>np.random.RandomState</code> or <code>np.random.default_rng</code>.

## Going further (multivariate)
For multiple features, keep <code>X</code> as <code>(n_samples, n_features)</code>. Pipelines let you combine preprocessing and the estimator.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_multi = rng.normal(size=(n_samples, 2))
y_multi = 1.0 * X_multi[:, 0] - 2.0 * X_multi[:, 1] + 0.3 + rng.normal(size=n_samples)

pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_multi, y_multi)
```

{{% notice note %}}
Code blocks are for learning purposes; figures are pre-rendered for the site.
{{% /notice %}}

