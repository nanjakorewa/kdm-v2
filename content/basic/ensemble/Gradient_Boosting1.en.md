---
title: "Gradient Boosting (Basics) | Intuition, formulas, and practice"
linkTitle: "Gradient Boosting (Basics)"
seo_title: "Gradient Boosting (Basics) | Intuition, formulas, and practice"
pre: "2.4.5 "
weight: 5
title_suffix: "Intuition, formulas, and practice"
---

{{% youtube "ZgssfFWQbZ8" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.ensemble import GradientBoostingRegressor
```

## Intuition (formulas)

Minimize a loss \\(L(y,F(x))\\) by stagewise additive modeling:

- Initialize \\(F_0(x)\\).
- For \\(m=1,\dots,M\\): compute negative gradients (pseudo-residuals)
  \\(r_{im} = -\left[\partial L(y_i, F(x_i))/\partial F\right]_{F=F_{m-1}}\\), fit a tree \\(h_m\\) to \\(r_{im}\\), and update
  \\(F_m(x) = F_{m-1}(x) + \nu\, \rho_m\, h_m(x)\\) (\\(\nu\\): learning rate).

---

## Fit on synthetic data

```python
X = np.linspace(-10, 10, 500)[:, np.newaxis]
noise = np.random.rand(X.shape[0]) * 10
y = (
    (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10
    + 10
    + np.linspace(-10, 10, 500)
    + noise
)

reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.5)
reg.fit(X, y)
y_pred = reg.predict(X)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", label="train")
plt.plot(X, y_pred, c="r", label="prediction", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fitting on training data")
plt.legend()
plt.show()
```

![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_5_0.png)

---

## Effect of the loss function (outliers)

Compare `loss` in `{ "squared_error", "absolute_error", "huber", "quantile" }`.

```python
X = np.linspace(-10, 10, 500)[:, np.newaxis]
noise = np.random.rand(X.shape[0]) * 10
for i in range(0, X.shape[0], 80):
    noise[i] = 70 + np.random.randint(-10, 10)
y = (
    (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10
    + 10
    + np.linspace(-10, 10, 500)
    + noise
)

for loss in ["squared_error", "absolute_error", "huber", "quantile"]:
    reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.5, loss=loss)
    reg.fit(X, y)
    y_pred = reg.predict(X)

    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, c="k", marker="x", label="train")
    plt.plot(X, y_pred, c="r", label="prediction", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"loss={loss}")
    plt.legend()
    plt.show()
```

![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_0.png)
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_1.png)

