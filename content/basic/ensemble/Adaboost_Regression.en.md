---
title: "AdaBoost (Regression)"
pre: "2.4.4 "
weight: 4
title_suffix: "Intuition, loss options, and practice"
---

{{% youtube "1K-h4YzrnsY" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
```

## How AdaBoost.R2 reweights errors
At each round, larger prediction errors receive larger weights so the next weak learner focuses on them. In scikit-learn you can choose the loss:

- <b>loss='linear'</b>: proportional to error
- <b>loss='square'</b>: emphasizes larger errors (more sensitive to outliers)
- <b>loss='exponential'</b>: even more sensitive

{{% notice document %}}
[AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
{{% /notice %}}

## Fit on synthetic data
```python
# training data
X = np.linspace(-10, 10, 500)[:, np.newaxis]
y = (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10 + 10 + np.linspace(-2, 2, 500)

# model
reg = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=100,
    random_state=100,
    loss="linear",
    learning_rate=0.8,
)
reg.fit(X, y)
y_pred = reg.predict(X)

# plot fit
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", label="train")
plt.plot(X, y_pred, c="r", label="prediction", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fitting on training data")
plt.legend()
plt.show()
```

![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_6_0.png)

## Visualize sample weighting (idea)
We can inspect how often each sample gets drawn in the internal reweighting by wrapping a tree and tracking appearances (illustrative helper omitted here for brevity).

