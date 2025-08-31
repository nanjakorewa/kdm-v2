---
title: "AdaBoost (Regresi)"
pre: "2.4.4 "
weight: 4
title_suffix: "Intuisi, loss, dan praktik"
---

{{% youtube "1K-h4YzrnsY" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
```

## Cara AdaBoost.R2 memberi bobot
Tiap putaran, galat yang lebih besar mendapat bobot lebih besar. Opsi loss di scikit-learn:

- <b>loss='linear'</b>: sebanding dengan galat
- <b>loss='square'</b>: menekankan galat besar (sensitif outlier)
- <b>loss='exponential'</b>: lebih sensitif

{{% notice document %}}
[AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
{{% /notice %}}

## Fit pada data sintetis
```python
X = np.linspace(-10, 10, 500)[:, np.newaxis]
y = (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10 + 10 + np.linspace(-2, 2, 500)

reg = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=100,
    random_state=100,
    loss="linear",
    learning_rate=0.8,
)
reg.fit(X, y)
y_pred = reg.predict(X)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", label="latih")
plt.plot(X, y_pred, c="r", label="prediksi", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Pencocokan pada data latih")
plt.legend()
plt.show()
```

![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_6_0.png)

## Visualisasi pembobotan (gagasan)
Bungkus pohon dasar untuk menghitung kemunculan sampel dan memahami pembobotan internal.

