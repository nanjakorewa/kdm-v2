---
title: "AdaBoost (Regresión)"
pre: "2.4.4 "
weight: 4
title_suffix: "Intuición, pérdidas y práctica"
---

{{% youtube "1K-h4YzrnsY" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
```

## Cómo repondera AdaBoost.R2
En cada ronda, los puntos con mayor error reciben más peso. En scikit-learn:

- <b>loss='linear'</b>: proporcional al error
- <b>loss='square'</b>: enfatiza grandes errores (sensible a outliers)
- <b>loss='exponential'</b>: aún más sensible

{{% notice document %}}
[AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
{{% /notice %}}

## Ajuste en datos sintéticos
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
plt.scatter(X, y, c="k", marker="x", label="entrenamiento")
plt.plot(X, y_pred, c="r", label="predicción", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste sobre los datos de entrenamiento")
plt.legend()
plt.show()
```

![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_6_0.png)

## Visualizar el ponderado (idea)
Se puede envolver el árbol base para contar apariciones y entender el reponderado interno.

