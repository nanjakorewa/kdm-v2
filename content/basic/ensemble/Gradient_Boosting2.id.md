---
title: "Visualisasi Gradient Boosting | Kontribusi bertahap"
linkTitle: "Visualisasi Gradient Boosting"
seo_title: "Visualisasi Gradient Boosting | Kontribusi bertahap"
pre: "2.4.6 "
weight: 6
title_suffix: "Kontribusi bertahap"
---

{{% youtube "ZgssfFWQbZ8" %}}

Kita visualisasikan bagaimana gradient boosting regresi membaik tahap demi tahap melalui kontribusi tiap pohon.

{{% notice document %}}
- [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
{{% /notice %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.ensemble import GradientBoostingRegressor
```

## Latih dan prediksi akhir

```python
n_samples = 500
X = np.linspace(-10, 10, n_samples)[:, np.newaxis]
noise = np.random.rand(X.shape[0]) * 10
y = (np.sin(X).ravel()) * 10 + 10 + noise

n_estimators = 10
learning_rate = 0.5
reg = GradientBoostingRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
)
reg.fit(X, y)
y_pred = reg.predict(X)

plt.figure(figsize=(20, 10))
plt.scatter(X, y, c="k", marker="x", label="latih")
plt.plot(X, y_pred, c="r", label="akhir", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(y=np.mean(y), color="gray", linestyle=":", label="baseline")
plt.title("Pencocokan pada data latih")
plt.legend()
plt.show()
```

![png](/images/basic/ensemble/Gradient_Boosting2_files/Gradient_Boosting2_5_0.png)

## Tumpuk kontribusi tiap pohon

```python
fig, ax = plt.subplots(figsize=(20, 10))
temp = np.zeros(n_samples) + np.mean(y)

for i in range(n_estimators):
    res = reg.estimators_[i][0].predict(X) * learning_rate
    ax.bar(X.flatten(), res, bottom=temp, label=f"pohon {i+1}", alpha=0.05)
    temp += res

plt.scatter(X.flatten(), y, c="k", marker="x", label="latih")
plt.plot(X, y_pred, c="r", label="akhir", linewidth=1)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
```

![png](/images/basic/ensemble/Gradient_Boosting2_files/Gradient_Boosting2_7_1.png)

## Tumpukan parsial (perbaikan bertahap)

```python
for i in range(5):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.title(f"Hingga pohon {i+1}")
    temp = np.zeros(n_samples) + np.mean(y)

    for j in range(i + 1):
        res = reg.estimators_[j][0].predict(X) * learning_rate
        ax.bar(X.flatten(), res, bottom=temp, label=f"{j+1}", alpha=0.05)
        temp += res

    plt.scatter(X.flatten(), y, c="k", marker="x", label="latih")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
```

