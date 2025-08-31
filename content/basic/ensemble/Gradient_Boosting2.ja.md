---
title: "勾配ブースティングの可視化"
pre: "2.4.6 "
weight: 6
title_suffix: "段階的な改善の見える化"
---

{{% youtube "ZgssfFWQbZ8" %}}

勾配ブースティング回帰の学習過程を、木ごとの寄与として可視化します。

{{% notice document %}}
- [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
{{% /notice %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.ensemble import GradientBoostingRegressor
```

## 学習と最終予測

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
plt.scatter(X, y, c="k", marker="x", label="訓練データ")
plt.plot(X, y_pred, c="r", label="最終予測", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(y=np.mean(y), color="gray", linestyle=":", label="初期値の直感")
plt.title("訓練データへのフィッティング")
plt.legend()
plt.show()
```

![png](/images/basic/ensemble/Gradient_Boosting2_files/Gradient_Boosting2_5_0.png)

## 木ごとの寄与を積み上げて表示

```python
fig, ax = plt.subplots(figsize=(20, 10))
temp = np.zeros(n_samples) + np.mean(y)

for i in range(n_estimators):
    res = reg.estimators_[i][0].predict(X) * learning_rate
    ax.bar(X.flatten(), res, bottom=temp, label=f"{i+1} 本目の木の寄与", alpha=0.05)
    temp += res

plt.scatter(X.flatten(), y, c="k", marker="x", label="訓練データ")
plt.plot(X, y_pred, c="r", label="最終予測", linewidth=1)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
```

![png](/images/basic/ensemble/Gradient_Boosting2_files/Gradient_Boosting2_7_1.png)

## 途中までの積み上げ（段階的な改善）

```python
for i in range(5):
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.title(f"{i+1} 本目までの木の寄与")
    temp = np.zeros(n_samples) + np.mean(y)

    for j in range(i + 1):
        res = reg.estimators_[j][0].predict(X) * learning_rate
        ax.bar(X.flatten(), res, bottom=temp, label=f"{j+1} 本目", alpha=0.05)
        temp += res

    plt.scatter(X.flatten(), y, c="k", marker="x", label="訓練データ")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
```

