---
title: "勾配ブースティング（基礎）"
pre: "2.4.5 "
weight: 5
title_suffix: "の直感・数式・実装"
---

{{% youtube "ZgssfFWQbZ8" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.ensemble import GradientBoostingRegressor
```

## 直感（数式）

目的関数 \(L(y, F(x))\) を最小化するため、段階的に関数を加えていきます：

- 初期モデル \(F_0(x)\)
- 負の勾配（擬似残差） \(r_{im} = -[\partial L/\partial F]_{F=F_{m-1}}\)
- 更新 \(F_m(x) = F_{m-1}(x) + \nu\, \rho_m\, h_m(x)\)

---

## 訓練データに回帰モデルを当てはめる

```python
# 訓練データ（三角関数にノイズ）
X = np.linspace(-10, 10, 500)[:, np.newaxis]
noise = np.random.rand(X.shape[0]) * 10
y = (
    (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10
    + 10
    + np.linspace(-10, 10, 500)
    + noise
)

# 回帰モデル
reg = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.5,
)
reg.fit(X, y)
y_pred = reg.predict(X)

# 可視化
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", label="訓練データ")
plt.plot(X, y_pred, c="r", label="予測", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("訓練データへのフィッティング")
plt.legend()
plt.show()
```

![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_5_0.png)

---

## 損失関数の違い（外れ値への挙動）

`loss` を `{"squared_error", "absolute_error", "huber", "quantile"}` で比較します。二乗誤差は外れ値に敏感、absolute/huber は外れ値の影響を抑えます。

```python
# 外れ値を混ぜたデータ
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
    plt.scatter(X, y, c="k", marker="x", label="訓練データ")
    plt.plot(X, y_pred, c="r", label="予測", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"loss={loss}")
    plt.legend()
    plt.show()
```

![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_0.png)
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_1.png)
