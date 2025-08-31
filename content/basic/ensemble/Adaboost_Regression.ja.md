---
title: "AdaBoost（回帰）"
pre: "2.4.4 "
weight: 4
title_suffix: "の直感・数式・実装"
---

{{% youtube "1K-h4YzrnsY" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
```

## 仕組み（AdaBoost.R2 の直感）

回帰では、各反復での予測誤差から重みを更新し、誤差の大きいサンプルにより注意を向けます。scikit-learn の AdaBoostRegressor では損失の種類を選択可能：

- loss='linear': 重みを誤差に比例させる
- loss='square': 誤差二乗で強調（外れ値に敏感）
- loss='exponential': 指数的に強調（さらに敏感）

{{% notice document %}}
[AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
{{% /notice %}}

## 訓練データに回帰モデルを当てはめる

```python
# 訓練データ
X = np.linspace(-10, 10, 500)[:, np.newaxis]
y = (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10 + 10 + np.linspace(-2, 2, 500)

# 回帰モデルを作成
reg = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=5),
    n_estimators=100,
    random_state=100,
    loss="linear",
    learning_rate=0.8,
)
reg.fit(X, y)
y_pred = reg.predict(X)

# 訓練データへのフィッティング具合を確認
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", label="訓練データ")
plt.plot(X, y_pred, c="r", label="最終モデルの予測", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("訓練データへのフィッティング")
plt.legend()
plt.show()
```

![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_6_0.png)

## 標本の重みを可視化（loss='linear' の例）

AdaBoost 回帰では誤差の大きい点の重みが増え、次の弱学習器で重点的に学習されます。ここでは内部の sample_weight に相当する挙動を可視化するための補助クラス例を用意します。

```python
# NOTE: 内部の重み付け挙動を可視化するための簡単なラッパ
class DummyRegressor:
    def __init__(self):
        self.model = DecisionTreeRegressor(max_depth=5)
        self.error_vector = None
        self.X_for_plot = None
        self.y_for_plot = None

    def fit(self, X, y):
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        # AdaBoostRegressor の実装に合わせ、誤差の大きさを重みの指標にする
        self.error_vector = np.abs(y_pred - y)
        self.X_for_plot = X.copy()
        self.y_for_plot = y.copy()
        return self.model

    def predict(self, X, check_input=True):
        return self.model.predict(X)

    def get_params(self, deep=False):
        return {}

    def set_params(self, deep=False):
        return {}
```

```python
# 可視化用の関数
def visualize_weight(reg, X, y, y_pred):
    """標本の重みに相当する値（サンプリング回数）をプロットする補助関数"""
    assert reg.estimators_ is not None, "len(reg.estimators_) > 0"

    for i, estimators_i in enumerate(reg.estimators_):
        if i % 100 == 0:
            # i番目の弱学習器の学習に何回登場したかをカウント
            weight_dict = {xi: 0 for xi in X.ravel()}
            for xi in estimators_i.X_for_plot.ravel():
                weight_dict[xi] += 1

            plt.figure(figsize=(10, 4))
            plt.bar(weight_dict.keys(), weight_dict.values(), width=0.04)
            plt.xlabel("x")
            plt.ylabel("サンプリング回数（重みの直感）")
            plt.title(f"{i+1} 個目までの弱学習器学習での出現回数")
            plt.show()
```

