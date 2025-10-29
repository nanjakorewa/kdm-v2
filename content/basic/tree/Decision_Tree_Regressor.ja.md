---
title: "決定木（回帰）"
pre: "2.3.2 "
weight: 2
title_suffix: "区分定数で連続値を近似する"
---

{{% summary %}}
- 回帰木は特徴量空間を条件分岐で区切り、各葉ノードで目標値の代表値（平均や中央値など）を予測する。
- 分割は平均二乗誤差などの不純度をどれだけ減らせるかで選び、区分定数関数としてデータを近似する。
- 深さや葉ノードの最小サンプル数を調整することで、当てはまりと汎化性能のバランスをとる。
- 木構造は可視化しやすく、特徴重要度や予測領域を確認するのに適している。
{{% /summary %}}

## 直感
分類木と同様に「もし◯◯なら左へ」と特徴量を基準に分割しますが、葉ノードでは連続値を返します。葉に含まれるサンプルの平均値（`criterion="squared_error"` の場合）などを予測値とするため、予測関数は段差状になります。木を深くすると細かな変化も表現できますが、過学習に注意が必要です。

## 具体的な数式
ノード \\(t\\) における平均二乗誤差（MSE）は

$$
\mathrm{MSE}(t) = \frac{1}{n_t} \sum_{i \in t} \bigl(y_i - \bar{y}_t\bigr)^2
$$

で定義されます。親ノード \\(t\\) を特徴量 \\(x_j\\) としきい値 \\(s\\) で左右に分割したとき、不純度減少量は

$$
\Delta = \mathrm{MSE}(t)
- \frac{n_L}{n_t} \mathrm{MSE}(t_L)
- \frac{n_R}{n_t} \mathrm{MSE}(t_R)
$$

です。減少量が最大になる分割を繰り返し、葉ノードでは \\(\bar{y}_t\\)（あるいは中央値）を予測値とします。

## Pythonを用いた実験や説明
以下は一次元・二次元の人工データで回帰木を学習し、段差状の予測や等高線を確認するコードです。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

```python
# 1次元のノイズ付きデータを作成
rng = np.random.RandomState(42)
X1 = np.sort(5 * rng.rand(120, 1), axis=0)                   # x in [0, 5)
y1_true = np.sin(X1).ravel()                                 # 本当の関数（例）
y1 = y1_true + rng.normal(scale=0.2, size=X1.shape[0])       # ノイズを加える

# 木を学習（深さを制御して形を観察）
reg1 = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X1, y1)
y1_pred = reg1.predict(X1)

plt.figure(figsize=(8, 4))
plt.scatter(X1, y1, s=15, c="gray", label="観測データ")
plt.plot(X1, y1_true, lw=2, label="真の関数")
plt.step(X1.ravel(), y1_pred, where="mid", lw=2, label="回帰木の予測（段差状）")
plt.xlabel("x")
plt.ylabel("y")
plt.title("回帰木は区分定数で予測する（直感）")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

![decision-tree-regressor block 2](/images/basic/tree/decision-tree-regressor_block02.svg)

```python
# 2特徴量の回帰データ
X, y = make_regression(n_samples=400, n_features=2, noise=15.0, random_state=777)

reg = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)

# 評価
r2 = r2_score(y, reg.predict(X))
rmse = mean_squared_error(y, reg.predict(X), squared=False)
mae = mean_absolute_error(y, reg.predict(X))
print(f"R2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

# メッシュで予測面を可視化（等高線）
x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                     np.linspace(y_min, y_max, 150))
zz = reg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7, 6))
cs = plt.contourf(xx, yy, zz, levels=15, cmap="viridis", alpha=0.8)
plt.colorbar(cs, label="予測値")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=20, edgecolor="k", alpha=0.7)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("回帰木の予測（等高線表示）")
plt.show()
```

![decision-tree-regressor block 3](/images/basic/tree/decision-tree-regressor_block03.svg)

```python
plt.figure(figsize=(12, 10))
plot_tree(
    reg, filled=True,
    feature_names=["x1", "x2"],
    rounded=True,
)
plt.title("決定木（回帰）の構造")
plt.show()
```

![decision-tree-regressor block 4](/images/basic/tree/decision-tree-regressor_block04.svg)

## 参考文献
{{% references %}}
<li>Breiman, L., Friedman, J. H., Olshen, R. A., &amp; Stone, C. J. (1984). <i>Classification and Regression Trees</i>. Wadsworth.</li>
<li>scikit-learn developers. (2024). <i>Decision Trees</i>. https://scikit-learn.org/stable/modules/tree.html</li>
{{% /references %}}
