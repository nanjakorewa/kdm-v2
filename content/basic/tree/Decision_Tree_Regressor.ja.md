---
title: "決定木(回帰)"
pre: "2.3.2 "
weight: 2
title_suffix: "について仕組みを理解する"
---


{{% youtube "E5WOgzoEs1M" %}}

<div class="pagetop-box">
  <p><b>決定木（回帰）</b>は、特徴量に対する「はい/いいえ」の分岐を繰り返し、各葉ノードで<b>連続値</b>を予測するモデルです。  
  分岐ルールの集合は<b>木構造</b>で表されるため、モデルの挙動を可視化・解釈しやすいのが特徴です。</p>
  <p>このページでは、回帰木の仕組み（不純度＝分散の考え方）を押さえたうえで、Python（scikit-learn）で学習・可視化し、さらに <code>dtreeviz</code> で木の内部を眺めます。</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

{{% notice document %}}
- [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)  
- [sklearn.tree.plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
{{% /notice %}}

---

## 1. 仕組み（直感と数式）

回帰木は、特徴量空間を軸に平行なルールで区切っていき、**葉ノードではその領域の平均値**を予測として返します。  
分岐は「目的変数のばらつき（分散）をどれだけ減らせるか」で選びます。

- ノード \\(t\\) にある目的変数の不純度（分散）：
  $$
  \mathrm{MSE}(t) \;=\; \frac{1}{n_t}\sum_{i \in t}\bigl(y_i - \bar{y}_t\bigr)^2
  $$
- ある特徴量のしきい値 \\(s\\) で左右に分けたときの<b>減少量</b>（分割利得）：
  $$
  \Delta = \mathrm{MSE}(t)\;-\;\frac{n_L}{n_t}\mathrm{MSE}(L)\;-\;\frac{n_R}{n_t}\mathrm{MSE}(R)
  $$
  → \\(\Delta\\) が最大になる特徴量としきい値を選ぶ

> 予測は**区分的定数（段差状）**になります。滑らかさはありませんが、非線形関係を単純なルールの積み重ねで表現できます。

---

## 2. まずは1次元で直感を掴む（段差状の予測）

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
plt.title("回帰木は区分的定数で予測する（直感）")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

![decision-tree-regressor block 2](/images/basic/tree/decision-tree-regressor_block02.svg)

**ポイント**  
- 深さ（`max_depth`）を大きくすると細かく分割でき、当てはまりは上がるが過学習しやすい  
- 小さすぎると逆に表現力不足（アンダーフィット）

---

## 3. 2次元データで決定領域を見てみる

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

> 予測面が**矩形的に変化**しているのが見えます（軸に平行な分割のため）。

---

## 4. 木そのものを可視化する（plot_tree / dtreeviz）

### 4.1 scikit-learn 標準の可視化

```python
reg = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)

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

- ノードには「しきい値」「サンプル数」「MSE」「予測値（葉）」が表示されます。

### 4.2 dtreeviz でリッチに可視化

`dtreeviz` は各ノードの分布や分岐を視覚的に示してくれます（※別途インストールと Graphviz が必要）。

```python
# pip install dtreeviz graphviz
from dtreeviz.trees import dtreeviz

viz = dtreeviz(
    reg, X, y,
    feature_names=["x1", "x2"],
    target_name="y"
)
viz.save("./regression_tree.svg")  # SVG で保存（Notebookなら viz.view() でも可）
```

{{% notice document %}}
- [dtreeviz: Decision Tree Visualization](https://github.com/parrt/dtreeviz)  
- Graphviz のインストールが必要です（OS ごとの手順に従ってください）
{{% /notice %}}

---

## 5. ハイパーパラメータと過学習対策

回帰木は放っておくと細かく分割しすぎて<b>過学習</b>しがちです。以下の正則化パラメータでコントロールします。

- `max_depth` … 木の深さの上限  
- `min_samples_split` … これ未満なら子ノードを作らない  
- `min_samples_leaf` … 葉ノードに必要な最小サンプル数（**強力**）  
- `max_leaf_nodes` … 葉ノード数の上限  
- `min_impurity_decrease` … 利得がこの値未満なら分割しない

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 4, 5, None],
    "min_samples_leaf": [1, 3, 5, 10],
}
grid = GridSearchCV(DecisionTreeRegressor(random_state=0), param_grid,
                    cv=5, scoring="neg_mean_squared_error")
grid.fit(X, y)
print("Best params:", grid.best_params_)
```

---

## 6. 特徴重要度（どの特徴が効いたか）

回帰木は、どの特徴量で不純度をどれだけ減らしたかを指標に**特徴重要度**を出してくれます。

```python
reg = DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
for name, imp in zip(["x1", "x2"], reg.feature_importances_):
    print(f"{name}: {imp:.3f}")
```

> 値は 0〜1 で、合計は 1。値が大きいほど、分岐に頻繁に選ばれ、有効だったことを示します。

---

## 7. 長所・短所・実務のコツ

### 長所
- ルールが明快で<b>解釈しやすい</b>  
- 前処理（スケーリング等）がほぼ不要  
- 外れ値に比較的頑健（分割の閾値が中央値的に効くことが多い）

### 短所
- 単独の木は<b>過学習</b>しやすい／予測が<b>段差状</b>（滑らかでない）  
- 軸に平行な境界のみ → 複雑な関係の表現は苦手  
- 学習データのわずかな変化で木の形が大きく変わる（不安定）

### 実務のコツ
- `min_samples_leaf` を適切に大きくすると汎化が改善しやすい  
- **アンサンブル**（ランダムフォレスト、勾配ブースティング）で精度・安定性・滑らかさを改善  
- 木の深さ・葉のサイズを CV で調整、外れ値が多い場合はロバストな評価指標（MAE）も併用

---

## 8. まとめ

- 回帰木は、分散（MSE）減少に基づいて特徴空間を分割し、葉で平均値を返す **区分的定数モデル**。  
- 直感的・可視化しやすい反面、過学習や不連続な予測の課題がある。  
- 正則化（深さ・葉サイズ）と CV でコントロール、必要に応じて **アンサンブル**へ発展させるのが実務的。

---
