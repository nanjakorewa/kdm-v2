---
title: "決定木のパラメータ"
pre: "2.3.3 "
weight: 3
title_suffix: "複雑さを制御して汎化させる"
---

{{% youtube "AOEtom_l3Wk" %}}

{{% summary %}}
- 決定木の学習は不純度の減少量が最大になる分割を繰り返すため、深く育てるほど当てはまりは向上するが過学習に陥りやすい。
- `max_depth` や `min_samples_leaf` などのパラメータは木の複雑さを抑え、汎化性能を改善するためのブレーキとして働く。
- コスト複雑度剪定（`ccp_alpha`）は訓練誤差と木のサイズのトレードオフを明示的に最適化する方法。
- 実データではグリッドサーチや交差検証を使ってハイパーパラメータを調整し、可視化で挙動を確認すると理解が深まる。
{{% /summary %}}

## 直感
決定木は分割を止めなければ訓練データを完全に覚え込むことができます。ところが、データのわずかな揺らぎにも敏感になり、未知データへの汎化性能が低下します。`max_depth` や `min_samples_split` などのパラメータは「どこまで分割してよいか」を制限することで、過学習を防ぎます。また学習後に枝を切り落とす「剪定」を行うことで、簡潔で説明しやすい木に仕上げることも可能です。

## 具体的な数式
親ノード \\(P\\) を左右の子ノード \\(L, R\\) に分割したときの不純度減少量は

$$
\Delta I = I(P) - \frac{|L|}{|P|} I(L) - \frac{|R|}{|P|} I(R)
$$

で与えられます。回帰木では \\(I(t)\\) を平均二乗誤差（`squared_error`）や平均絶対誤差（`absolute_error`）として定義します。コスト複雑度剪定では、木 \\(T\\) のコスト

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

を最小化する部分木を選びます。ここで \\(R(T)\\) は訓練誤差、\\(|T|\\) は葉ノード数、\\(\alpha \ge 0\\) は複雑さに対するペナルティを表します。

## Pythonを用いた実験や説明
次のコードでは、ハイパーパラメータを変更しながら回帰木の挙動を観察します。必要に応じて `dtreeviz` で詳細な可視化も行えます。

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)
from dtreeviz.trees import rtreeviz_bivar_3D
```

### 基本的な挙動を確認する

```python
# サンプルデータ
X, y = make_regression(n_samples=100, n_features=2, random_state=11)

# 決定木を学習
dt = DecisionTreeRegressor(max_depth=3, random_state=117117)
dt.fit(X, y)
print("feature importances:", dt.feature_importances_)
```

```python
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(
    dt, X, y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40, azim=120, dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```

### 複雑さに関わるパラメータの影響

```python
# もう少し複雑なデータ
X, y = make_regression(
    n_samples=500,
    n_features=2,
    effective_rank=4,
    noise=0.1,
    random_state=1,
)

baseline = DecisionTreeRegressor(max_depth=3, random_state=117117).fit(X, y)
deep = DecisionTreeRegressor(max_depth=10, random_state=117117).fit(X, y)
shallow = DecisionTreeRegressor(max_depth=2, random_state=117117).fit(X, y)

for name, model in [("baseline", baseline), ("deep", deep), ("shallow", shallow)]:
    pred = model.predict(X)
    print(
        f"{name:9s} R2={r2_score(y, pred):.3f} "
        f"RMSE={mean_squared_error(y, pred, squared=False):.2f} "
        f"MAE={mean_absolute_error(y, pred):.2f}"
    )
```

```python
controls = DecisionTreeRegressor(
    max_depth=5,
    min_samples_leaf=5,
    random_state=117117,
).fit(X, y)

print("max_depth=5, min_samples_leaf=5 R2:",
      r2_score(y, controls.predict(X)))
```

### コスト複雑度剪定（`ccp_alpha`）

```python
path = DecisionTreeRegressor(random_state=0).cost_complexity_pruning_path(X, y)
ccp_alphas = np.unique(path.ccp_alphas)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = []

for a in ccp_alphas:
    fold_scores = []
    for tr, va in kf.split(X):
        model = DecisionTreeRegressor(random_state=0, ccp_alpha=a)
        model.fit(X[tr], y[tr])
        fold_scores.append(r2_score(y[va], model.predict(X[va])))
    scores.append(np.mean(fold_scores))

best_alpha = ccp_alphas[int(np.argmax(scores))]
print("Best alpha:", best_alpha, "CV R2:", max(scores))
```

### 深さごとの過学習を可視化

```python
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

depths = range(1, 16)
train_scores, test_scores = [], []

for d in depths:
    model = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_tr, y_tr)
    train_scores.append(r2_score(y_tr, model.predict(X_tr)))
    test_scores.append(r2_score(y_te, model.predict(X_te)))

for d, tr, te in zip(depths, train_scores, test_scores):
    print(f"depth={d:2d}  train R2={tr: .3f}  test R2={te: .3f}")
```

### グリッドサーチでパラメータを調整する

```python
param_grid = {
    "max_depth": [3, 4, 5, None],
    "min_samples_leaf": [1, 3, 5, 10],
}
grid = GridSearchCV(
    DecisionTreeRegressor(random_state=0),
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
)
grid.fit(X, y)
print("Best params:", grid.best_params_)
```

## 参考文献
{{% references %}}
<li>Breiman, L., Friedman, J. H., Olshen, R. A., &amp; Stone, C. J. (1984). <i>Classification and Regression Trees</i>. Wadsworth.</li>
<li>Breiman, L., Friedman, J. H. (1991). <i>Cost-Complexity Pruning</i>. In: <i>Classification and Regression Trees</i>. Chapman &amp; Hall.</li>
<li>scikit-learn developers. (2024). <i>Decision Trees</i>. https://scikit-learn.org/stable/modules/tree.html</li>
{{% /references %}}
