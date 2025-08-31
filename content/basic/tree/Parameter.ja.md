---
title: "決定木のパラメータ"
pre: "2.3.3 "
weight: 3
title_suffix: "について仕組みを理解する"
---

{{< katex />}}

{{% youtube "AOEtom_l3Wk" %}}

<div class="pagetop-box">
  <p><b>決定木（回帰）</b>の挙動は、ハイパーパラメータの設定で大きく変わります。このページでは、各パラメータがどのように木の形や予測性能に影響するのかを、<b>数式</b>と<b>可視化</b>の両面からやさしく解説します。</p>
</div>

## このページのゴール

- 決定木（回帰）が「何を最小化」して分岐を作っているのか、式で直感を持つ  
- 主要パラメータの意味・典型値・副作用（過学習/汎化）を把握する  
- 実データ（合成データ）でパラメータを変えたときの見え方を体感する  
- 剪定（`ccp_alpha`）の考え方とチューニングの流れをつかむ  

---

## 決定木（回帰）の基本

### どんな分割を選ぶの？
回帰木は、ノード（親）を左右の子ノードに分割するときに、**不純度（誤差）**の減少が最大になるように特徴量としきい値を選びます。

親ノード \(P\) の不純度を \(I(P)\)、左右の子 \(L, R\) の不純度を \(I(L), I(R)\)、それぞれのサンプル数を \(|P|, |L|, |R|\) とすると、  
**不純度減少**は

\[
\Delta I \;=\; I(P) \;-\; \frac{|L|}{|P|} I(L) \;-\; \frac{|R|}{|P|} I(R)
\]

です。これが最大になる分割を選びます。

- `squared_error` のとき、葉の予測値は **平均** \(\bar{y}\)  
- `absolute_error` のとき、葉の予測値は **中央値** \(\mathrm{median}(y)\)

---

## 主要パラメータのチートシート

| パラメータ | 役割 / 効果 | 典型的な使い方・注意点 |
|---|---|---|
| `max_depth` | 木の**深さ**の上限。深いほど表現力↑／過学習↑ | 小さめ（3〜10）から開始 |
| `min_samples_split` | **分割を許す最小サンプル数** | 大きいほど過学習抑制 |
| `min_samples_leaf` | **葉に残す最小サンプル数** | ノイズが多いときに有効 |
| `max_leaf_nodes` | **葉の枚数**の上限 | 区画数を直接制御 |
| `criterion` | 誤差の定義（MSE, MAEなど） | 外れ値や目的に応じて選択 |
| `ccp_alpha` | **コスト複雑度剪定** | 大きいほどシンプルな木 |
| `random_state` | 乱数シード | 再現性の確保 |

---

## 剪定（`ccp_alpha`）の考え方

学習後の木 \(T\) に対して、**コスト複雑度**は

\[
R_\alpha(T) \;=\; R(T) \;+\; \alpha \, |T|
\]

で評価します。ここで \(R(T)\) は訓練誤差、\(|T|\) は葉の数。  
\(\alpha\) を大きくすると葉が減り、単純化します。

---

## まずはシンプルなデータで可視化

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
from dtreeviz.trees import dtreeviz, rtreeviz_bivar_3D

# サンプルデータ
X, y = make_regression(n_samples=100, n_features=2, random_state=11)

# 決定木を学習
dt = DecisionTreeRegressor(max_depth=3, random_state=117117)
dt.fit(X, y)

# 可視化（3D：領域と分割面）
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(
    dt, X, y, feature_names=["x1", "x2"], target_name="y",
    elev=40, azim=120, dist=8.0, show={"splits", "title"}, ax=ax
)
plt.show()
```

![png](/images/basic/tree/Parameter_files/Parameter_5_0.png)

---

## いろいろなパラメータで挙動を比べる

```python
# データを少し複雑に
X, y = make_regression(
    n_samples=500, n_features=2, effective_rank=4, noise=0.1, random_state=1
)
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# 基準モデル
dt = DecisionTreeRegressor(max_depth=3, random_state=117117).fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(
    dt, X, y, feature_names=["x1", "x2"], target_name="y",
    elev=40, azim=240, dist=8.0, show={"splits", "title"}, ax=ax
)
plt.show()
```

![png](/images/basic/tree/Parameter_files/Parameter_7_0.png)

![png](/images/basic/tree/Parameter_files/Parameter_7_1.png)

---

### `max_depth=10`（木の深さを大きくする）

```python
dt = DecisionTreeRegressor(max_depth=10, random_state=117117).fit(X, y)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(dt, X, y, ["x1","x2"], "y", elev=40, azim=240, dist=8.0, show={"splits","title"}, ax=ax)
plt.show()
```

![png](/images/basic/tree/Parameter_files/Parameter_9_0.png)

**説明**: 深さを大きくすると、データを細かく分けて複雑なルールを表現できます。その結果、訓練データにはよく適合しますが、テストデータでは過学習しやすくなります。

---

### `max_depth=5`（ほどよい深さ）

```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117).fit(X, y)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(dt, X, y, ["x1","x2"], "y", elev=40, azim=240, dist=8.0, show={"splits","title"}, ax=ax)
plt.show()
```

![png](/images/basic/tree/Parameter_files/Parameter_11_0.png)

**説明**: 適度な深さを設定することで、表現力と汎化性能のバランスが取れます。多くの場合、深さを5〜10程度に制限すると過学習を防ぎやすいです。

---

### `min_samples_split=60`（分割に必要な最小サンプル数）

```python
dt = DecisionTreeRegressor(max_depth=5, min_samples_split=60, random_state=117117).fit(X, y)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(dt, X, y, ["x1","x2"], "y", elev=40, azim=240, dist=8.0, show={"splits","title"}, ax=ax)
plt.show()
```

![png](/images/basic/tree/Parameter_files/Parameter_13_0.png)

**説明**: 分割の条件を厳しくしたため、木の細分化が抑えられます。結果としてシンプルになり、過学習を防ぎやすくなりますが、表現力は下がります。

---

### `ccp_alpha=0.4`（剪定で単純化）

```python
dt = DecisionTreeRegressor(max_depth=5, ccp_alpha=0.4, random_state=117117).fit(X, y)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(dt, X, y, ["x1","x2"], "y", elev=40, azim=240, dist=8.0, show={"splits","title"}, ax=ax)
plt.show()
```

![png](/images/basic/tree/Parameter_files/Parameter_15_0.png)

**説明**: `ccp_alpha` を大きくすると、複雑さにペナルティがかかり、不要な枝が剪定されます。木が小さくなり、解釈しやすい単純なモデルになります。

---

### `max_leaf_nodes=5`（葉の数を制限）

```python
dt = DecisionTreeRegressor(max_depth=5, max_leaf_nodes=5, random_state=117117).fit(X, y)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(dt, X, y, ["x1","x2"], "y", elev=40, azim=240, dist=8.0, show={"splits","title"}, ax=ax)
plt.show()
```

![png](/images/basic/tree/Parameter_files/Parameter_17_0.png)

**説明**: 葉の数を直接制御するため、分割の数＝予測区間の数が少なくなります。シンプルで解釈性は高まりますが、データの細かい特徴は表現できなくなります。

---

## 外れ値がある場合と `criterion`

```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=2, random_state=11)
y[1:20] = y[1:20] * 5
```

`absolute_error`:

```python
dt = DecisionTreeRegressor(max_depth=5, criterion="absolute_error", random_state=117117).fit(X, y)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(dt, X, y, ["x1","x2"], "y", elev=40, azim=240, dist=8.0, show={"splits","title"}, ax=ax)
plt.show()
```

![png](/images/basic/tree/Parameter_files/Parameter_20_0.png)

**説明**: `absolute_error` は外れ値の影響を受けにくく、中央値で予測するためロバストな分割になります。

`squared_error`:

```python
dt = DecisionTreeRegressor(max_depth=5, criterion="squared_error", random_state=117117).fit(X, y)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
_ = rtreeviz_bivar_3D(dt, X, y, ["x1","x2"], "y", elev=40, azim=240, dist=8.0, show={"splits","title"}, ax=ax)
plt.show()
```

![png](/images/basic/tree/Parameter_files/Parameter_21_0.png)

**説明**: `squared_error` は外れ値を強く罰するため、外れ値を分離するような分割が作られやすくなります。

---

## `ccp_alpha` のチューニング

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

path = DecisionTreeRegressor(random_state=0).cost_complexity_pruning_path(X, y)
ccp_alphas = np.unique(path.ccp_alphas)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = []

for a in ccp_alphas:
    fold = []
    for tr, va in kf.split(X):
        model = DecisionTreeRegressor(random_state=0, ccp_alpha=a)
        model.fit(X[tr], y[tr])
        fold.append(r2_score(y[va], model.predict(X[va])))
    scores.append(np.mean(fold))

best_alpha = ccp_alphas[np.argmax(scores)]
print("Best alpha:", best_alpha, "CV R2:", max(scores))
```

---

## 過学習の見分け方

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

depths = range(1, 16)
train_scores, test_scores = [], []

for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(Xtr, ytr)
    train_scores.append(r2_score(ytr, m.predict(Xtr)))
    test_scores.append(r2_score(yte, m.predict(Xte)))

for d, tr, te in zip(depths, train_scores, test_scores):
    print(f"depth={d:2d}  train R2={tr: .3f}  test R2={te: .3f}")
```

---

## 実務のヒント

- スケーリング不要  
- 欠損値は補完が必要  
- カテゴリ変数はエンコード必須  
- `random_state` を固定して再現性確保  
- `max_depth` と `min_samples_leaf` の合わせ技でバランス調整  
- `max_features` を制限するとランダム性と汎化力↑  

---

{{% notice document %}}
- [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)  
- [parrt/dtreeviz](https://github.com/parrt/dtreeviz)
{{% /notice %}}
