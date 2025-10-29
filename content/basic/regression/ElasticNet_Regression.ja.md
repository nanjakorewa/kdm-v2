---
title: "Elastic Net 回帰"
pre: "2.1.5 "
weight: 5
title_suffix: "L1 と L2 正則化のいいとこ取り"
---

{{< lead >}}
リッジ回帰（L2）とラッソ回帰（L1）の良いところを両取りしたのが Elastic Net です。多数の相関した特徴量があるときでも安定して係数を推定できます。
{{< /lead >}}

---

## 1. Elastic Net の考え方

リッジ回帰は係数を滑らかに縮小し、ラッソ回帰は不要な係数をゼロにします。Elastic Net は 2 つの正則化を組み合わせた目的関数を最小化することで、その中間的なふるまいを実現します。

![Elastic Net penalty geometry](/images/elastic-net-penalty.png)


- \\(\alpha\\) : 正則化の強さ  
- \\(\rho\\) (`l1_ratio`): L1 と L2 をどの割合で混ぜるか (\\(0 \leq \rho \leq 1\\))

### 特徴

- 相関が強い特徴量が複数あっても、一つだけに絞り込まずグループとして残しやすい  
- ラッソの「係数ゼロ化」効果と、リッジの「数値安定性」の両方を得られる  
- ハイパーパラメータが 2 つ（\\(\alpha\\) と \\(\rho\\)）あるため、交差検証での探索が重要

---

## 2. Python 実装例（`ElasticNetCV`）

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 相関の強い重複特徴量を含むサンプルデータを生成
X, y, coef = make_regression(
    n_samples=500,
    n_features=30,
    n_informative=10,
    noise=15.0,
    coef=True,
    random_state=123,
)

# 特徴量を複製して相関を意図的に強める
X = np.hstack([X, X[:, :5] + np.random.normal(scale=0.1, size=(X.shape[0], 5))])
feature_names = [f"x{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ElasticNetCV で alpha と l1_ratio を同時にチューニング
enet_cv = ElasticNetCV(
    l1_ratio=[0.2, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=np.logspace(-3, 1, 30),
    cv=5,
    random_state=42,
    max_iter=5000,
)
enet_cv.fit(X_train, y_train)

print("最適 alpha:", enet_cv.alpha_)
print("最適 l1_ratio:", enet_cv.l1_ratio_)

# 最適ハイパーパラメータで学習し、性能を評価
enet = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_, max_iter=5000)
enet.fit(X_train, y_train)

train_pred = enet.predict(X_train)
test_pred = enet.predict(X_test)

print("Train R^2:", r2_score(y_train, train_pred))
print("Test R^2:", r2_score(y_test, test_pred))
print("Test RMSE:", mean_squared_error(y_test, test_pred, squared=False))

# 係数の大きい順に並べ替えて確認
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": enet.coef_,
}).sort_values("coef", key=lambda s: s.abs(), ascending=False)
print(coef_df.head(10))
```

> 実際に図を作るときは、`enet_cv.path_`（`ElasticNetCV` の属性）を使うと \\(\alpha\\) による係数の変化を線グラフで描けます。

---

## 3. ハイパーパラメータ設計のヒント

- `l1_ratio` は 0.5 付近（L1 と L2 を半分ずつ）から試し、交差検証で広めに探索する  
- 特徴量数がサンプル数より多い場合は、`l1_ratio` を高めに設定し疎構造を優先  
- 数値解法の収束を安定させるため `StandardScaler` などで特徴量を標準化する  
- 交差検証時に `max_iter` を十分大きく設定し、収束警告が出ないか確認する

---

## 4. まとめ

- Elastic Net は L1 と L2 正則化を組み合わせた汎用性の高い線形回帰  
- 相関の強い特徴量が多い場合にラッソよりも安定した選択ができる  
- `ElasticNetCV` を使うと \\(\alpha\\) と `l1_ratio` を一括でチューニングできる  
- 標準化と交差検証をセットで行い、再現性のある性能を追求しましょう

---
