---
title: "Elastic Net 回帰"
pre: "2.1.5 "
weight: 5
title_suffix: "L1とL2正則化の長所を融合する"
---

{{% summary %}}
- Elastic Net は L1 と L2 の正則化を組み合わせ、疎性と安定性の両立を図る回帰手法である。
- 強く相関した特徴量をグループとして残しながら重要度を調整できるため、解釈しやすいモデルを得やすい。
- ハイパーパラメータ \(\alpha\) と `l1_ratio` を交差検証で調整すると、汎化性能とバイアスのバランスを取りやすい。
- 標準化や十分な反復回数の確保によって、数値最適化の安定性を高められる。
{{% /summary %}}

## 直感
ラッソ回帰は係数をゼロにして特徴量選択を実現しますが、相関の強い特徴量群からは 1 つだけを残して他を落としてしまうことがあります。リッジ回帰は係数を滑らかに縮めて安定させますが、係数がゼロにはなりません。Elastic Net は両者を組み合わせることで、相関した特徴量をまとめて残しつつ、不要な係数は 0 に近づける柔軟な挙動を実現します。

## 具体的な数式
Elastic Net の目的関数は

$$
\min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left( y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b) \right)^2 + \alpha \left( \rho \lVert \boldsymbol\beta \rVert_1 + (1 - \rho) \lVert \boldsymbol\beta \rVert_2^2 \right)
$$

で、\(\alpha > 0\) が正則化の強さ、\(\rho \in [0,1]\) (`l1_ratio`) が L1 と L2 の混合比率です。リッジとラッソの中間を連続的に探索できる点が特徴です。

## Pythonを用いた実験や説明
`ElasticNetCV` を用いて \(\alpha\) と `l1_ratio` を同時にチューニングし、係数と性能を確認します。

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# 相関の強い特徴量を含むサンプルデータを生成
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

### 実行結果の読み方
- `ElasticNetCV` を使うと、L1 と L2 のバランスを含めた複数の候補を自動で評価できる。
- 相関の強い特徴量が複数残る場合でも、係数が似た大きさに調整されるため解釈しやすい。
- 収束が遅いときは特徴量を標準化したり `max_iter` を増やしたりすると改善する。

## 参考文献
{{% references %}}
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
<li>Friedman, J., Hastie, T., &amp; Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. <i>Journal of Statistical Software</i>, 33(1), 1–22.</li>
{{% /references %}}
