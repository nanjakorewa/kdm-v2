---
title: "リッジ回帰・ラッソ回帰"
pre: "2.1.2 "
weight: 2
title_suffix: "の仕組みの説明"
---

{{% youtube "rhGYOBrxPXA" %}}



<div class="pagetop-box">
  <p><b>リッジ回帰（Ridge）</b>と<b>ラッソ回帰（Lasso）</b>は、係数の大きさにペナルティ（正則化項）を課すことで、<b>過学習を抑え</b>、<b>汎化性能</b>を高める線形回帰モデルです。こうしたテクニックを<b>正則化（regularization）</b>と呼びます。</p>
  <p>このページでは、最小二乗法・リッジ回帰・ラッソ回帰を <code>scikit-learn</code> で実行し、係数の振る舞いとモデルの違いを直感的に理解します。</p>
</div>

---

## 1. 直感と数式（なぜ正則化が効くのか）

最小二乗法（通常の線形回帰）は、残差平方和
$$
\text{RSS}(\boldsymbol\beta, b)=\sum_{i=1}^n \big(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\big)^2
$$
を最小化します。特徴量が多い・相関が強い・ノイズが大きい場合、係数が過度に大きくなり過学習しやすくなります。

そこで、係数の大きさにペナルティを足して、<b>大きすぎる係数を抑える</b>のが正則化です。

- **リッジ（L2正則化）**  
  $$
  \min_{\boldsymbol\beta,b}\;\text{RSS}(\boldsymbol\beta,b)\;+\;\alpha \lVert \boldsymbol\beta\rVert_2^2
  $$
  すべての係数を<b>なだらかに小さく</b>します（ゼロにはなりにくい）。

- **ラッソ（L1正則化）**  
  $$
  \min_{\boldsymbol\beta,b}\;\text{RSS}(\boldsymbol\beta,b)\;+\;\alpha \lVert \boldsymbol\beta\rVert_1
  $$
  係数の一部を<b>ちょうどゼロ</b>にします（特徴量選択の効果）。

> 直感：<br>
> - リッジ＝「全体を縮める」→ 安定する、でもスパースにはなりにくい。  
> - ラッソ＝「一部を0にする」→ 自動で特徴選択、ただし相関が強いと不安定になりやすい。  

---

## 2. 実験環境の準備

{{% notice tip %}}
正則化では<b>特徴量のスケール（単位）</b>が重要です。スケールが違うとペナルティの効き方が変わるため、<b>標準化（StandardScaler）</b>を併用します。
{{% /notice %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

---

## 3. 実験用データの作成（有効特徴は2つだけ）

{{% notice document %}}
- [sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)
{{% /notice %}}

`make_regression` で「ほんとうに効いている特徴量は2つ（`n_informative=2`）」という状況を作ります。あとから冗長な特徴を足して、<b>不要特徴がたくさんある</b>状態にしてみます。

```python
n_features = 5
n_informative = 2

X, y = make_regression(
    n_samples=500,
    n_features=n_features,
    n_informative=n_informative,
    noise=0.5,
    random_state=777,
)

# 冗長な非線形変換を足して、特徴を水増し（冗長特徴）
X = np.concatenate([X, np.log(X + 100)], axis=1)

# 目的変数を標準化（スケール合わせ）
y = (y - y.mean()) / np.std(y)
```

---

## 4. 3つのモデルを同じ条件で学習

{{% notice document %}}
- [sklearn.pipeline.make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)  
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)  
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
{{% /notice %}}

```python
# 標準化 → 回帰 をひとつのパイプラインに
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
rid_r = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2)).fit(X, y)
las_r = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.1, max_iter=10_000)).fit(X, y)
```

> `Lasso` は収束しづらいことがあるので `max_iter` を増やしておくのが実務では安全です。

---

## 5. 係数の大きさを比較する（可視化）

- 最小二乗法（LinearRegression）：係数は<b>ほとんど0にならない</b>。  
- リッジ：係数が<b>まんべんなく小さく</b>なる。  
- ラッソ：係数が<b>一部ちょうど0</b>になる（特徴選択）。

```python
feat_index = np.arange(X.shape[1])

plt.figure(figsize=(12, 4))
plt.bar(feat_index - 0.25, np.abs(lin_r.steps[1][1].coef_), width=0.25, label="Linear")
plt.bar(feat_index,         np.abs(rid_r.steps[1][1].coef_), width=0.25, label="Ridge")
plt.bar(feat_index + 0.25,  np.abs(las_r.steps[1][1].coef_), width=0.25, label="Lasso")

plt.xlabel(r"$\beta$ のインデックス")
plt.ylabel(r"$|\beta|$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

![ridge-and-lasso block 4](/images/basic/regression/ridge-and-lasso_block04.svg)

{{% notice tip %}}
このデータでは「有用な特徴は2つだけ」でした（<code>n_informative=2</code>）。  
ラッソは不要特徴の多くで係数を 0 にしやすく、モデルを<b>スパース化</b>できます。  
一方リッジは係数を平均的に縮め、<b>分散の大きい推定</b>を安定化します。
{{% /notice %}}

---

## 6. 汎化性能をざっくり比較（交差検証）

固定の <code>alpha</code> で良し悪しを語るのは危険なので、簡易に交差検証でスコアも見ておきます。

```python
from sklearn.metrics import make_scorer, mean_squared_error

scorer = make_scorer(mean_squared_error, greater_is_better=False)

models = {
    "Linear": lin_r,
    "Ridge (α=2)": rid_r,
    "Lasso (α=0.1)": las_r,
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name:12s}  R^2 (CV mean±std): {scores.mean():.3f} ± {scores.std():.3f}")
```

{{% notice tip %}}
<b>結論はデータ次第</b>。  
高相関や多次元・少データでは、無印よりリッジ／ラッソの方が安定＆高スコアになりやすいです。  
ただし<b>α（正則化強度）は必ず調整</b>しましょう。
{{% /notice %}}

---

## 7. α（正則化強度）を自動で選ぶ

実務では <code>GridSearchCV</code> や <code>RidgeCV</code> / <code>LassoCV</code> をよく使います。

```python
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
).fit(X, y)

lasso_cv = make_pipeline(
    StandardScaler(with_mean=False),
    LassoCV(alphas=np.logspace(-3, 1, 9), cv=5, max_iter=50_000)
).fit(X, y)

print("Best α (RidgeCV):", ridge_cv.steps[1][1].alpha_)
print("Best α (LassoCV):", lasso_cv.steps[1][1].alpha_)
```

---

## 8. どっちを使う？（選び方の目安）

- **特徴量が多い・相関が強い**：まずは **Ridge**（安定化）。  
- **特徴選択も兼ねたい／解釈しやすいモデルが欲しい**：**Lasso**。  
- **相関が強くLassoが不安定**：**Elastic Net**（L1+L2の折衷）を検討。

---

## 9. まとめ

- 正則化は<b>係数の大きさにペナルティ</b>を設定して過学習を抑える。  
- **Ridge（L2）**：係数を<b>なめらかに縮める</b>。ゼロにはなりにくい。  
- **Lasso（L1）**：係数の<b>一部をゼロ</b>にし、特徴選択効果がある。  
- <b>標準化は必須級</b>、<b>αは交差検証でチューニング</b>するのが実務の基本。

---
