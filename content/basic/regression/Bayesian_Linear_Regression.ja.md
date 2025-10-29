---
title: "ベイズ線形回帰"
pre: "2.1.6 "
weight: 6
title_suffix: "予測の不確実性まで推論する"
---

{{< lead >}}
ベイズ線形回帰は、係数を確率分布として扱い、予測値とその不確実性を同時に推定できる拡張版の線形回帰です。
{{< /lead >}}

---

## 1. ベイズ的な見方

通常の線形回帰では、最尤推定によって「最も良い」係数 \\(\boldsymbol\beta\\) を 1 つ求めます。ベイズ線形回帰では、係数に対して事前分布を置き、データを観測した後の事後分布を計算します。

1. **事前分布**: 係数がどのくらいの大きさになりそうかの事前信念（例：平均 0、分散 \\(\tau^{-1}\\) のガウス分布）  
2. **尤度**: 観測データが得られる確率（誤差分散は \\(\alpha^{-1}\\) のガウスノイズと仮定）  
3. **事後分布**: ベイズの定理を通じて、係数の確率分布を更新する

この結果、予測値もガウス分布になり、平均だけでなく分散（不確実性）を得られます。

---

## 2. 数式のイメージ

### 事前分布
$$
p(\boldsymbol\beta) = \mathcal{N}(\boldsymbol\beta \mid \mathbf{0}, \tau^{-1} \mathbf{I})
$$

### 尤度
$$
p(\mathbf{y} \mid \mathbf{X}, \boldsymbol\beta, \alpha) = \prod_{i=1}^{n} \mathcal{N}(y_i \mid \boldsymbol\beta^\top \mathbf{x}_i, \alpha^{-1})
$$

### 事後分布
$$
p(\boldsymbol\beta \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\boldsymbol\beta \mid \boldsymbol\mu, \mathbf{\Sigma})
$$
ここで
$$
\mathbf{\Sigma} = (\alpha \mathbf{X}^\top \mathbf{X} + \tau \mathbf{I})^{-1}, \qquad
\boldsymbol\mu = \alpha \mathbf{\Sigma} \mathbf{X}^\top \mathbf{y}
$$

`scikit-learn` の `BayesianRidge` は、\\(\alpha\\) と \\(\tau\\) もデータから推定しながら係数の事後分布を求めてくれます。

---

## 3. Python 実装例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error

# ノイズのある一次関数データ（外れ値を一部混ぜる）
rng = np.random.default_rng(0)
X = np.linspace(-4, 4, 120)
y = 1.8 * X - 0.5 + rng.normal(scale=1.0, size=X.shape)
outlier_idx = rng.choice(len(X), size=6, replace=False)
y[outlier_idx] += rng.normal(scale=8.0, size=outlier_idx.shape)
X = X[:, None]

# モデル学習
ols = LinearRegression().fit(X, y)
bayes = BayesianRidge(compute_score=True).fit(X, y)

grid = np.linspace(-6, 6, 200)[:, None]
ols_mean = ols.predict(grid)
bayes_mean, bayes_std = bayes.predict(grid, return_std=True)

print("最尤推定の MSE:", mean_squared_error(y, ols.predict(X)))
print("ベイズ回帰の MSE:", mean_squared_error(y, bayes.predict(X)))
print("学習された係数の平均:", bayes.coef_)
print("事後分散（対角成分）:", np.diag(bayes.sigma_))

# 予測分布の 95% 信頼区間を可視化するための数値
upper = bayes_mean + 1.96 * bayes_std
lower = bayes_mean - 1.96 * bayes_std

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="#ff7f0e", alpha=0.6, label="観測値")
plt.plot(grid, ols_mean, color="#1f77b4", linestyle="--", label="最尤推定（OLS）")
plt.plot(grid, bayes_mean, color="#2ca02c", linewidth=2, label="ベイズ線形回帰の平均")
plt.fill_between(grid.ravel(), lower, upper, color="#2ca02c", alpha=0.2, label="95% 信頼区間")
plt.xlabel("入力 $x$")
plt.ylabel("出力 $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![bayesian-linear-regression block 1](/images/basic/regression/bayesian-linear-regression_block01.svg)

> 後で本物の図を描くときは、信頼区間を塗りつぶして「不確実性がどこで大きいか」を視覚的に示すと理解が深まります。

---

## 4. どんな場面で有効か

- **予測の不確実性が重要**なとき（リスク評価、需要予測など）  
- サンプル数が少なく、外れ値の影響を受けやすいとき（ベイズ推定で過学習を抑制）  
- コスト関数に明示的な正則化パラメータをチューニングしたくないとき（ハイパーパラメータも周辺化する）  
- 係数に意味づけを行いたい研究用途（「重みの信頼区間」を提示できる）

---

## 5. まとめ

- ベイズ線形回帰は係数を確率的に推論し、平均と不確実性を同時に扱える  
- `BayesianRidge` で手軽に実装でき、外れ値の影響にも比較的強い  
- 予測分布を可視化して意思決定に活かすのがポイント  
- より高度なベイズモデリング（ガウス過程、ベイズ階層モデル）への入口としても有用です

---
