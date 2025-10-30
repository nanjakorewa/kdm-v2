---
title: "ベイズ線形回帰"
pre: "2.1.6 "
weight: 6
title_suffix: "予測の不確実性まで推論する"
---

{{% summary %}}
- ベイズ線形回帰は係数を確率変数として扱い、予測値と不確実性を同時に推定できる。
- 事前分布と尤度から事後分布を解析的に求められ、小規模データや外れ値に対して頑健に振る舞う。
- 予測分布がガウス形となるため、平均と分散を可視化して意思決定に活用できる。
- `BayesianRidge` を使えばノイズ分散まで自動調整され、実務導入が容易になる。
{{% /summary %}}

## 直感
最小二乗法は「最もありそうな係数」を 1 組だけ推定しますが、現実のデータではその推定にも不確実性が残ります。ベイズ線形回帰では係数を確率分布として推論し、観測データと事前知識を組み合わせることで予測値の平均と幅の両方を得られます。データが少ない場面でも、モデルがどの程度自信を持っているのかを可視化できるのが強みです。

## 具体的な数式
係数ベクトル \(\boldsymbol\beta\) に平均 0、分散 \(\tau^{-1}\) の多変量ガウス事前分布を置き、観測ノイズ \(\epsilon_i \sim \mathcal{N}(0, \alpha^{-1})\) を仮定すると、事後分布は

$$
p(\boldsymbol\beta \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\boldsymbol\beta \mid \boldsymbol\mu, \mathbf{\Sigma})
$$

となります。ここで

$$
\mathbf{\Sigma} = (\alpha \mathbf{X}^\top \mathbf{X} + \tau \mathbf{I})^{-1}, \qquad
\boldsymbol\mu = \alpha \mathbf{\Sigma} \mathbf{X}^\top \mathbf{y}
$$

です。予測分布もガウス形となり、入力 \(\mathbf{x}_*\) に対して \(\mathcal{N}(\hat{y}_*, \sigma_*^2)\) が得られます。`scikit-learn` の `BayesianRidge` は \(\alpha\) と \(\tau\) もデータから推定してくれるため、手軽にこの枠組みを利用できます。

## Pythonを用いた実験や説明
外れ値を含む一次関数データで、最尤推定による線形回帰とベイズ線形回帰を比較します。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error

# ノイズのある一次関数データに外れ値を数点混ぜる
rng = np.random.default_rng(0)
X = np.linspace(-4, 4, 120)
y = 1.8 * X - 0.5 + rng.normal(scale=1.0, size=X.shape)
outlier_idx = rng.choice(len(X), size=6, replace=False)
y[outlier_idx] += rng.normal(scale=8.0, size=outlier_idx.shape)
X = X[:, None]

# モデルを学習
ols = LinearRegression().fit(X, y)
bayes = BayesianRidge(compute_score=True).fit(X, y)

grid = np.linspace(-6, 6, 200)[:, None]
ols_mean = ols.predict(grid)
bayes_mean, bayes_std = bayes.predict(grid, return_std=True)

print("最尤推定の MSE:", mean_squared_error(y, ols.predict(X)))
print("ベイズ回帰の MSE:", mean_squared_error(y, bayes.predict(X)))
print("学習された係数の平均:", bayes.coef_)
print("事後分散共分散行列の対角成分:", np.diag(bayes.sigma_))

upper = bayes_mean + 1.96 * bayes_std
lower = bayes_mean - 1.96 * bayes_std

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="#ff7f0e", alpha=0.6, label="観測値")
plt.plot(grid, ols_mean, color="#1f77b4", linestyle="--", label="最尤推定 (OLS)")
plt.plot(grid, bayes_mean, color="#2ca02c", linewidth=2, label="ベイズ線形回帰の平均")
plt.fill_between(grid.ravel(), lower, upper, color="#2ca02c", alpha=0.2, label="95% 信頼区間")
plt.xlabel("入力 $x$")
plt.ylabel("出力 $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![bayesian-linear-regression block 1](/images/basic/regression/bayesian-linear-regression_block01.svg)

### 実行結果の読み方
- OLS は外れ値に引きずられて直線が傾きやすいが、ベイズ線形回帰は平均の変動が抑えられる。
- `return_std=True` で得られた標準偏差から、予測の信頼区間を簡単に描ける。
- 係数の事後分散を確認すると、どの特徴量に不確実性が残っているかを把握できる。

## 参考文献
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
