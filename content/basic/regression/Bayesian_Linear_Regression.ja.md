---
title: "ベイズ線形回帰"
pre: "2.1.6 "
weight: 6
title_suffix: "予測の不確実性まで推論する"
---

{{% summary %}}
- ベイズ線形回帰は係数を確率変数として扱い、予測値と不確実性の両方を同時に推定できる。
- 事前分布と尤度から事後分布を解析的に求められ、小規模データや外れ値に対して頑健に振る舞う。
- 予測分布もガウス形となるため、平均と分散を可視化して意思決定に活かせる。
- `BayesianRidge` を使えば正則化強度も自動調整され、手軽に実務へ応用可能。
{{% /summary %}}

## 直感
通常の最小二乗法では「最も確からしい係数」を 1 組だけ求めますが、ベイズ線形回帰では係数の不確実性も含めて推論します。観測データが少ない場合やノイズが大きい場合でも、ベイズ的な更新によって事前の知識とデータから得られる情報をバランス良く統合できます。

## 具体的な数式
係数ベクトル \(\boldsymbol\beta\) に対し、0 平均・分散 \(\tau^{-1}\) のガウス分布を事前分布として置き、ノイズ分散 \(\alpha^{-1}\) のガウス尤度を仮定すると、事後分布は

$$
p(\boldsymbol\beta \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\boldsymbol\beta \mid \boldsymbol\mu, \mathbf{\Sigma})
$$

で与えられます。ここで

$$
\mathbf{\Sigma} = (\alpha \mathbf{X}^\top \mathbf{X} + \tau \mathbf{I})^{-1}, \qquad
\boldsymbol\mu = \alpha \mathbf{\Sigma} \mathbf{X}^\top \mathbf{y}
$$

です。予測分布もガウス分布になり、平均と分散を解析的に計算できます。`scikit-learn` の `BayesianRidge` は \(\alpha, \tau\) もデータから推定してくれます。

## Pythonを用いた実験や説明
外れ値を含むデータで、通常の線形回帰とベイズ線形回帰を比較する例です。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error

# ノイズのある一次関数チE�Eタ�E�外れ値を一部混ぜる�E�E
rng = np.random.default_rng(0)
X = np.linspace(-4, 4, 120)
y = 1.8 * X - 0.5 + rng.normal(scale=1.0, size=X.shape)
outlier_idx = rng.choice(len(X), size=6, replace=False)
y[outlier_idx] += rng.normal(scale=8.0, size=outlier_idx.shape)
X = X[:, None]

# モチE��学翁E
ols = LinearRegression().fit(X, y)
bayes = BayesianRidge(compute_score=True).fit(X, y)

grid = np.linspace(-6, 6, 200)[:, None]
ols_mean = ols.predict(grid)
bayes_mean, bayes_std = bayes.predict(grid, return_std=True)

print("最尤推定�E MSE:", mean_squared_error(y, ols.predict(X)))
print("ベイズ回帰の MSE:", mean_squared_error(y, bayes.predict(X)))
print("学習された係数の平坁E", bayes.coef_)
print("事後�E散�E�対角�E刁E��E", np.diag(bayes.sigma_))

# 予測刁E��E�E 95% 信頼区間を可視化するための数値
upper = bayes_mean + 1.96 * bayes_std
lower = bayes_mean - 1.96 * bayes_std

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="#ff7f0e", alpha=0.6, label="観測値")
plt.plot(grid, ols_mean, color="#1f77b4", linestyle="--", label="最尤推定！ELS�E�E)
plt.plot(grid, bayes_mean, color="#2ca02c", linewidth=2, label="ベイズ線形回帰の平坁E)
plt.fill_between(grid.ravel(), lower, upper, color="#2ca02c", alpha=0.2, label="95% 信頼区閁E)
plt.xlabel("入劁E$x$")
plt.ylabel("出劁E$y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![bayesian-linear-regression block 1](/images/basic/regression/bayesian-linear-regression_block01.svg)

### 実行結果の読み方
- 外れ値が混ざってもベイズ線形回帰は回帰直線が極端に傾きにくく、平均予測に幅を持たせられる。
- 予測分布の標準偏差 `bayes_std` を使えば、信頼区間や予測区間を簡単に可視化できる。
- 係数の事後分散から、どの特徴量の不確実性が高いかを推定できる。

## 参考文献
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
