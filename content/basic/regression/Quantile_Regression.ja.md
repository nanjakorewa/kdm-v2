---
title: "分位点回帰 (Quantile Regression)"
pre: "2.1.7 "
weight: 7
title_suffix: "条件付き分布の輪郭を推定する"
---

{{% summary %}}
- 分位点回帰は平均ではなく中央値や 10% 点など任意の分位点を直接推定する回帰モデルである。
- ピンボール損失を最小化することで、外れ値に頑健で非対称なノイズにも対応できる。
- 分位点ごとに独立したモデルを学習するため、上下バンドを組み合わせれば予測区間として利用できる。
- 標準化や正則化パラメータを活用すると、収束の安定性と汎化性能を確保しやすい。
{{% /summary %}}

## 直感
最小二乗法が平均的な挙動を捉えるのに対し、分位点回帰は「どのくらいの頻度でこの値を下回るか」を直接モデル化します。需要予測の悲観・中央値・楽観シナリオやリスク管理の Value at Risk を推定したい場面など、平均だけでは意思決定に十分でないケースで威力を発揮します。

## 具体的な数式
残差を \(r = y - \hat{y}\)、分位点を \(\tau \in (0,1)\) とすると、ピンボール損失は

$$
L_\tau(r) =
\begin{cases}
\tau \, r & (r \ge 0) \\
(\tau - 1) r & (r < 0)
\end{cases}
$$

で定義されます。この損失を最小化すると、\(\tau\) 分位点に対応する線形予測子が得られます。例えば \(\tau=0.5\) なら中央値回帰になり、絶対値損失によるロバスト回帰と同じ振る舞いをします。

## Pythonを用いた実験や説明
`QuantileRegressor` を使って 0.1・0.5・0.9 分位点を推定し、線形回帰と比較します。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(123)
n_samples = 400
X = np.linspace(0, 10, n_samples)
# ノイズを非対称にする
noise = rng.gamma(shape=2.0, scale=1.0, size=n_samples) - 2.0
y = 1.5 * X + 5 + noise
X = X[:, None]

taus = [0.1, 0.5, 0.9]
models = {}
for tau in taus:
    model = make_pipeline(
        StandardScaler(with_mean=True),
        QuantileRegressor(alpha=0.001, quantile=tau, solver="highs"),
    )
    model.fit(X, y)
    models[tau] = model

ols = LinearRegression().fit(X, y)

grid = np.linspace(0, 10, 200)[:, None]
preds = {tau: m.predict(grid) for tau, m in models.items()}
ols_pred = ols.predict(grid)

for tau, pred in preds.items():
    print(f"tau={tau:.1f}, 予測の最小値 {pred.min():.2f}, 最大値 {pred.max():.2f}")

plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=15, alpha=0.4, label="観測値")
colors = {0.1: "#1f77b4", 0.5: "#2ca02c", 0.9: "#d62728"}
for tau, pred in preds.items():
    plt.plot(grid, pred, color=colors[tau], linewidth=2, label=f"分位点 τ={tau}")
plt.plot(grid, ols_pred, color="#9467bd", linestyle="--", label="平均を表す OLS")
plt.xlabel("入力 X")
plt.ylabel("出力 y")
plt.legend()
plt.tight_layout()
plt.show()
```

![quantile-regression block 1](/images/basic/regression/quantile-regression_block01.svg)

### 実行結果の読み方
- 分位点ごとに異なる直線が得られ、データの上下方向のばらつきを表現できる。
- 平均を表す最小二乗法と比べ、片側に長いノイズにも柔軟に対応している。
- 分位点を複数組み合わせると予測区間が得られ、意思決定に必要な情報を提示できる。

## 参考文献
{{% references %}}
<li>Koenker, R., &amp; Bassett, G. (1978). Regression Quantiles. <i>Econometrica</i>, 46(1), 33–50.</li>
<li>Koenker, R. (2005). <i>Quantile Regression</i>. Cambridge University Press.</li>
{{% /references %}}
