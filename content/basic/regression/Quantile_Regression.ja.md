---
title: "分位点回帰（Quantile Regression）"
pre: "2.1.7 "
weight: 7
title_suffix: "条件付き分布の幅まで推定する"
---

{{< lead >}}
平均だけでなく「中央値」や「上位 90% 点」など、目的変数の分布全体を知りたいときは分位点回帰が便利です。特定の分位を直接最適化することで、外れ値に強く、上下バンドの推定にも役立ちます。
{{< /lead >}}

---

## 1. 分位点回帰のアイデア

- 最小二乗法は誤差を二乗して平均を推定 → **平均値に敏感**  
- 分位点回帰は **ピンボール損失（チェック関数）** を最小化し、指定した分位 \\(\tau\\) の条件付き分布を推定  
- 例えば \\(\tau = 0.5\\) なら中央値、\\(\tau = 0.9\\) なら上位 10% 付近の値を予測

ピンボール損失:

$$
L_\tau(r) =
\begin{cases}
\tau \, r & (r \ge 0)\\\\
(\tau - 1) r & (r < 0)
\end{cases}
$$

ここで \\(r = y - \hat{y}\\)。中央値を学習したいときは正負対称の絶対値損失になり、外れ値に頑健です。

---

## 2. どんな場面で使う？

- 需要・売上の **予測レンジ**（悲観/中央値/楽観）を示したいとき  
- リスク評価で「何％の確率でこの水準を超えるか」を見積もりたいとき  
- ノイズ分布が非対称だったり、平均よりも中央値が意思決定に直結する場合

---

## 3. Python 実装（`QuantileRegressor`）

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
    plt.plot(grid, pred, color=colors[tau], linewidth=2, label=f"分位 τ={tau}")
plt.plot(grid, ols_pred, color="#9467bd", linestyle="--", label="平均（OLS）")
plt.xlabel("入力 X")
plt.ylabel("出力 y")
plt.legend()
plt.tight_layout()
plt.show()
```

![quantile-regression block 1](/images/basic/regression/quantile-regression_block01.svg)

> 実際のプロットでは、各分位の線を重ねると「予測区間」が視覚的に伝わります。

---

## 4. ハイパーパラメータの注意

- `alpha` は L1 正則化の強さ。ゼロに近づけると精度は上がるが収束しにくい  
- `solver="highs"` は高速で安定。大規模データでは `interior-point` なども検討  
- 特徴量は標準化しておくと数値安定性が上がる  
- 分位ごとに独立したモデルを学習するので、必要な \\(\tau\\) を複数列挙して学習する

---

## 5. まとめ

- 分位点回帰は条件付き分布の幅や歪みを推定でき、リスク管理に有効  
- `QuantileRegressor` を使えば scikit-learn だけで実装可能  
- 複数の分位を学習して予測レンジを提示するのが実務での定番パターン
