---
title: "加重最小二乗法 (WLS)"
pre: "2.1.11 "
weight: 11
title_suffix: "ばらつきが異なる観測を適切に扱う"
---

{{% summary %}}
- 加重最小二乗法 (WLS) は観測ごとの信頼度に応じて重みを割り当て、異質なノイズを持つデータでも妥当な回帰直線を推定する。
- 重みを二乗誤差に掛けることで、分散の小さい観測ほど強く反映され、ノイズの大きい点に引きずられにくくなる。
- 標準の `LinearRegression` に `sample_weight` を指定すれば WLS を実行できる。
- 重みは既知の分散、残差の推定、ドメイン知識など複数の観点を組み合わせて設計する。
{{% /summary %}}

## 直感
通常の最小二乗法はすべての観測が同じ信頼度を持つと仮定します。しかし実務では、センサー性能や測定回数によって精度が大きく異なることがよくあります。WLS は「信頼できる点の意見をより尊重する」ように重みを付け直し、線形回帰の枠組みで異質なデータを扱います。

## 具体的な数式
観測ごとに重み \(w_i > 0\) を与えて目的関数

$$
L(\boldsymbol\beta, b) = \sum_{i=1}^{n} w_i \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2
$$

を最小化します。理想的には \(w_i \propto 1/\sigma_i^2\)（分散の逆数）と設定し、信頼度の高いデータ点ほど重みを大きくします。

## Pythonを用いた実験や説明
ノイズレベルが区間で異なるデータに WLS を適用する例です。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(7)
n_samples = 200
X = np.linspace(0, 10, n_samples)
true_y = 1.2 * X + 3

# 区間ごとにノイズ水準を変える（異質な誤差分散を想定）
noise_scale = np.where(X < 5, 0.5, 2.5)
y = true_y + rng.normal(scale=noise_scale)

weights = 1.0 / (noise_scale ** 2)  # 誤差分散の逆数を重みとみなす
X = X[:, None]

ols = LinearRegression().fit(X, y)
wls = LinearRegression().fit(X, y, sample_weight=weights)

grid = np.linspace(0, 10, 200)[:, None]
ols_pred = ols.predict(grid)
wls_pred = wls.predict(grid)

print("OLS 傾き:", ols.coef_[0], " 切片:", ols.intercept_)
print("WLS 傾き:", wls.coef_[0], " 切片:", wls.intercept_)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, c=noise_scale, cmap="coolwarm", s=25, label="観測値（色=ノイズ）")
plt.plot(grid, 1.2 * grid.ravel() + 3, color="#2ca02c", label="真の直線")
plt.plot(grid, ols_pred, color="#1f77b4", linestyle="--", linewidth=2, label="OLS")
plt.plot(grid, wls_pred, color="#d62728", linewidth=2, label="WLS")
plt.xlabel("入力 $x$")
plt.ylabel("出力 $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![weighted-least-squares block 1](/images/basic/regression/weighted-least-squares_block01.svg)

### 実行結果の読み方
- `weights` を与えることでノイズの小さい区間がより重視され、真の直線に近い推定になる。
- OLS の直線はノイズの大きい区間に引っ張られ、傾きが過小評価されやすい。
- 重みを適切に設定することが性能改善の鍵となる。

## 参考文献
{{% references %}}
<li>Carroll, R. J., &amp; Ruppert, D. (1988). <i>Transformation and Weighting in Regression</i>. Chapman &amp; Hall.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
