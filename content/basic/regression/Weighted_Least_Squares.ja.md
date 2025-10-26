---
title: "加重最小二乗法（WLS）"
pre: "2.1.10 "
weight: 10
title_suffix: "ばらつきの異なる観測を適切に扱う"
---

{{< lead >}}
測定精度やサンプルサイズがデータ点ごとに異なるとき、通常の最小二乗法ではノイズの大きい点に引っ張られてしまいます。加重最小二乗法（Weighted Least Squares）は信頼度に応じて重みを付け、より妥当な直線を求める手法です。
{{< /lead >}}

---

## 1. 何が問題になるのか

- 通常の線形回帰は誤差が同じ分散（等分散）だと仮定  
- 実務では「観測回数が多いデータほど信頼できる」「センサーごとにノイズレベルが違う」といった **異分散性** が頻発  
- 等分散を仮定したままフィットすると、ばらつきの大きい点が平均化され、重要な傾向が埋もれてしまう

---

## 2. 加重最小二乗法の数式

各サンプルに重み $w_i > 0$ を付与し、次の目的関数を最小化します。

$$
L(\boldsymbol\\beta, b) = \sum_{i=1}^{n} w_i \, \\bigl(y_i - (\boldsymbol\\beta^\\top \\mathbf{x}_i + b)\\bigr)^2
$$

重みは分散の逆数（$w_i \\propto 1 / \\sigma_i^2$）に設定するのが理想です。信頼性の高い観測に大きな重みを与えるイメージです。

---

## 3. Python 実装例（`sample_weight` を活用）

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(7)
n_samples = 200
X = np.linspace(0, 10, n_samples)
true_y = 1.2 * X + 3

# 区間ごとにノイズ水準を変える（異分散）
noise_scale = np.where(X < 5, 0.5, 2.5)
y = true_y + rng.normal(scale=noise_scale)

weights = 1.0 / (noise_scale ** 2)  # 逆分散を重みとみなす
X = X[:, None]

ols = LinearRegression().fit(X, y)
wls = LinearRegression().fit(X, y, sample_weight=weights)

grid = np.linspace(0, 10, 200)[:, None]
ols_pred = ols.predict(grid)
wls_pred = wls.predict(grid)

print("OLS 傾き:", ols.coef_[0], " 切片:", ols.intercept_)
print("WLS 傾き:", wls.coef_[0], " 切片:", wls.intercept_)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, c=noise_scale, cmap="coolwarm", s=25, label="観測（色=ノイズ）")
plt.plot(grid, true_y := 1.2 * grid.ravel() + 3, color="#2ca02c", label="真の直線")
plt.plot(grid, ols_pred, color="#1f77b4", linestyle="--", linewidth=2, label="OLS")
plt.plot(grid, wls_pred, color="#d62728", linewidth=2, label="WLS")
plt.xlabel("入力 \\(x\\)")
plt.ylabel("出力 \\(y\\)")
plt.legend()
plt.tight_layout()
plt.show()
```

![ダミー図: 加重最小二乗法と通常回帰の比較](/images/placeholder_regression.png)

---

## 4. 重みの決め方

- **既知の分散**: センサー仕様やサンプル数などから直接計算  
- **残差から推定**: 一度 OLS を当てて残差の絶対値や平方をもとに推定 → IRLS（反復再重み付け最小二乗）  
- **ビジネスルール**: 例えば「最新データを重視」するため指数的に重みを減衰させる

---

## 5. いつ使うべき？

- アンケートやログなど、サンプルごとの母集団サイズが違う  
- センサーが複数あり、性能がバラバラ  
- 時系列で時間が進むにつれてノイズが増大する

---

## 6. まとめ

- 加重最小二乗法は、異分散データで平均的な回帰直線が歪む問題を緩和できる  
- `LinearRegression(..., sample_weight=...)` だけでも簡単に試せる  
- 重みの設定が結果を大きく左右するため、ドメイン知識と統計的推定を組み合わせると精度が上がります

---
