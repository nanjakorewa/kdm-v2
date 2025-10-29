---
title: "多項式回帰"
pre: "2.1.4 "
weight: 4
title_suffix: "非線形な関係を線形モデルで捉える"
---

{{< lead >}}
直線では捉えきれない曲線的なパターンも、特徴量を拡張すれば線形モデルで表現できます。本ページでは多項式回帰の考え方と実装手順を整理します。
{{< /lead >}}

---

## 1. いつ多項式回帰を使うのか

- 入力と出力の関係が滑らかな曲線になっているが、複雑なモデルはまだ使いたくないとき  
- 線形回帰の枠組み（係数に基づく解釈、解析解、学習の安定性）を保ったまま表現力を高めたいとき  
- 特徴量が 1〜数個で、関数の形を人間が理解したいとき

単純な線形回帰では \\(y = w_1 x + b\\) のように直線しか表現できませんが、入力を \\(x, x^2, x^3, \dots\\) と増やすことで、

$$
y = w_1 x + w_2 x^2 + \dots + w_d x^d + b
$$

のような多項式を学習できます。ここでも「係数の線形結合」を最適化しているため、線形回帰のノウハウがそのまま使えます。

---

## 2. 特徴量を多項式展開する仕組み

`scikit-learn` では `PolynomialFeatures` が特徴量の拡張を担います。例えば \\(x=(x_1, x_2)\\) に対して 2 次まで展開すると、

$$
\phi(x) = (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2)
$$

のようなベクトルになります。この \\(\phi(x)\\) を通常の線形回帰（またはリッジ回帰など）に入力するだけで多項式回帰が完成します。
特徴量の次元が指数的に増えるため、次数の上限や正則化の設定が重要です。

---

## 3. Python 実装例

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語ラベルを楽に描画できる
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# データ生成（3 次の真の関数 + ノイズ）
rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 200)
y_true = 0.5 * X**3 - 1.2 * X**2 + 2.0 * X + 1.5
y = y_true + rng.normal(scale=2.0, size=X.shape)

X = X[:, None]  # 2 次元配列に変換

# 1 次（通常の線形）と 3 次のモデルを比較
linear_model = LinearRegression().fit(X, y)
poly_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False),
                           LinearRegression()).fit(X, y)

grid = np.linspace(-3.5, 3.5, 300)[:, None]
linear_pred = linear_model.predict(grid)
poly_pred = poly_model.predict(grid)
true_curve = 0.5 * grid.ravel()**3 - 1.2 * grid.ravel()**2 + 2.0 * grid.ravel() + 1.5

print("線形回帰 MSE:", mean_squared_error(y, linear_model.predict(X)))
print("3次多項式回帰 MSE:", mean_squared_error(y, poly_model.predict(X)))

# 可視化（後で高解像度で描き直す前提）
plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=20, color="#ff7f0e", alpha=0.6, label="観測値")
plt.plot(grid, true_curve, color="#2ca02c", linewidth=2, label="真の関数")
plt.plot(grid, linear_pred, color="#1f77b4", linestyle="--", linewidth=2, label="線形回帰")
plt.plot(grid, poly_pred, color="#d62728", linewidth=2, label="3次多項式回帰")
plt.xlabel("入力 $x$")
plt.ylabel("出力 $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![polynomial-regression block 1](/images/basic/regression/polynomial-regression_block01.svg)

> 後でこの図を描き直すときは、学習済みモデルの曲線と真の関数を同じプロットに重ねると差が分かりやすくなります。

---

## 4. 次数選択と過学習

- 次数を上げるほどトレーニングデータにはよく当てはまるが、外挿では不安定になりやすい  
- 多項式の次数が高すぎると係数が非常に大きくなり数値的な不安定性が出る  
- `PolynomialFeatures` と `Ridge` を組み合わせた **多項式リッジ回帰** が実務では扱いやすい  
- 交差検証 (`GridSearchCV` や `ValidationCurve`) で次数と正則化係数を同時にチューニングするのが定番

---

## 5. まとめ

- 多項式回帰は「線形モデル + 特徴量拡張」で曲線パターンを学習できる  
- `PolynomialFeatures` → `LinearRegression` のパイプラインで実装がシンプル  
- 過学習しやすくなるため次数と正則化のバランスが重要  
- まずは 2〜3 次から試し、交差検証で最適化するのがおすすめです

---
