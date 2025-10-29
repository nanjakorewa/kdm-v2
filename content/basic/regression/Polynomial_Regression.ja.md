---
title: "多項式回帰"
pre: "2.1.4 "
weight: 4
title_suffix: "非線形パターンを線形モデルで捉える"
---

{{% summary %}}
- 多項式回帰は特徴量を冪乗展開して線形回帰に渡すことで、非線形な関係も扱えるようにする手法。
- モデルは係数の線形結合のままなので、解析解や解釈のしやすさといった線形回帰の利点を保てる。
- 次数を上げるほど表現力は増すが過学習しやすく、正則化や交差検証による制御が重要。
- 事前に特徴量を標準化し、次数やペナルティ強度を丁寧に選ぶことで安定した予測が得られる。
{{% /summary %}}

## 直感
直線では説明しきれない滑らかな曲線や山なりのパターンも、入力を多項式で展開してあげれば線形モデルの枠内で表現できます。単回帰なら \\(x, x^2, x^3, \dots\\) を新しい特徴量として追加し、重回帰ならそれぞれの変数の冪や交差項を含めます。

## 具体的な数式
入力ベクトル \\(\mathbf{x} = (x_1, \dots, x_m)\\) に対して次数 \\(d\\) の多項式基底 \\(\phi(\mathbf{x})\\) を作り、線形回帰を行います。例えば \\(m = 2, d = 2\\) なら

$$
\phi(\mathbf{x}) = (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2)
$$

となり、モデルは

$$
y = \mathbf{w}^\top \phi(\mathbf{x})
$$

で表されます。次数 \\(d\\) を増やすと指数的に特徴量が増えるため、実務では 2～3 次程度から試し、必要に応じて正則化（リッジ回帰など）を併用します。

## Pythonを用いた実験や説明
以下は 3 次の多項式特徴量を追加した線形回帰モデルで、曲線的なデータを学習する例です。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語ラベルを楽に描画できる
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# チE�Eタ生�E�E�E 次の真�E関数 + ノイズ�E�E
rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 200)
y_true = 0.5 * X**3 - 1.2 * X**2 + 2.0 * X + 1.5
y = y_true + rng.normal(scale=2.0, size=X.shape)

X = X[:, None]  # 2 次允E�E列に変換

# 1 次�E�通常の線形�E�と 3 次のモチE��を比輁E
linear_model = LinearRegression().fit(X, y)
poly_model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False),
                           LinearRegression()).fit(X, y)

grid = np.linspace(-3.5, 3.5, 300)[:, None]
linear_pred = linear_model.predict(grid)
poly_pred = poly_model.predict(grid)
true_curve = 0.5 * grid.ravel()**3 - 1.2 * grid.ravel()**2 + 2.0 * grid.ravel() + 1.5

print("線形回帰 MSE:", mean_squared_error(y, linear_model.predict(X)))
print("3次多頁E��回帰 MSE:", mean_squared_error(y, poly_model.predict(X)))

# 可視化�E�後で高解像度で描き直す前提！E
plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=20, color="#ff7f0e", alpha=0.6, label="観測値")
plt.plot(grid, true_curve, color="#2ca02c", linewidth=2, label="真�E関数")
plt.plot(grid, linear_pred, color="#1f77b4", linestyle="--", linewidth=2, label="線形回帰")
plt.plot(grid, poly_pred, color="#d62728", linewidth=2, label="3次多頁E��回帰")
plt.xlabel("入劁E$x$")
plt.ylabel("出劁E$y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![polynomial-regression block 1](/images/basic/regression/polynomial-regression_block01.svg)

### 結果の読み方
- 単純な線形回帰では曲線の中央付近で大きくずれているが、多項式回帰は真の曲線に沿って学習できている。
- 多項式の次数を上げると学習データへの適合は良くなるが、外挿では不安定になりやすい。
- 正則化付き回帰（例: リッジ回帰）をパイプラインに組み込むと過学習を抑制しやすい。

## 参考文献
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
