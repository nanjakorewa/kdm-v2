---
title: "多項式回帰 | 非線形パターンを線形モデルで捉える"
linkTitle: "多項式回帰"
seo_title: "多項式回帰 | 非線形パターンを線形モデルで捉える"
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
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def compare_polynomial_regression(
    n_samples: int = 200,
    degree: int = 3,
    noise_scale: float = 2.0,
    label_observations: str = "observations",
    label_true_curve: str = "true curve",
    label_linear: str = "linear regression",
    label_poly_template: str = "degree-{degree} polynomial",
) -> tuple[float, float]:
    """Fit linear vs. polynomial regression to a cubic trend and plot the results.

    Args:
        n_samples: Number of synthetic samples generated along the curve.
        degree: Polynomial degree used in the feature expansion.
        noise_scale: Standard deviation of the Gaussian noise added to targets.
        label_observations: Legend label for scatter observations.
        label_true_curve: Legend label for the underlying true curve.
        label_linear: Legend label for the linear regression fit.
        label_poly_template: Format string for the polynomial label.

    Returns:
        A tuple containing the mean-squared errors of (linear, polynomial) models.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(seed=42)

    x: np.ndarray = np.linspace(-3.0, 3.0, n_samples, dtype=float)
    y_true: np.ndarray = 0.5 * x**3 - 1.2 * x**2 + 2.0 * x + 1.5
    y_noisy: np.ndarray = y_true + rng.normal(scale=noise_scale, size=x.shape)

    X: np.ndarray = x[:, np.newaxis]

    linear_model = LinearRegression()
    linear_model.fit(X, y_noisy)
    poly_model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression(),
    )
    poly_model.fit(X, y_noisy)

    grid: np.ndarray = np.linspace(-3.5, 3.5, 300, dtype=float)[:, np.newaxis]
    linear_pred: np.ndarray = linear_model.predict(grid)
    poly_pred: np.ndarray = poly_model.predict(grid)
    true_curve: np.ndarray = (
        0.5 * grid.ravel()**3 - 1.2 * grid.ravel()**2 + 2.0 * grid.ravel() + 1.5
    )

    linear_mse: float = float(mean_squared_error(y_noisy, linear_model.predict(X)))
    poly_mse: float = float(mean_squared_error(y_noisy, poly_model.predict(X)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        X,
        y_noisy,
        s=20,
        color="#ff7f0e",
        alpha=0.6,
        label=label_observations,
    )
    ax.plot(
        grid,
        true_curve,
        color="#2ca02c",
        linewidth=2,
        label=label_true_curve,
    )
    ax.plot(
        grid,
        linear_pred,
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label=label_linear,
    )
    ax.plot(
        grid,
        poly_pred,
        color="#d62728",
        linewidth=2,
        label=label_poly_template.format(degree=degree),
    )
    ax.set_xlabel("input $x$")
    ax.set_ylabel("output $y$")
    ax.legend()
    fig.tight_layout()
    plt.show()

    return linear_mse, poly_mse



degree = 3
linear_mse, poly_mse = compare_polynomial_regression(
    degree=degree,
    label_observations="観測データ",
    label_true_curve="真の曲線",
    label_linear="線形回帰",
    label_poly_template="次数{degree}の多項式",
)
print(f"線形回帰のMSE: {linear_mse:.3f}")
print(f"次数{degree}の多項式回帰のMSE: {poly_mse:.3f}")

```


![polynomial-regression block 1](/images/basic/regression/polynomial-regression_block01_ja.png)

### 結果の読み方
- 単純な線形回帰では曲線の中央付近で大きくずれているが、多項式回帰は真の曲線に沿って学習できている。
- 多項式の次数を上げると学習データへの適合は良くなるが、外挿では不安定になりやすい。
- 正則化付き回帰（例: リッジ回帰）をパイプラインに組み込むと過学習を抑制しやすい。

## 参考文献
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
