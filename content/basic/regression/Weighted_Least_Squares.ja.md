---
title: "加重最小二乗法 (WLS) | ばらつきが異なる観測を適切に扱う"
linkTitle: "加重最小二乗法 (WLS)"
seo_title: "加重最小二乗法 (WLS) | ばらつきが異なる観測を適切に扱う"
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
観測ごとに重み \\(w_i > 0\\) を与えて目的関数

$$
L(\boldsymbol\beta, b) = \sum_{i=1}^{n} w_i \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2
$$

を最小化します。理想的には \\(w_i \propto 1/\sigma_i^2\\)（分散の逆数）と設定し、信頼度の高いデータ点ほど重みを大きくします。

## Pythonを用いた実験や説明
ノイズレベルが区間で異なるデータに WLS を適用する例です。

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def run_weighted_vs_ols(
    n_samples: int = 200,
    threshold: float = 5.0,
    low_noise: float = 0.5,
    high_noise: float = 2.5,
    xlabel: str = "input $",
    ylabel: str = "output $",
    label_scatter: str = "observations (color=noise)",
    label_truth: str = "true line",
    label_ols: str = "OLS",
    label_wls: str = "WLS",
    title: str | None = None,
) -> dict[str, float]:
    """Compare OLS and weighted least squares on heteroscedastic data.

    Args:
        n_samples: Number of evenly spaced samples to generate.
        threshold: Breakpoint separating low- and high-noise regions.
        low_noise: Noise scale for the lower region.
        high_noise: Noise scale for the higher region.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        label_scatter: Legend label for the colored scatter plot.
        label_truth: Legend label for the ground-truth line.
        label_ols: Legend label for the OLS fit.
        label_wls: Legend label for the WLS fit.
        title: Optional title for the plot.

    Returns:
        Dictionary with slopes and intercepts of both fits.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(7)

    X_vals: np.ndarray = np.linspace(0.0, 10.0, n_samples, dtype=float)
    true_y: np.ndarray = 1.2 * X_vals + 3.0

    noise_scale = np.where(X_vals < threshold, low_noise, high_noise)
    y_noisy = true_y + rng.normal(scale=noise_scale)

    weights = 1.0 / (noise_scale**2)
    X = X_vals[:, np.newaxis]

    ols = LinearRegression()
    ols.fit(X, y_noisy)

    wls = LinearRegression()
    wls.fit(X, y_noisy, sample_weight=weights)

    grid = np.linspace(0.0, 10.0, 200, dtype=float)[:, np.newaxis]
    ols_pred = ols.predict(grid)
    wls_pred = wls.predict(grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        X,
        y_noisy,
        c=noise_scale,
        cmap="coolwarm",
        s=25,
        label=label_scatter,
    )
    ax.plot(grid, 1.2 * grid.ravel() + 3.0, color="#2ca02c", label=label_truth)
    ax.plot(grid, ols_pred, color="#1f77b4", linestyle="--", linewidth=2, label=label_ols)
    ax.plot(grid, wls_pred, color="#d62728", linewidth=2, label=label_wls)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "ols_slope": float(ols.coef_[0]),
        "ols_intercept": float(ols.intercept_),
        "wls_slope": float(wls.coef_[0]),
        "wls_intercept": float(wls.intercept_),
    }



metrics = run_weighted_vs_ols(
    xlabel="入力 $",
    ylabel="出力 $",
    label_scatter="観測値 (色=ノイズ)",
    label_truth="真の直線",
    label_ols="OLS",
    label_wls="WLS",
    title="重み付き最小二乗法とOLSの比較",
)
print(f"OLS の傾き: {metrics['ols_slope']:.3f}, 切片: {metrics['ols_intercept']:.3f}")
print(f"WLS の傾き: {metrics['wls_slope']:.3f}, 切片: {metrics['wls_intercept']:.3f}")

```

![weighted-least-squares block 1](/images/basic/regression/weighted-least-squares_block01_ja.png)

### 実行結果の読み方
- `weights` を与えることでノイズの小さい区間がより重視され、真の直線に近い推定になる。
- OLS の直線はノイズの大きい区間に引っ張られ、傾きが過小評価されやすい。
- 重みを適切に設定することが性能改善の鍵となる。

## 参考文献
{{% references %}}
<li>Carroll, R. J., &amp; Ruppert, D. (1988). <i>Transformation and Weighting in Regression</i>. Chapman &amp; Hall.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
