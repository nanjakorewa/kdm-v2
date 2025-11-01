---
title: "分位点回帰 (Quantile Regression) | 条件付き分布の輪郭を推定する"
linkTitle: "分位点回帰 (Quantile Regression)"
seo_title: "分位点回帰 (Quantile Regression) | 条件付き分布の輪郭を推定する"
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
残差を \\(r = y - \hat{y}\\)、分位点を \\(\tau \in (0,1)\\) とすると、ピンボール損失は

$$
L_\tau(r) =
\begin{cases}
\tau \, r & (r \ge 0) \\
(\tau - 1) r & (r < 0)
\end{cases}
$$

で定義されます。この損失を最小化すると、\\(\tau\\) 分位点に対応する線形予測子が得られます。例えば \\(\tau=0.5\\) なら中央値回帰になり、絶対値損失によるロバスト回帰と同じ振る舞いをします。

## Pythonを用いた実験や説明
`QuantileRegressor` を使って 0.1・0.5・0.9 分位点を推定し、線形回帰と比較します。

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_quantile_regression_demo(
    taus: tuple[float, ...] = (0.1, 0.5, 0.9),
    n_samples: int = 400,
    xlabel: str = "input x",
    ylabel: str = "output y",
    label_observations: str = "observations",
    label_mean: str = "mean (OLS)",
    label_template: str = "quantile τ={tau}",
    title: str | None = None,
) -> dict[float, tuple[float, float]]:
    """Fit quantile regressors alongside OLS and plot the conditional bands.

    Args:
        taus: Quantile levels to fit (each in (0, 1)).
        n_samples: Number of synthetic observations to generate.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        label_observations: Legend label for the scatter plot.
        label_mean: Legend label for the OLS line.
        label_template: Format string for quantile labels.
        title: Optional title for the plot.

    Returns:
        Mapping of quantile level to (min prediction, max prediction).
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(123)

    x_values: np.ndarray = np.linspace(0.0, 10.0, n_samples, dtype=float)
    noise: np.ndarray = rng.gamma(shape=2.0, scale=1.0, size=n_samples) - 2.0
    y_values: np.ndarray = 1.5 * x_values + 5.0 + noise
    X: np.ndarray = x_values[:, np.newaxis]

    quantile_models: dict[float, make_pipeline] = {}
    for tau in taus:
        model = make_pipeline(
            StandardScaler(with_mean=True),
            QuantileRegressor(alpha=0.001, quantile=float(tau), solver="highs"),
        )
        model.fit(X, y_values)
        quantile_models[tau] = model

    ols = LinearRegression()
    ols.fit(X, y_values)

    grid: np.ndarray = np.linspace(0.0, 10.0, 200, dtype=float)[:, np.newaxis]
    preds = {tau: model.predict(grid) for tau, model in quantile_models.items()}
    ols_pred: np.ndarray = ols.predict(grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X, y_values, s=15, alpha=0.4, label=label_observations)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])
    for idx, tau in enumerate(taus):
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(
            grid,
            preds[tau],
            color=color,
            linewidth=2,
            label=label_template.format(tau=tau),
        )

    ax.plot(grid, ols_pred, color="#9467bd", linestyle="--", label=label_mean)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    summary: dict[float, tuple[float, float]] = {
        tau: (float(pred.min()), float(pred.max())) for tau, pred in preds.items()
    }
    return summary



summary = run_quantile_regression_demo(
    xlabel="入力 x",
    ylabel="出力 y",
    label_observations="観測データ",
    label_mean="平均 (OLS)",
    label_template="分位点 τ={tau}",
    title="分位点回帰による条件分布",
)
for tau, (ymin, ymax) in summary.items():
    print(f"τ={tau:.1f}: 予測最小値 {ymin:.2f}, 予測最大値 {ymax:.2f}")

```

![quantile-regression block 1](/images/basic/regression/quantile-regression_block01_ja.png)

### 実行結果の読み方
- 分位点ごとに異なる直線が得られ、データの上下方向のばらつきを表現できる。
- 平均を表す最小二乗法と比べ、片側に長いノイズにも柔軟に対応している。
- 分位点を複数組み合わせると予測区間が得られ、意思決定に必要な情報を提示できる。

## 参考文献
{{% references %}}
<li>Koenker, R., &amp; Bassett, G. (1978). Regression Quantiles. <i>Econometrica</i>, 46(1), 33–50.</li>
<li>Koenker, R. (2005). <i>Quantile Regression</i>. Cambridge University Press.</li>
{{% /references %}}
