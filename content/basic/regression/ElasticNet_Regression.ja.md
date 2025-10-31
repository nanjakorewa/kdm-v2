---
title: "Elastic Net 回帰"
pre: "2.1.5 "
weight: 5
title_suffix: "L1とL2正則化の長所を融合する"
---

{{% summary %}}
- Elastic Net は L1 と L2 の正則化を組み合わせ、疎性と安定性の両立を図る回帰手法である。
- 強く相関した特徴量をグループとして残しながら重要度を調整できるため、解釈しやすいモデルを得やすい。
- ハイパーパラメータ \\(\alpha\\) と `l1_ratio` を交差検証で調整すると、汎化性能とバイアスのバランスを取りやすい。
- 標準化や十分な反復回数の確保によって、数値最適化の安定性を高められる。
{{% /summary %}}

## 直感
ラッソ回帰は係数をゼロにして特徴量選択を実現しますが、相関の強い特徴量群からは 1 つだけを残して他を落としてしまうことがあります。リッジ回帰は係数を滑らかに縮めて安定させますが、係数がゼロにはなりません。Elastic Net は両者を組み合わせることで、相関した特徴量をまとめて残しつつ、不要な係数は 0 に近づける柔軟な挙動を実現します。

## 具体的な数式
Elastic Net の目的関数は

$$
\min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left( y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b) \right)^2 + \alpha \left( \rho \lVert \boldsymbol\beta \rVert_1 + (1 - \rho) \lVert \boldsymbol\beta \rVert_2^2 \right)
$$

で、\\(\alpha > 0\\) が正則化の強さ、\\(\rho \in [0,1]\\) (`l1_ratio`) が L1 と L2 の混合比率です。リッジとラッソの中間を連続的に探索できる点が特徴です。

## Pythonを用いた実験や説明
`ElasticNetCV` を用いて \\(\alpha\\) と `l1_ratio` を同時にチューニングし、係数と性能を確認します。

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def run_elastic_net_demo(
    n_samples: int = 500,
    n_features: int = 30,
    n_informative: int = 10,
    noise: float = 15.0,
    duplicate_features: int = 5,
    label_scatter_x: str = "predicted",
    label_scatter_y: str = "actual",
    label_scatter_title: str = "Predicted vs. actual",
    label_bar_title: str = "Top coefficients",
    label_bar_ylabel: str = "coefficient",
    top_n: int = 10,
) -> dict[str, float]:
    """Fit Elastic Net with CV, report metrics, and plot predictions/coefs.

    Args:
        n_samples: Number of generated samples.
        n_features: Total features before duplication.
        n_informative: Features with non-zero weights in the generator.
        noise: Standard deviation of noise added to targets.
        duplicate_features: Number of leading features to duplicate for correlation.
        label_scatter_x: Label for the scatter plot x-axis.
        label_scatter_y: Label for the scatter plot y-axis.
        label_scatter_title: Title for the scatter plot.
        label_bar_title: Title for the coefficient bar plot.
        label_bar_ylabel: Y-axis label for the coefficient plot.
        top_n: Number of largest-magnitude coefficients to display.

    Returns:
        Dictionary with training/test metrics for inspection.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(seed=123)

    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        random_state=123,
    )

    correlated = X[:, :duplicate_features] + rng.normal(
        scale=0.1, size=(X.shape[0], duplicate_features)
    )
    X = np.hstack([X, correlated])
    feature_names = np.array([f"x{i}" for i in range(X.shape[1])], dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    enet_cv = ElasticNetCV(
        l1_ratio=[0.2, 0.5, 0.7, 0.9, 0.95, 1.0],
        alphas=np.logspace(-3, 1, 30),
        cv=5,
        random_state=42,
        max_iter=5000,
    )
    enet_cv.fit(X_train, y_train)

    enet = ElasticNet(
        alpha=float(enet_cv.alpha_),
        l1_ratio=float(enet_cv.l1_ratio_),
        max_iter=5000,
        random_state=42,
    )
    enet.fit(X_train, y_train)

    train_pred = enet.predict(X_train)
    test_pred = enet.predict(X_test)

    metrics = {
        "best_alpha": float(enet_cv.alpha_),
        "best_l1_ratio": float(enet_cv.l1_ratio_),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "test_rmse": float(mean_squared_error(y_test, test_pred, squared=False)),
    }

    top_idx = np.argsort(np.abs(enet.coef_))[-top_n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_scatter, ax_bar = axes

    ax_scatter.scatter(test_pred, y_test, alpha=0.6, color="#1f77b4")
    ax_scatter.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
    )
    ax_scatter.set_title(label_scatter_title)
    ax_scatter.set_xlabel(label_scatter_x)
    ax_scatter.set_ylabel(label_scatter_y)

    ax_bar.bar(
        np.arange(top_n),
        enet.coef_[top_idx],
        color="#ff7f0e",
    )
    ax_bar.set_xticks(np.arange(top_n))
    ax_bar.set_xticklabels(feature_names[top_idx], rotation=45, ha="right")
    ax_bar.set_title(label_bar_title)
    ax_bar.set_ylabel(label_bar_ylabel)

    fig.tight_layout()
    plt.show()

    return metrics



metrics = run_elastic_net_demo(
    label_scatter_x="予測値",
    label_scatter_y="実測値",
    label_scatter_title="予測と実測の比較",
    label_bar_title="重要な係数",
    label_bar_ylabel="係数の大きさ",
)
print(f"最適な alpha: {metrics['best_alpha']:.4f}")
print(f"最適な l1_ratio: {metrics['best_l1_ratio']:.2f}")
print(f"訓練データのR^2: {metrics['train_r2']:.3f}")
print(f"テストデータのR^2: {metrics['test_r2']:.3f}")
print(f"テストRMSE: {metrics['test_rmse']:.3f}")

```

### 実行結果の読み方
- `ElasticNetCV` を使うと、L1 と L2 のバランスを含めた複数の候補を自動で評価できる。
- 相関の強い特徴量が複数残る場合でも、係数が似た大きさに調整されるため解釈しやすい。
- 収束が遅いときは特徴量を標準化したり `max_iter` を増やしたりすると改善する。

## 参考文献
{{% references %}}
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
<li>Friedman, J., Hastie, T., &amp; Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. <i>Journal of Statistical Software</i>, 33(1), 1–22.</li>
{{% /references %}}
