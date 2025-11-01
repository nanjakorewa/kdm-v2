---
title: "Orthogonal Matching Pursuit (OMP) | 疎な係数を貪欲に選ぶ線形回帰"
linkTitle: "Orthogonal Matching Pursuit (OMP)"
seo_title: "Orthogonal Matching Pursuit (OMP) | 疎な係数を貪欲に選ぶ線形回帰"
pre: "2.1.12 "
weight: 12
title_suffix: "疎な係数を貪欲に選ぶ線形回帰"
---

{{% summary %}}
- Orthogonal Matching Pursuit (OMP) は残差と最も相関の高い特徴量を順番に選ぶ貪欲法で、疎な線形モデルを構築する。
- 選択済み特徴量に限定した最小二乗解を逐次的に求めるため、係数が直感的に解釈しやすい。
- 正則化パラメータではなく「残す特徴量数」を直接指定できる点が特徴で、疎性制御が明確になる。
- 特徴量の標準化や相関チェックを行うと、安定してスパース解を得られる。
{{% /summary %}}

## 直感
大量の特徴量の中から本当に効いているものだけを残したいとき、OMP は残差を最も減らす特徴量を 1 つずつ追加していきます。辞書式学習やスパースコーディングの基本アルゴリズムとしても知られており、少数の特徴量だけで説明したい場面で重宝します。

## 具体的な数式
初期残差を \\(\mathbf{r}^{(0)} = \mathbf{y}\\) とし、各ステップ \\(t\\) で以下を繰り返します。

1. 各特徴量 \\(\mathbf{x}_j\\) と残差 \\(\mathbf{r}^{(t-1)}\\) の内積を計算し、絶対値が最大の特徴量 \\(j\\) を選ぶ。
2. 選択済み特徴量集合 \\(\mathcal{A}_t\\) に \\(j\\) を追加する。
3. \\(\mathcal{A}_t\\) の特徴量だけを使って最小二乗解 \\(\hat{\boldsymbol\beta}_{\mathcal{A}_t}\\) を求める。
4. 新しい残差 \\(\mathbf{r}^{(t)} = \mathbf{y} - \mathbf{X}_{\mathcal{A}_t} \hat{\boldsymbol\beta}_{\mathcal{A}_t}\\) を計算する。

指定したステップ数に達するか残差が十分小さくなるまでこの操作を続けます。

## Pythonを用いた実験や説明
疎な真の係数を持つデータで OMP とラッソを比較します。

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error


def run_omp_vs_lasso(
    n_samples: int = 200,
    n_features: int = 40,
    sparsity: int = 4,
    noise_scale: float = 0.5,
    xlabel: str = "feature index",
    ylabel: str = "coefficient",
    label_true: str = "true",
    label_omp: str = "OMP",
    label_lasso: str = "Lasso",
    title: str | None = None,
) -> dict[str, object]:
    """Compare OMP and lasso on synthetic sparse regression data.

    Args:
        n_samples: Number of training samples to generate.
        n_features: Total number of features in the dictionary.
        sparsity: Count of non-zero coefficients in the ground truth.
        noise_scale: Standard deviation of Gaussian noise added to targets.
        xlabel: Label for the coefficient plot x-axis.
        ylabel: Label for the coefficient plot y-axis.
        label_true: Legend label for the ground-truth bars.
        label_omp: Legend label for the OMP bars.
        label_lasso: Legend label for the lasso bars.
        title: Optional title for the bar chart.

    Returns:
        Dictionary containing recovered supports and MSE values.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n_samples, n_features))
    true_coef = np.zeros(n_features)
    true_support = rng.choice(n_features, size=sparsity, replace=False)
    true_coef[true_support] = rng.normal(loc=0.0, scale=3.0, size=sparsity)
    y = X @ true_coef + rng.normal(scale=noise_scale, size=n_samples)

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
    omp.fit(X, y)
    lasso = Lasso(alpha=0.05)
    lasso.fit(X, y)

    omp_pred = omp.predict(X)
    lasso_pred = lasso.predict(X)

    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(n_features)
    ax.bar(indices - 0.3, true_coef, width=0.2, label=label_true, color="#2ca02c")
    ax.bar(indices, omp.coef_, width=0.2, label=label_omp, color="#1f77b4")
    ax.bar(indices + 0.3, lasso.coef_, width=0.2, label=label_lasso, color="#d62728")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "true_support": np.flatnonzero(true_coef),
        "omp_support": np.flatnonzero(omp.coef_),
        "lasso_support": np.flatnonzero(np.abs(lasso.coef_) > 1e-6),
        "omp_mse": float(mean_squared_error(y, omp_pred)),
        "lasso_mse": float(mean_squared_error(y, lasso_pred)),
    }



metrics = run_omp_vs_lasso(
    xlabel="特徴量インデックス",
    ylabel="係数",
    label_true="真の係数",
    label_omp="OMP",
    label_lasso="Lasso",
    title="OMP と Lasso の係数比較",
)
print("真のサポート:", metrics['true_support'])
print("OMP のサポート:", metrics['omp_support'])
print("Lasso のサポート:", metrics['lasso_support'])
print(f"OMP の MSE: {metrics['omp_mse']:.4f}")
print(f"Lasso の MSE: {metrics['lasso_mse']:.4f}")

```

### 実行結果の読み方
- `n_nonzero_coefs` を真の非ゼロ係数数に合わせると、OMP は対象となる特徴量を高確率で復元できる。
- ラッソと比べると、OMP は選ばれた特徴量以外の係数が完全に 0 になる。
- 特徴量間の相関が強い場合は選択順序が不安定になることがあるため、事前の標準化や特徴量設計が重要になる。

## 参考文献
{{% references %}}
<li>Pati, Y. C., Rezaiifar, R., &amp; Krishnaprasad, P. S. (1993). Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition. In <i>Conference Record of the Twenty-Seventh Asilomar Conference on Signals, Systems and Computers</i>.</li>
<li>Tropp, J. A. (2004). Greed is Good: Algorithmic Results for Sparse Approximation. <i>IEEE Transactions on Information Theory</i>, 50(10), 2231–2242.</li>
{{% /references %}}
