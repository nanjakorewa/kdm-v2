---
title: "Orthogonal Matching Pursuit（OMP）"
pre: "2.1.12 "
weight: 12
title_suffix: "疎な係数を貪欲に選ぶ線形回帰"
---

{{% summary %}}
- Orthogonal Matching Pursuit（OMP）は残差と最も相関の高い特徴量を順番に選び、疎な線形モデルを構築する貪欲法。
- 各ステップで選択済み特徴量のみに限定した最小二乗解を求めるため、係数が直感的に解釈しやすい。
- ラッソのような正則化パラメータではなく「残す特徴量数」を直接指定できるのが特徴。
- 特徴量が強く相関する場合は選択順序に左右されやすいため、標準化や相関チェックが重要。
{{% /summary %}}

## 直感
大量の特徴量の中から本当に効いているものだけを選びたいとき、OMP は残差を最も減らす特徴量を一つずつ追加していきます。辞書式学習やスパースコーディングの基本アルゴリズムとしても知られ、疎な係数ベクトルを得たい場面で重宝します。

## 具体的な数式
初期残差を \\(\mathbf{r}^{(0)} = \mathbf{y}\\) とし、各ステップ \\(t\\) で以下を行います。

1. \\( \mathbf{x}_j\\) と残差の内積が最大の特徴量 \\(j\\) を選択。
2. 選択済み特徴量集合 \\(\mathcal{A}_t\\) に \\(j\\) を追加。
3. \\(\mathcal{A}_t\\) の特徴量だけを使って最小二乗解 \\(\hat{\boldsymbol\beta}_{\mathcal{A}_t}\\) を求める。
4. 新しい残差 \\(\mathbf{r}^{(t)} = \mathbf{y} - \mathbf{X}_{\mathcal{A}_t} \hat{\boldsymbol\beta}_{\mathcal{A}_t}\\) を計算。

これを指定したステップ数か残差が小さくなるまで繰り返します。

## Pythonを用いた実験や説明
疎な真の係数を持つデータで OMP とラッソを比較する例です。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
n_samples, n_features = 200, 40
X = rng.normal(size=(n_samples, n_features))
true_coef = np.zeros(n_features)
true_support = [1, 5, 12, 33]
true_coef[true_support] = [2.5, -1.5, 3.0, 1.0]
y = X @ true_coef + rng.normal(scale=0.5, size=n_samples)

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=4).fit(X, y)
lasso = Lasso(alpha=0.05).fit(X, y)

print("OMP 係数�E�非ゼロ�E�E", np.flatnonzero(omp.coef_))
print("Lasso 係数�E�非ゼロ�E�E", np.flatnonzero(np.abs(lasso.coef_) > 1e-6))

print("OMP MSE:", mean_squared_error(y, omp.predict(X)))
print("Lasso MSE:", mean_squared_error(y, lasso.predict(X)))

coef_df = pd.DataFrame({
    "true": true_coef,
    "omp": omp.coef_,
    "lasso": lasso.coef_,
})
print(coef_df.head(10))
```

### 実行結果の読み方
- `n_nonzero_coefs` を真の非ゼロ係数数に合わせると、OMP は対象となる特徴量を高確率で復元できる。
- ラッソと比較すると、OMP は選ばれた特徴量以外の係数が完全に 0 になる。
- 相関の強い特徴量が存在すると、選択がぶれる可能性があるため注意が必要。

## 参考文献
{{% references %}}
<li>Pati, Y. C., Rezaiifar, R., &amp; Krishnaprasad, P. S. (1993). Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition. In <i>Conference Record of The Twenty-Seventh Asilomar Conference on Signals, Systems and Computers</i>.</li>
<li>Tropp, J. A. (2004). Greed is Good: Algorithmic Results for Sparse Approximation. <i>IEEE Transactions on Information Theory</i>, 50(10), 2231–2242.</li>
{{% /references %}}
