---
title: "リッジ回帰とラッソ回帰"
pre: "2.1.2 "
weight: 2
title_suffix: "正則化で汎化性能を高める"
---

{{% summary %}}
- リッジ回帰は L2 正則化によって係数を滑らかに縮小し、多重共線性があっても安定した推定を行う。
- ラッソ回帰は L1 正則化により一部の係数を 0 にし、特徴量選択とモデルの解釈性向上に寄与する。
- 正則化強度 \(\alpha\) を調整することで、学習データへの適合と汎化性能のバランスを取れる。
- 標準化と交差検証を組み合わせると、過学習を抑えながら最適なハイパーパラメータを選択しやすい。
{{% /summary %}}

## 直感
最小二乗法は外れ値や説明変数間の強い相関に弱く、係数が極端な値になってしまうことがあります。リッジ回帰とラッソ回帰は、係数が大きくなりすぎないようペナルティを加えることで、予測性能と解釈性のバランスをとるための拡張です。リッジは「全体を滑らかに縮める」イメージ、ラッソは「不要な係数を大胆に 0 にする」イメージで捉えると理解しやすくなります。

## 具体的な数式
リッジ回帰とラッソ回帰の目的関数は、最小二乗誤差にそれぞれ L2・L1 ノルムのペナルティを加えた形になります。

- **リッジ回帰**
  $$
  \min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2 + \alpha \lVert \boldsymbol\beta \rVert_2^2
  $$

- **ラッソ回帰**
  $$
  \min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2 + \alpha \lVert \boldsymbol\beta \rVert_1
  $$

\(\alpha\) を大きくすると係数がより強く抑えられます。ラッソでは\(\alpha\) がある閾値を超えると係数が完全に 0 になるため、疎なモデルを得たい場合に有効です。

## Pythonを用いた実験や説明
以下は重回帰データにリッジ・ラッソ・通常の線形回帰を適用し、係数の大きさと汎化性能を比較する例です。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# �p�����[�^
n_samples = 200
n_features = 10
n_informative = 3
rng = np.random.default_rng(42)

# �^�̌W���i�X�p�[�X�j
coef = np.zeros(n_features)
coef[:n_informative] = rng.normal(loc=3.0, scale=1.0, size=n_informative)

X = rng.normal(size=(n_samples, n_features))
y = X @ coef + rng.normal(scale=5.0, size=n_samples)

linear = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
ridge = make_pipeline(StandardScaler(with_mean=False), RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)).fit(X, y)
lasso = make_pipeline(StandardScaler(with_mean=False), LassoCV(alphas=np.logspace(-3, 1, 9), cv=5, max_iter=50_000)).fit(X, y)

models = {
    "Linear": linear,
    "Ridge": ridge,
    "Lasso": lasso,
}

# �W������
indices = np.arange(n_features)
width = 0.25
plt.figure(figsize=(10, 4))
plt.bar(indices - width, np.abs(linear[-1].coef_), width=width, label="Linear")
plt.bar(indices, np.abs(ridge[-1].coef_), width=width, label="Ridge")
plt.bar(indices + width, np.abs(lasso[-1].coef_), width=width, label="Lasso")
plt.xlabel("�����ʃC���f�b�N�X")
plt.ylabel("|�W��|")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# �������؃X�R�A
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name:>6}: R^2 = {scores.mean():.3f} �} {scores.std():.3f}")
```

### 実行結果の読み方
- リッジはすべての係数を緩やかに縮め、多重共線性があっても安定して推定できます。
- ラッソは一部の係数を 0 にし、重要な特徴量だけを残す傾向があります。
- 正則化パラメータは交差検証で選ぶと、過学習とバイアスのバランスを取りやすくなります。

## 参考文献
{{% references %}}
<li>Hoerl, A. E., &amp; Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. <i>Technometrics</i>, 12(1), 55–67.</li>
<li>Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. <i>Journal of the Royal Statistical Society: Series B</i>, 58(1), 267–288.</li>
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
{{% /references %}}
