---
title: "PLS 回帰 (Partial Least Squares)"
pre: "2.1.9 "
weight: 9
title_suffix: "目標を意識した潜在因子で回帰する"
---

{{% summary %}}
- PLS 回帰は説明変数と目的変数の共分散を最大化する潜在因子を抽出し、その上で回帰を行う教師あり次元圧縮法である。
- PCA と異なり目的変数の情報を反映した軸を学習するため、予測性能を保ちながら次元削減できる。
- 潜在因子数を調整すると、多重共線性が強い場合でも安定したモデルを構築できる。
- ローディングを可視化すると、どの特徴量の組み合わせが目的変数と強く関係しているかを説明できる。
{{% /summary %}}

## 直感
主成分回帰は説明変数の分散だけを基準に軸を決めるため、目的変数に効く方向が削られてしまうことがあります。PLS 回帰では説明変数と目的変数の両方を見ながら潜在因子を構成し、予測に有効な情報を残したまま次元を圧縮します。その結果、少ない因子でも予測性能を維持しやすくなります。

## 具体的な数式
説明変数行列 \(\mathbf{X}\) と目的変数ベクトル \(\mathbf{y}\) に対し、潜在スコア \(\mathbf{t} = \mathbf{X} \mathbf{w}\) と \(\mathbf{u} = \mathbf{y} c\) を交互に更新しながら、共分散 \(\mathbf{t}^\top \mathbf{u}\) が最大となる \(\mathbf{w}, c\) を求めます。この操作を繰り返し、得られた潜在因子上で線形回帰を行います。潜在因子数を \(k\) とすると、最終的な回帰モデルは

$$
\hat{y} = \mathbf{t} \boldsymbol{b} + b_0
$$

の形になります。潜在因子数はクロスバリデーションなどで選ぶのが一般的です。

## Pythonを用いた実験や説明
運動データセットで潜在因子数ごとの性能を比較します。

```python
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_linnerud
import matplotlib.pyplot as plt
import japanize_matplotlib

data = load_linnerud()
X = data["data"]          # トレーニングメニューや体格情報
y = data["target"][:, 0]  # ここではチンニング回数だけを予測

components = range(1, min(X.shape[1], 6) + 1)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

scores = []
for k in components:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("pls", PLSRegression(n_components=k)),
    ])
    score = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error").mean()
    scores.append(score)

best_k = components[int(np.argmax(scores))]
print("最適な潜在因子数:", best_k)

best_model = Pipeline([
    ("scale", StandardScaler()),
    ("pls", PLSRegression(n_components=best_k)),
]).fit(X, y)

print("X 側 loading:\n", best_model["pls"].x_loadings_)
print("Y 側 loading:\n", best_model["pls"].y_loadings_)

plt.figure(figsize=(8, 4))
plt.plot(components, [-s for s in scores], marker="o")
plt.axvline(best_k, color="red", linestyle="--")
plt.xlabel("潜在因子数")
plt.ylabel("CV MSE (小さいほど良い)")
plt.tight_layout()
plt.show()
```

![partial-least-squares block 1](/images/basic/regression/partial-least-squares_block01_ja.png)

### 実行結果の読み方
- 潜在因子数を増やすにつれて CV MSE が下がり、最小となる点を過ぎると悪化し始める。
- `x_loadings_` と `y_loadings_` を確認すると、どの特徴量が潜在因子に寄与しているかが分かる。
- 標準化を行うことで単位の異なる特徴量でもバランスの良い潜在因子が得られる。

## 参考文献
{{% references %}}
<li>Wold, H. (1975). Soft Modelling by Latent Variables: The Non-Linear Iterative Partial Least Squares (NIPALS) Approach. In <i>Perspectives in Probability and Statistics</i>. Academic Press.</li>
<li>Geladi, P., &amp; Kowalski, B. R. (1986). Partial Least-Squares Regression: A Tutorial. <i>Analytica Chimica Acta</i>, 185, 1–17.</li>
{{% /references %}}
