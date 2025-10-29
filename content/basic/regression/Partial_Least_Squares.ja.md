---
title: "PLS 回帰（Partial Least Squares）"
pre: "2.1.9 "
weight: 9
title_suffix: "目的変数を意識した潜在因子で回帰する"
---

{{< lead >}}
主成分回帰が「説明変数の分散」を最大化するのに対して、PLS は「説明変数と目的変数の共分散」を最大化する潜在因子を学習します。目的変数に効く方向を見逃しにくいのが特徴です。
{{< /lead >}}

---

## 1. PLS の直感

- 目的：説明変数 \\(X\\) と目的変数 \\(Y\\) の両方に情報を持つ潜在ベクトル（スコア）を抽出  
- 各潜在ベクトルについて、\\(X\\) 側・\\(Y\\) 側の荷重（loading）を推定し、逐次的に残差を更新  
- 少数の潜在因子だけで回帰するため、高次元・多重共線性データでも安定

---

## 2. PCA との違い

| 観点 | PCA | PLS |
|------|-----|-----|
| 最適化目標 | \\(X\\) の分散 | \\(X\\) と \\(Y\\) の共分散 |
| 教師あり/なし | 教師なし | 教師あり |
| 目的変数を意識？ | しない | する |
| 向いている場面 | ノイズ除去、可視化 | 回帰性能重視、因果探索 |

---

## 3. Python 実装（`PLSRegression`）

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
X = data["data"]          # トレーニングメニュー（身長や体重など）
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

![partial-least-squares block 1](/images/basic/regression/partial-least-squares_block01.svg)

---

## 4. 実務でのポイント

- `n_components` は PCA 同様に CV で決める。多すぎるとノイズが戻ってしまう  
- 多目的変数（多出力）の場合でも、PLS は同時に扱える（`y` を多列で渡す）  
- Loading 行列を可視化すると「どの特徴の組み合わせが効いているか」を説明しやすい  
- スケーリングは必須。単位の差が共分散最大化に影響するため

---

## 5. PCR vs PLS の使い分け

- **PCR**: 目的変数がかなりノイズに埋もれている、まずはデータ構造を把握したいとき  
- **PLS**: 目的変数の予測精度を優先し、説明変数との関係を直接モデリングしたいとき  
- 実務では両方試し、CV で良い方を採用するのが無難

---

## 6. まとめ

- PLS 回帰は「目的変数を意識した圧縮＋回帰」で、多重共線性と高次元に強い  
- `PLSRegression` を `Pipeline` に組み込めば、スケーリングや CV も簡単  
- Loading を解釈することで、データに潜む共変動構造を把握できる
