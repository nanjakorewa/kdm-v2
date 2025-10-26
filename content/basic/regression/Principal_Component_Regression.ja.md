---
title: "主成分回帰（PCR）"
pre: "2.1.8 "
weight: 8
title_suffix: "多重共線性を緩和する次元圧縮＋回帰"
---

{{< lead >}}
特徴量が多く相関も強いと、係数が不安定になります。主成分回帰（PCR）は先に PCA で次元圧縮し、情報を保ったまま線形回帰する手法です。
{{< /lead >}}

---

## 1. なぜ PCR が効くのか

- PCA で **相関の高い軸をまとめ、ノイズ軸を切り捨てる**  
- 回帰は主成分空間で行うため、多重共線性による係数の暴れが抑えられる  
- 重要度の高い主成分だけを残すことで、過学習も緩和できる

### ワークフロー
1. 特徴量を標準化  
2. PCA で \\(k\\) 個の主成分を抽出  
3. その主成分得点を説明変数として線形回帰

---

## 2. Python 実装（`Pipeline`）

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import japanize_matplotlib

X, y = load_diabetes(return_X_y=True)

def build_pcr(n_components):
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=0)),
        ("reg", LinearRegression()),
    ])

components = range(1, X.shape[1] + 1)
cv_scores = []
for k in components:
    model = build_pcr(k)
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    cv_scores.append(score.mean())

best_k = components[int(np.argmax(cv_scores))]
print("CVで最良の主成分数:", best_k)

best_model = build_pcr(best_k).fit(X, y)
print("学習済みモデルの PCA 寄与率:", best_model["pca"].explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(components, [-s for s in cv_scores], marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"best={best_k}")
plt.xlabel("主成分数 k")
plt.ylabel("CV MSE (小さいほどよい)")
plt.legend()
plt.tight_layout()
plt.show()
```

![ダミー図: 主成分数と CV スコア](/images/placeholder_regression.png)

> 実際のプロットでは、主成分数を横軸、交差検証 MSE を縦軸にすると最適な圧縮次元が視覚的に選べます。

---

## 3. ハイパーパラメータ設計

- `n_components` をどこまで残すか：累積寄与率 90% などの基準 or CV で探索  
- 標準化は必須。単位が異なると PCA が歪む  
- 第一次段階でノイズを落とすため、欠損値処理や外れ値処理も忘れない  
- 回帰係数の解釈は「主成分空間」での係数になるため、元の特徴量への逆変換が必要

---

## 4. PCR の長所と注意点

**長所**
- 多重共線性に強い  
- 高次元データでも安定して学習可能  
- 可視化（主成分散布図）と一体で分析しやすい

**注意点**
- 主成分は「データの分散」を最大化する軸であり、目的変数とは無関係  
- 目的変数に効く特徴が低い主成分に潜むと、切り捨てで性能が落ちる  
- その場合は次に紹介する PLS 回帰の方が適していることも

---

## 5. まとめ

- PCR は「PCA → 線形回帰」のシンプルな2段構えで、多重共線性をうまく処理できる  
- 主成分数を CV で選び、寄与率と性能を両方チェックするのがコツ  
- 特徴量の意味を保ったまま解釈したいなら、主成分負荷量を併記するなど工夫しよう
