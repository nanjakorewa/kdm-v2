---
title: "Orthogonal Matching Pursuit（OMP）"
pre: "2.1.11 "
weight: 11
title_suffix: "疎な係数を greedy に選ぶ線形回帰"
---

{{< lead >}}
特徴量が非常に多いとき、すべてに係数を割り当てる必要はありません。OMP は「寄与が大きい特徴量を 1 つずつ追加し、残差が小さくなる方向を選ぶ」貪欲法で疎な線形モデルを作ります。
{{< /lead >}}

---

## 1. どうやって動く？

1. 初期状態では残差を目的変数 \(y\) に設定  
2. 残差と最も相関が高い特徴量を 1 つ選択  
3. 選択済み特徴量だけで最小二乗解を計算し、残差を更新  
4. 既定のステップ数や誤差しきい値まで繰り返す

これにより、重要な特徴が順番に積み上がっていき、ラッソ回帰に似た疎な係数ベクトルが得られます。

---

## 2. 長所と短所

**長所**
- 計算が速く、係数が疎になる  
- 特徴量選択のステップが明示的で解釈しやすい  
- 正則化ハイパーパラメータよりも「選択する特徴量数」を直接指定しやすい

**短所**
- Greedy 手法なので最適とは限らない  
- 特徴量が強く相関していると順番に依存し、ノイズに弱くなる  
- 選択順序が結果に影響するため、標準化や相関の事前チェックが重要

---

## 3. Python 実装（`OrthogonalMatchingPursuit`）

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

print("OMP 係数（非ゼロ）:", np.flatnonzero(omp.coef_))
print("Lasso 係数（非ゼロ）:", np.flatnonzero(np.abs(lasso.coef_) > 1e-6))

print("OMP MSE:", mean_squared_error(y, omp.predict(X)))
print("Lasso MSE:", mean_squared_error(y, lasso.predict(X)))

coef_df = pd.DataFrame({
    "true": true_coef,
    "omp": omp.coef_,
    "lasso": lasso.coef_,
})
print(coef_df.head(10))
```

![ダミー図: OMP と Lasso の係数比較](/images/placeholder_regression.png)

> 実際の図では、真の係数と OMP/Lasso の推定値を棒グラフで並べると疎性の違いが伝わります。

---

## 4. ハイパーパラメータの決め方

- `n_nonzero_coefs`: 何本の特徴量を残すか。事前知識があれば固定、なければ CV で探索  
- `tol`: 残差ノルムのしきい値。満たした時点で停止  
- 特徴量を標準化しておかないと「スケールの大きい特徴」が優先される  
- `fit_intercept=True` のままでも、事前に平均を引いておくと安定

---

## 5. 使いどころ

- 高次元・少サンプル（\(n \ll p\)）で、スパース構造を仮定できるとき  
- ラッソ回帰に近い結果が欲しいが、シンプルな手順で特徴量選択を説明したいとき  
- シグナル処理やスパース表現（辞書学習）における基礎アルゴリズムとして

---

## 6. まとめ

- OMP は「相関の強い特徴量を順に追加していく」貪欲型の疎回帰アルゴリズム  
- `OrthogonalMatchingPursuit` で簡単に試せ、ラッソとの比較で直感が掴める  
- 相関が強い場合は Elastic Net やベイズ的スパース推定と併用して精度を確認しましょう

---
