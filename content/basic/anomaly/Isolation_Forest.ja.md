---
title: "Isolation Forest"
pre: "2.6.4 "
weight: 4
title_suffix: "ランダム分割で外れ値を隔離する"
---

{{< lead >}}
Isolation Forest はランダムな決定木を多数構築し、サンプルをどれだけ短い経路で隔離できるかを指標に異常スコアを付ける手法です。高次元データでも高速に動作します。
{{< /lead >}}

---

## 1. 仕組み

- サンプルをランダムにサブサンプリング
- 一様に選んだ特徴量と閾値でデータを分割するランダム木（Isolation Tree）を構築
- 平均経路長が短いサンプルほど「孤立しやすい」= 異常度が高い

異常スコアは平均経路長と、完全二分木の期待パス長 \\(c(n)\\) を用いて正規化します。

---

## 2. Python 実装

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.ensemble import IsolationForest

rng = np.random.default_rng(0)
X_inliers = 0.3 * rng.normal(size=(200, 2))
X_anom = rng.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X_inliers, X_anom])

model = IsolationForest(n_estimators=200, contamination=0.1, random_state=0)
model.fit(X)
scores = -model.score_samples(X)
labels = model.predict(X)  # -1 = 異常

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=scores, cmap="magma", s=30)
plt.colorbar(label="anomaly score")
plt.title("Isolation Forest のスコア")
plt.tight_layout()
plt.show()

print("異常と判定された数:", np.sum(labels == -1))
```

![ダミー図: Isolation Forest のスコアヒートマップ](/images/placeholder_regression.png)

---

## 3. ハイパーパラメータ

- `n_estimators`: 木の数。多いほど安定
- `max_samples`: 各木に使うサンプル数。デフォルトは `min(256, n_samples)`
- `contamination`: 異常データの割合推定。自動閾値の基準になる
- `max_features`: 各分割で使う特徴量数

---

## 4. 長所と短所

| 長所 | 短所 |
| ---- | ---- |
| 高次元でも比較的高速 | 乱数シードで結果が多少変わる |
| スケーリングが不要（ただし推奨） | 小さな局所異常は見逃すことがある |
| モデル学習・推論がシンプル | `contamination` の設定が難しい場合がある |

---

## 5. まとめ

- Isolation Forest は「分割回数の少なさ」を異常指標にするユニークな木ベース手法
- パラメータは概ね木の数とサンプル数だけで、実装も scikit-learn で簡単
- ログ検知やセンサーデータなどで高速に異常候補を絞り込みたいときに有用です

---
