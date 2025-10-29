---
title: "バギング (Bagging)"
pre: "2.5.4 "
weight: 4
title_suffix: "分散を抑えるアンサンブルの基本"
---

{{< lead >}}
バギングは Bootstrap Aggregating の略で、データをブートストラップサンプリングして複数の弱学習器を学習し、その平均や多数決を取るアンサンブル手法です。決定木と組み合わせるとランダムフォレストに発展します。
{{< /lead >}}

---

## 1. 手順

1. 学習データから重複ありサンプリング（ブートストラップ）で複数セットを生成
2. 各セットで同じモデルを学習
3. 予測は平均（回帰）または多数決（分類）

バギングは主にモデルの分散を減らし、安定性を高めます。

---

## 2. Python 実装

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

base = DecisionTreeRegressor(max_depth=None, random_state=0)
bagging = BaggingRegressor(
    estimator=base,
    n_estimators=100,
    max_samples=0.8,
    max_features=0.8,
    bootstrap=True,
    random_state=0,
)
bagging.fit(X_train, y_train)

pred = bagging.predict(X_test)
print("RMSE:", mean_squared_error(y_test, pred, squared=False))
print("OOB スコア:", bagging.oob_score_)
```

---

## 3. ハイパーパラメータ

- `n_estimators`: 学習器の数。多いほど安定するが計算コスト増
- `max_samples`, `max_features`: 1 個の学習器に渡すデータ/特徴量の割合
- `bootstrap`: True でサンプルを重複抽出、`bootstrap_features` で特徴量も重複抽出
- `oob_score`: アンサンブルに含まれなかったサンプル（Out-of-Bag）で汎化性能を推定できる

---

## 4. 長所と短所

| 長所 | 短所 |
| ---- | ---- |
| 実装が簡単で並列化しやすい | 大量の学習器を保持する必要がある |
| 分散を大きく削減できる | バイアスは減らせないので弱学習器自体がある程度強くないと効果薄 |
| OOB 推定で追加の検証セット不要 | 解釈性は単体モデルより低い |

---

## 5. まとめ

- バギングは「データを何度も引き直して平均を取る」シンプルなアイデアで高い安定性を得る
- 決定木 + バギング = ランダムフォレストという位置付けを把握しておくと理解が深まる
- サンプル数や特徴量が多い場合でも、並列化すれば実用的な速度で動作します

---
