---
title: "XGBoost"
pre: "2.4.9 "
weight: 9
title_suffix: "高速・高精度な勾配ブースティング実装"
---

{{< lead >}}
XGBoost（eXtreme Gradient Boosting）は、正則化と高速化を重視した勾配ブースティング実装です。欠損値処理、木構造の最適化、並列学習など実務に必要な機能が豊富で、コンペでも定番のアルゴリズムです。
{{< /lead >}}

---

## 1. 特徴

- **正則化付き損失**：L1/L2 正則化で過学習を抑制。
- **シャドウ分岐**：欠損値は自動で最適な方向へ分岐（default direction）。
- **並列化**：ツリー構築をブロック単位で並列化し高速学習。
- **高度なパラメータ**：木の深さや葉の数だけでなく、列・行サブサンプリングも細かく設定可能。

---

## 2. Python（xgboost パッケージ）による学習

```python
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
}

evals = [(dtrain, "train"), (dvalid, "valid")]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
)

pred = bst.predict(xgb.DMatrix(X_test), iteration_range=(0, bst.best_iteration + 1))
print("MAE:", mean_absolute_error(y_test, pred))
```

`early_stopping_rounds` を指定すると検証スコアが改善しなくなった時点で学習が止まり、最適な反復回数が自動で選ばれます。

---

## 3. 主要ハイパーパラメータ

| パラメータ | 役割 | 調整の勘所 |
| --- | --- | --- |
| `eta` | 学習率 | 小さくすると安定するが、`num_boost_round` を増やす必要あり |
| `max_depth` | 木の深さ | 大きいと表現力向上だが過学習しやすい |
| `min_child_weight` | 子ノードに必要な重みの合計 | 大きくすると保守的、ノイズが多いときに上げる |
| `subsample` / `colsample_bytree` | サンプリング率 | 0.6〜0.9 で汎化性能向上 |
| `lambda`, `alpha` | L2 / L1 正則化 | 大きくすると過学習抑制。スパース性が欲しいときは `alpha` を利用 |

---

## 4. 実務での活用

- **構造化データ**：カテゴリをエンコードしたテーブルデータで高精度。
- **欠損値処理**：欠損をそのまま渡しても最適に処理される。
- **特徴量重要度**：Gain/Weight/Cover など複数指標を取得可能。
- **SHAP 値**：`xgboost.to_graphviz` や `shap.TreeExplainer` と連携し、解釈性を高められる。

---

## 5. その他のヒント

- **学習率 0.1 → 0.02** に下げるなど、多段階でブースターを増やしながら微調整すると精度が向上しやすい。
- **`tree_method`**：データサイズに合わせて `"hist"`（高速）、`"gpu_hist"`（GPU）、`"approx"` を選択。
- **クロスバリデーション**：`xgb.cv` で `early_stopping_rounds` を使いながら最適ラウンドを推定。

---

## まとめ

- XGBoost は正則化・欠損処理・高速化が揃った勾配ブースティング実装で、構造化データで高精度を実現しやすい。
- `eta`、`max_depth`、`min_child_weight`、サブサンプリング、正則化などをバランスよくチューニングすることが重要。
- LightGBM や CatBoost との違いを理解しつつ、データ特性に最も適したブースターを選択しよう。

---
