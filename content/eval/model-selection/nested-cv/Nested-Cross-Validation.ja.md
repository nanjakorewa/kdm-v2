---
title: "ネステッド交差検証"
pre: "4.1.6 "
weight: 6
title_suffix: "ハイパーパラメータ探索を含む汎化性能の推定"
---

{{< lead >}}
ネステッド交差検証（Nested Cross-Validation）は、外側のループで汎化性能を評価し、内側のループでハイパーパラメータ探索を行う手法です。過学習を防ぎ、ハイパーパラメータ調整を含めた真の汎化誤差を推定できます。
{{< /lead >}}

---

## 1. 仕組み

1. **外側ループ**：データを `K_outer` 分割し、各分割をテストセットとして保持。
2. **内側ループ**：残りのデータで `K_inner` の交差検証を行い、グリッドサーチやランダムサーチで最適ハイパーパラメータを選択。
3. **評価**：選ばれたハイパーパラメータで再学習し、外側ループのテストセットでスコアを計算。

これを `K_outer` 回繰り返し、スコアの平均と分散を算出します。

---

## 2. Python での実装例

```python
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
}

inner_cv = KFold(n_splits=3, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

grid = GridSearchCV(
    RandomForestClassifier(random_state=0),
    param_grid=param_grid,
    cv=inner_cv,
    scoring="roc_auc",
    n_jobs=-1,
)

scores = cross_val_score(grid, X, y, cv=outer_cv, scoring="roc_auc", n_jobs=-1)

print("Nested CV ROC-AUC:", scores.mean(), "+/-", scores.std())
```

`cross_val_score` にサーチオブジェクトを渡すだけでネステッド CV を実行できます。

---

## 3. 利点

- **ハイパーパラメータ調整のリークを防ぐ**：同じデータでチューニングと評価を行うことによる過大評価を防止。
- **複数モデルの公平な比較**：モデルごとに最適なハイパーパラメータを選びつつ、汎化性能を同じ条件で比較できる。
- **分散の推定**：外側ループのスコアを用いて性能分散を推定し、信頼区間を計算できる。

---

## 4. 注意点

- 計算コストが高い：`K_outer × K_inner` 回モデルを学習するため、モデルが重い場合は時間がかかる。
- グリッドサーチの範囲を広げすぎると現実的な時間で終わらない。ランダムサーチやベイズ最適化を検討する。
- データサイズが小さい場合は、分割を工夫したり、過剰な CV を避ける。

---

## 5. 実務でのヒント

- **小規模データ**：データが少ないほどハイパーパラメータ調整のリークが問題になりやすく、ネステッド CV の価値が高い。
- **主要モデルのみ**：全モデルで実施すると時間がかかるため、候補を絞った上でネステッド CV を実行する。
- **報告書への記載**：探索範囲や分割数を明記し、過学習リスクを適切に管理したことを示す。

---

## まとめ

- ネステッド交差検証はハイパーパラメータ探索を組み込んだ汎化性能推定法で、過大評価を防げる。
- `GridSearchCV` と `cross_val_score` を組み合わせるだけで実装できるが、計算コストには注意。
- 重要なモデルの最終評価に用い、信頼できる性能指標を報告しよう。

---
