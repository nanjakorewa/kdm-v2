---
title: "Top-k Accuracy と Recall@k"
pre: "4.3.12 "
weight: 12
title_suffix: "候補順位の中に正解が含まれるかを評価"
---

{{< lead >}}
Top-k Accuracy（Recall@k）は、モデルが提示した上位 k 件の候補の中に正解ラベルが入っている割合を測る指標です。多クラス分類やレコメンドで「候補リストの中に正解が含まれていればよい」場合に用いられます。
{{< /lead >}}

---

## 1. 定義

モデルがクラスごとのスコアを出力し、上位 k 件のクラス集合を \\(S_k(x)\\) とすると、

$$
\mathrm{Top\text{-}k\ Accuracy} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{ y_i \in S_k(x_i) \}
$$

`Recall@k` と同義で、正解ラベルが Top-k に含まれたかどうかを評価します。

---

## 2. Python で計算

```python
import numpy as np
from sklearn.metrics import top_k_accuracy_score

proba = model.predict_proba(X_test)  # shape: (n_samples, n_classes)
top3 = top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)
top5 = top_k_accuracy_score(y_test, proba, k=5, labels=model.classes_)

print("Top-3 Accuracy:", round(top3, 3))
print("Top-5 Accuracy:", round(top5, 3))
```

Scikit-learn の `top_k_accuracy_score` は確率行列を受け取り、指定した `k` の Top-k Accuracy を返します。`labels` を指定すると、クラスの順序がずれていても正しく評価できます。

---

## 3. 調整と注意点

- **k の選び方**：UI の候補数やランキングで上位何件を提示できるかに合わせて設定します。
- **タイブレーク**：スコアが同じ場合の並び順に注意。必要に応じて小数点を追加するなどして安定化します。
- **複数正解**：マルチラベルや複数正解がある場合は、Top-k Precision/Recall など別指標を併用する。

---

## 4. 実務での活用例

- **レコメンドシステム**：ユーザーに提示する候補リストの中に正解（ユーザーが最終的に選ぶアイテム）が含まれているか。
- **画像認識**：クラス数が多い ImageNet などで Top-5 Accuracy を報告するのが慣例。
- **検索システム**：ヒット候補の上位 10 件に正解ドキュメントが含まれるかを評価。

---

## 5. 他のランキング指標との比較

| 指標 | 特徴 | 適用シーン |
| --- | --- | --- |
| Top-k Accuracy | 上位 k に正解があれば OK | 候補リストに正解があれば十分 |
| NDCG | 順位の重み付けを考慮 | 順位の良さも評価したい |
| MAP | 適合率の平均 | 複数正解があるランキング |
| Hit Rate | Top-k Accuracy と同義 | レコメンド分野で広く利用 |

---

## まとめ

- Top-k Accuracy は候補リストに正解ラベルが含まれるかを評価するシンプルな指標。
- `top_k_accuracy_score` を使えば複数の k を簡単に評価できる。
- NDCG や MAP と組み合わせて、順位の質や複数正解の扱いも考慮しながらモデルを改善しよう。

---
