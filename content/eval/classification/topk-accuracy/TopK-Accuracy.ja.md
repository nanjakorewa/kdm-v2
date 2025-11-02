---
title: "Top-k Accuracy と Recall@k"
linkTitle: "Top-k Accuracy"
seo_title: "Top-k Accuracy と Recall@k｜候補リストに正解が含まれるかを測る"
title_suffix: "候補リストに正解が含まれるかを評価"
pre: "4.3.12 "
weight: 12
---

{{< lead >}}
Top-k Accuracy（または Recall@k）は、モデルが提示した上位 k 件の候補の中に正解ラベルが含まれている割合を測る指標です。多クラス分類やレコメンドのように「候補リストの中に正解があれば十分」というタスクで重宝します。
{{< /lead >}}

---

## 1. 定義
モデルがクラスごとのスコアを出力し、上位 k 件の候補集合を \\(S_k(x)\\) とすると

$$
\mathrm{Top\text{-}k\ Accuracy} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{ y_i \in S_k(x_i) \}
$$

`Recall@k` とほぼ同義で、正解ラベルが Top-k に含まれたかどうかを評価します。

---

## 2. Python 3.13 での計算
```bash
python --version  # 例: Python 3.13.0
pip install scikit-learn
```

```python
import numpy as np
from sklearn.metrics import top_k_accuracy_score

proba = model.predict_proba(X_test)  # shape: (n_samples, n_classes)
top3 = top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)
top5 = top_k_accuracy_score(y_test, proba, k=5, labels=model.classes_)

print("Top-3 Accuracy:", round(top3, 3))
print("Top-5 Accuracy:", round(top5, 3))
```

`top_k_accuracy_score` は確率スコアを受け取り、指定した `k` の Top-k Accuracy を返します。`labels` を渡しておくとクラスの並び順が異なる場合でも安全です。

---

## 3. 設計と注意点
- **k の選び方**：UI が表示できる候補数やビジネス要件に合わせて設定する。
- **タイブレーク**：同点の候補が多いときは並び順が不安定になるため、小数点付きスコアや安定ソートで制御する。
- **複数正解**：マルチラベルの場合は Hit Rate や Precision@k、Recall@k を併用して全体像を把握する。

---

## 4. 実務での活用例
- **レコメンドシステム**：提示した候補リストの中にユーザーが本当に選んだアイテムが入っているかを評価。
- **画像認識**：ImageNet などクラス数が多い課題で Top-5 Accuracy を報告するのが一般的。
- **検索システム**：検索結果上位 10 件に目的の文書が含まれているかを測る。

---

## 5. ランキング指標との比較
| 指標             | 特徴                         | 適用シーン                               |
| ---------------- | ---------------------------- | ---------------------------------------- |
| Top-k Accuracy   | 上位 k に正解があれば成功    | 候補リストに正解が含まれれば十分な場合  |
| NDCG             | 順位に重み付け               | 順位の良さも重視したいとき              |
| MAP              | 適合率の平均                 | 複数の正解が存在するランキング          |
| Hit Rate         | Top-k Accuracy とほぼ同義    | レコメンド分野で広く使用                |

---

## まとめ
- Top-k Accuracy は候補リストに正解ラベルが含まれているかを確認するシンプルな指標。
- scikit-learn の `top_k_accuracy_score` で複数の k を簡単に検証できる。
- NDCG や MAP などと組み合わせ、順位の質や複数正解の扱いも含めてモデルを評価しよう。
