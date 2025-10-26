---
title: "平均適合率（Average Precision, AP）"
pre: "4.3.9 "
weight: 9
title_suffix: "PR曲線の面積で確率順位を評価"
---

{{< lead >}}
平均適合率（Average Precision, AP）は、Precision-Recall 曲線の面積に相当する指標で、スコアの順位付けがどれだけ良いかを測ります。確率を出力するモデルやランキングモデルの評価に広く用いられます。
{{< /lead >}}

---

## 1. 定義

AP は離散的な再現率レベルでの適合率を積み上げた値として定義されます。

$$
\mathrm{AP} = \sum_n (R_n - R_{n-1}) P_n
$$

ここで \\(R_n\\) は再現率、\\(P_n\\) は対応する適合率です。再現率の差分を重みにして積分を近似するイメージです。

---

## 2. Python で計算

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_score)
ap = average_precision_score(y_test, y_score)

print("Average Precision:", round(ap, 3))
```

`y_score` は陽性クラスの予測確率やスコアを渡します。二値分類では `average_precision_score` が簡単に計算でき、多ラベルや多クラスでは `average` 引数で `macro`、`weighted` などを指定できます。

---

## 3. AP と PR-AUC の違い

AP は標準的には「再現率の階段関数に基づく面積」、PR-AUC は数値積分による面積です。Scikit-learn の `average_precision_score` は AP を返し、`sklearn.metrics.auc(recall, precision)` は単純積分の PR-AUC を返します。AP は再現率のステップに合わせて補間するため、順位評価として安定しています。

---

## 4. ランキング指標としての解釈

- スコアを降順に並べたときの Precision を、各再現率レベルで平均したもの。
- AP が高いほど、陽性サンプルを上位に集められている。
- AP = 1 の場合、陽性がすべて上位に位置し、一度も適合率が下がらない理想的なケース。

情報検索や推薦システムでは、平均適合率の平均（MAP）を用いてランキングモデル全体を評価します。

---

## 5. 実務での活用

- **しきい値に依存しない評価**：PR 曲線全体をまとめて評価できるため、固定しきい値の F1 よりも総合的な判定が可能。
- **不均衡データ**：陽性クラスが少ない場合でも、確率順位の良さを評価できる。
- **アンサンブル**：複数モデルの確率を組み合わせる際、AP を指標にして最適な重みを探索できる。

---

## まとめ

- Average Precision は PR 曲線の面積に相当し、確率スコアの順位付け性能を測る。
- `average_precision_score` で簡単に算出でき、特にクラス不均衡タスクで有効。
- F1 や ROC-AUC と併用し、しきい値固定の性能とランキング性能の両方を確認しよう。

---
