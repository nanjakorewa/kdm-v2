---
title: "感度と特異度（Sensitivity / Specificity）"
pre: "4.3.7 "
weight: 7
title_suffix: "偽陰性と偽陽性のバランスを把握"
---

{{< lead >}}
感度（Sensitivity）は陽性を取り逃さない能力、特異度（Specificity）は陰性を誤検出しない能力を表します。医療診断や不正検知など、偽陰性・偽陽性のコストが異なるタスクで基礎になる指標です。
{{< /lead >}}

---

## 1. 定義

混同行列を用いると次のように表されます。

$$
\mathrm{Sensitivity} = \frac{TP}{TP + FN}, \qquad
\mathrm{Specificity} = \frac{TN}{TN + FP}
$$

- 感度が高いほど、陽性サンプルを逃さない。
- 特異度が高いほど、陰性サンプルを誤って陽性と判定しない。

---

## 2. Python で計算

```python
from sklearn.metrics import confusion_matrix
import numpy as np

cm = np.array([[85, 5],
               [12, 98]])  # [[TN, FP], [FN, TP]]

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Sensitivity:", round(sensitivity, 3))
print("Specificity:", round(specificity, 3))
```

scikit-learn の `classification_report` では感度は `recall` として出力されます。特異度は `recall` を `pos_label=0` にして計算するか、上記のように手計算します。

---

## 3. しきい値調整でトレードオフ

確率を出力するモデルでは、しきい値を動かすと感度と特異度がトレードオフになります。

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, probas)
specificities = 1 - fpr  # 特異度
```

- ROC 曲線上の一点を選ぶことは、感度と特異度のペアを選ぶことに等しい。
- コストが明確なら、感度・特異度に重みを付けた目的関数（Youden Index など）で最適なしきい値を探す方法もあります。

---

## 4. Youden Index と平衡感度

Youden Index（J統計量）は感度と特異度の和から 1 を引いた値で、最適なしきい値探索に利用されます。

$$
J = \mathrm{Sensitivity} + \mathrm{Specificity} - 1
$$

J が最大になるしきい値は、感度と特異度をバランス良く確保できるポイントです。

---

## 5. 実務上のポイント

- **感度重視**：疾病スクリーニングなど、陽性を見逃すリスクが高い場合。
- **特異度重視**：誤検出がコストにつながるクレジットカード審査など。
- **報告の際**：Accuracy と併せ、感度・特異度を別々に示すことで意思決定者がリスクを評価しやすくなります。

---

## まとめ

- 感度（Sensitivity）は陽性の取り逃し、特異度（Specificity）は陰性の誤警報を示す基本指標。
- しきい値の調整で双方はトレードオフになるため、業務コストに基づいて適切なバランスを選ぶ。
- Youden Index などを活用しつつ、Accuracy だけでは分からないリスクを補完しよう。

---
