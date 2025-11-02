---
title: "感度と特異度（Sensitivity / Specificity）"
linkTitle: "Sensitivity / Specificity"
seo_title: "感度と特異度｜偽陰性と偽陽性のバランスを把握"
title_suffix: "偽陰性と偽陽性のバランスを把握"
pre: "4.3.7 "
weight: 7
---

{{< lead >}}
感度（Sensitivity）は陽性を取りこぼさない力、特異度（Specificity）は陰性を誤って陽性と判定しない力を表す指標です。医療診断や不正検知など、偽陰性と偽陽性のコストが大きく異なるタスクで基本となります。
{{< /lead >}}

---

## 1. 定義
混同行列を用いると次のように定義されます。

$$
\mathrm{Sensitivity} = \frac{TP}{TP + FN}, \qquad
\mathrm{Specificity} = \frac{TN}{TN + FP}
$$

- 感度が高いほど陽性サンプルの見逃しが少ない。  
- 特異度が高いほど陰性サンプルを誤検知しにくい。

---

## 2. Python 3.13 での計算
```bash
python --version  # 例: Python 3.13.0
pip install numpy scikit-learn
```

```python
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)  # [[TN, FP], [FN, TP]]
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Sensitivity:", round(sensitivity, 3))
print("Specificity:", round(specificity, 3))
```

scikit-learn の `classification_report` では感度が `recall` として出力されます。特異度は `recall(y_true, y_pred, pos_label=0)` として計算するか、上記のように手計算します。

---

## 3. しきい値調整とトレードオフ
確率を出力するモデルでは、しきい値を動かすと感度と特異度がトレードオフになります。

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, probas)
specificities = 1 - fpr
```

- ROC 曲線上の 1 点は感度と特異度の組み合わせを表します。
- コストが明確な場合は、Youden Index やコストベースの目的関数で最適なしきい値を探す方法も有効です。

---

## 4. Youden Index とバランス
Youden Index は感度と特異度の和から 1 を引いた値で、バランスの良いしきい値を選ぶ指標として使われます。

$$
J = \mathrm{Sensitivity} + \mathrm{Specificity} - 1
$$

\\(J\\) が最大になるしきい値では、双方をバランス良く確保できます。

---

## 5. 実務でのポイント
- **感度重視の例**：重症疾患のスクリーニングでは、見逃し（偽陰性）を避けるため感度を高く保つ。
- **特異度重視の例**：クレジットカードの不正検知で、正常取引を止める（偽陽性）コストが大きい場合は特異度を重視。
- **報告テンプレート**：Accuracy と併せて感度・特異度を並べて提示すると、意思決定者がリスクを評価しやすい。

---

## まとめ
- 感度は陽性の取りこぼし、特異度は陰性の誤検知を表す基本指標。
- しきい値を動かすと感度と特異度がトレードオフになるため、業務コストに基づきバランスを決める。
- Youden Index などを活用し、Accuracy だけでは見えないリスクを補完しよう。
