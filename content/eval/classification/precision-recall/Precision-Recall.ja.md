---
title: "適合率・再現率とF1スコア"
pre: "4.3.2 "
weight: 2
title_suffix: "不均衡データでの意思決定"
---


{{< lead >}}
精度（Precision）と再現率（Recall）は、陽性/陰性のバランスが崩れたデータで性能を把握するための最重要指標です。F1 スコアや平均適合率 (AP) と組み合わせることで、閾値選択やアンサンブルの重み付けが判断しやすくなります。
{{< /lead >}}

{{% notice document %}}
- [precision_score — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)  
- [precision_recall_curve — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)  
- [classification_report — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
{{% /notice %}}

---

## 1. 混同行列を基にした定義

|            | 予測:陽性 | 予測:陰性 |
| ---------- | --------- | --------- |
| **実際:陽性** | 真陽性 (TP) | 偽陰性 (FN) |
| **実際:陰性** | 偽陽性 (FP) | 真陰性 (TN) |

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}
$$

- 適合率は **検知した陽性の純度** を表す指標。FP を下げたいときに重視。
- 再現率は **本来陽性であるものを漏らさない能力**。FN が高いと医療や不正検知では致命的になる。
- 精度・再現率の調和平均が **F1 スコア**：

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

---

## 2. Python で計算する

```python
from sklearn.datasets import make_classification
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=20_000,
    n_features=30,
    n_informative=6,
    weights=[0.95, 0.05],  # 陽性が 5 %
    class_sep=1.0,
    random_state=0,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)

clf = LogisticRegression(max_iter=200, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))
```

`class_weight="balanced"` を指定すると、クラス比に応じて損失の重みが調整され、再現率が上がりやすくなります。

---

## 3. 閾値調整と Precision-Recall 曲線

ROC 曲線は不均衡データで優れて見えすぎることがあります。PR 曲線は陽性クラスの挙動に専念するため、実務での意思決定に向いています。

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

y_score = clf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
ap = average_precision_score(y_test, y_score)

plt.figure(figsize=(6, 5))
plt.step(recall, precision, where="post", label=f"PR curve (AP={ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim(0.0, 1.05)
plt.xlim(0.0, 1.0)
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# 0.3 以上のしきい値を試す
threshold = 0.3
y_pred_high_recall = (y_score >= threshold).astype(int)
print(
    "調整後 Precision:", precision_score(y_test, y_pred_high_recall),
    "Recall:", recall_score(y_test, y_pred_high_recall),
)
```

- PR 曲線の面積 (**AP**) が高いほど、広い閾値範囲で高い適合率を維持できる。
- しきい値を 0.5 から動かすだけでも、リコール重視・適合率重視のどちらにも寄せやすい。

---

## 4. マルチクラスと平均化戦略

マルチクラスではクラスごとの混同行列を平均します。`average` 引数を使い分けましょう。

- `micro`：全サンプルを一括で評価。クラス比の偏りを反映し、頻度の高いクラスが優先される。
- `macro`：クラスごとの指標を単純平均。クラスごとに同じ重みを与えるので、レアクラスの状態を見たいときに便利。
- `weighted`：クラスごとの指標をサンプル数で重み付け平均。全体指標として解釈しやすい。
- `samples`：マルチラベル用。各サンプルのラベル集合で評価する。

```python
precision_score(y_test, y_pred, average="macro")
recall_score(y_test, y_pred, average="weighted")
f1_score(y_test, y_pred, average="micro")
```

---

## 5. まとめ

- 適合率は偽陽性、再現率は偽陰性のコストに対応する。どちらを重視するかは業務要件次第。
- PR 曲線と平均適合率は不均衡データで ROC より情報量が多く、閾値調整の根拠になる。
- `average` の使い分け、`class_weight` や閾値調整、アンサンブルによる調整を組み合わせて、望ましい Precision-Recall のバランスに近づけよう。

---
