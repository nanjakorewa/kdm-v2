---
title: "Precision, Recall, and F1 | Threshold tuning in Python 3.13"
linkTitle: "Precision-Recall"
seo_title: "Precision, Recall, and F1 | Threshold tuning in Python 3.13"
pre: "4.3.2 "
weight: 2
---

{{< lead >}}
Precision measures how reliable positive predictions are, recall measures how many real positives we capture, and F1 harmonises the two. With reproducible Python 3.13 code we can inspect the trade-off on a precision–recall curve and decide which threshold works best.
{{< /lead >}}

---

## 1. Definitions at a glance

Given the confusion-matrix counts — true positives (TP), false positives (FP), false negatives (FN) — precision and recall are defined as:


\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}


- **Precision** — of the items predicted as positive, which fraction is truly positive? Important when false alarms are costly.
- **Recall** — of the actual positives, which fraction do we catch? Important when misses must be avoided.
- **F1 score** — harmonic mean that balances precision and recall in a single number.


F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}


---

## 2. Implementation and PR curve on Python 3.13

Install the dependencies in your interpreter:

`ash
python --version        # e.g. Python 3.13.0
pip install scikit-learn matplotlib
`

The snippet below creates an imbalanced dataset (positive class 5%), trains a weighted logistic regression, and plots the precision–recall (PR) curve along with the average precision (AP). The figure is stored at static/images/eval/classification/precision-recall/pr_curve.png so generate_eval_assets.py can refresh it automatically.

`python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=40_000,
    n_features=20,
    n_informative=6,
    weights=[0.95, 0.05],
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)
print(classification_report(y_test, y_pred, digits=3))

precision, recall, thresholds = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)

fig, ax = plt.subplots(figsize=(5, 4))
ax.step(recall, precision, where="post", label=f"PR curve (AP={ap:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
output_dir = Path("static/images/eval/classification/precision-recall")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "pr_curve.png", dpi=150)
plt.close(fig)
`

{{< figure src="/images/eval/classification/precision-recall/pr_curve.png" alt="Precision–Recall curve" caption="Average precision (AP) is the area under the PR curve. Moving the threshold slides along the curve." >}}

---

## 3. Choosing a decision threshold

- Keeping the default threshold (0.5) might yield low recall when the positive class is rare.
- Every point on the PR curve corresponds to a threshold from precision_recall_curve’s 	hresholds array.
- Lowering the threshold increases recall at the expense of precision; use business cost to decide how far you can move.

`python
threshold = 0.3
custom_pred = (proba >= threshold).astype(int)
print(
    "threshold=0.30",
    "Precision=", precision_score(y_test, custom_pred),
    "Recall=", recall_score(y_test, custom_pred),
    "F1=", f1_score(y_test, custom_pred),
)
`

---

## 4. Averaging strategies for multiclass tasks

scikit-learn’s verage parameter lets you aggregate class-wise metrics:

- macro — simple mean across classes; treats each class equally.
- weighted — weighted mean by class support; preserves overall balance.
- micro — recompute from the pooled confusion matrix; useful but can hide minority-class behaviour.

`python
precision_score(y_test, y_pred, average="macro")
recall_score(y_test, y_pred, average="weighted")
f1_score(y_test, y_pred, average="micro")
`

---

## Summary

- Precision penalises false positives, recall penalises false negatives; F1 balances both.
- Precision–recall curves reveal how the trade-off changes with the threshold; the average precision condenses it into a single number.
- In Python 3.13 you can compute and plot the curve with a few lines of scikit-learn, then share the threshold analysis for collaborative decision-making.
---
