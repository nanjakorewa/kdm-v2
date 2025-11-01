---
title: "ROC-AUC | A guide to threshold tuning and model comparison"
linkTitle: "ROC-AUC"
seo_title: "ROC-AUC | A guide to threshold tuning and model comparison"
pre: "4.3.3 "
weight: 1
---

{{< lead >}}
The ROC curve plots the trade-off between the true-positive rate and the false-positive rate, and the area underneath it (AUC) summarises the classifier’s ranking ability. With Python 3.13 code in hand, we can visualise the curve, compute its AUC, and use it to fine-tune thresholds.
{{< /lead >}}

---

## 1. What ROC and AUC represent

The ROC curve traces **False Positive Rate (FPR)** on the x-axis and **True Positive Rate (TPR)** on the y-axis while sweeping the decision threshold from 0 to 1. The area under the curve (AUC) ranges between 0.5 (random guessing) and 1.0 (perfect separation).

- AUC ≈ 1.0 → excellent discrimination
- AUC ≈ 0.5 → indistinguishable from random
- AUC < 0.5 → the model may have flipped polarity; inverting the decision rule could improve performance

---

## 2. Implementation and plotting in Python 3.13

Check your interpreter and install the required packages:

`ash
python --version        # e.g. Python 3.13.0
pip install scikit-learn matplotlib
`

The code below trains a logistic regression model on the breast cancer dataset, draws the ROC curve, and saves the figure to static/images/eval/classification/rocauc. It is compatible with generate_eval_assets.py so you can refresh assets automatically.

`python
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, solver="lbfgs"),
)
pipeline.fit(X_train, y_train)
proba = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"ROC-AUC: {auc:.3f}")

fig, ax = plt.subplots(figsize=(5, 5))
RocCurveDisplay.from_predictions(
    y_test,
    proba,
    name="Logistic Regression",
    ax=ax,
)
ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (Breast Cancer Dataset)")
ax.legend(loc="lower right")
fig.tight_layout()
output_dir = Path("static/images/eval/classification/rocauc")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "roc_curve.png", dpi=150)
plt.close(fig)
`

{{< figure src="/images/eval/classification/rocauc/roc_curve.png" alt="ROC curve example" caption="The AUC measures the area below the ROC curve; higher values indicate better ranking ability." >}}

---

## 3. Using the curve for threshold tuning

- **Recall-sensitive domains** (healthcare, fraud): move along the curve to pick a higher TPR while keeping FPR acceptable.
- **Balancing precision vs. recall**: models with high AUC typically maintain good performance across a wider range of thresholds.
- **Model comparison**: AUC provides a single scalar to compare classifiers before picking an operating point.

Combine ROC-AUC with precision–recall analysis to understand the cost of altering the decision threshold.

---

## 4. Operational checklist

1. **Inspect class imbalance** – even with AUC close to 0.5, a different threshold might still rescue important cases.
2. **Evaluate class-weight strategies** – adjust class weights or sample weights and verify whether AUC improves.
3. **Share the plot** – include the ROC curve in dashboards so teams can reason about trade-offs.
4. **Keep a Python 3.13 notebook** – reproduce the calculation effortlessly whenever the model is retrained.

---

## Summary

- ROC-AUC captures how well a classifier ranks positives ahead of negatives across all thresholds.
- In Python 3.13, RocCurveDisplay and oc_auc_score make computation and plotting straightforward.
- Use the curve in tandem with precision–recall metrics to choose operating thresholds aligned with business goals.
---
