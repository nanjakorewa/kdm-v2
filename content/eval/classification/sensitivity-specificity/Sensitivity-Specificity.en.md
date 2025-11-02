---
title: "Sensitivity and Specificity: Balancing False Negatives and False Positives"
linkTitle: "Sensitivity / Specificity"
seo_title: "Sensitivity vs. Specificity | Balancing false negatives and false positives"
title_suffix: "Balancing false negatives and false positives"
pre: "4.3.7 "
weight: 7
---

{{< lead >}}
Sensitivity (recall for the positive class) measures how well we capture positives, while specificity measures how well we avoid false alarms on negatives. Both are essential in domains where the cost of false negatives and false positives differs dramatically.
{{< /lead >}}

---

## 1. Definition
From the confusion matrix we can write:

$$
\mathrm{Sensitivity} = \frac{TP}{TP + FN}, \qquad
\mathrm{Specificity} = \frac{TN}{TN + FP}
$$

- Higher sensitivity means fewer missed positives.  
- Higher specificity means fewer negatives flagged incorrectly.

---

## 2. Computing in Python 3.13
```bash
python --version  # e.g. Python 3.13.0
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

In scikit-learn, sensitivity is the standard `recall`. Specificity can be obtained via `recall_score(y_true, y_pred, pos_label=0)` or by computing it manually as above.

---

## 3. Threshold trade-offs
For probabilistic models, adjusting the decision threshold changes sensitivity and specificity.

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, probas)
specificities = 1 - fpr
```

- Each point on the ROC curve corresponds to a specific sensitivity/specificity pair.
- When misclassification costs are known, optimise the threshold using metrics such as the Youden Index or cost-sensitive objectives.

---

## 4. Youden Index for balanced choices
The Youden Index balances the two metrics:

$$
J = \mathrm{Sensitivity} + \mathrm{Specificity} - 1
$$

The threshold that maximises \\(J\\) offers a good compromise when both types of errors matter.

---

## 5. Practical guidance
- **Sensitivity-first**: Screening for severe diseases values sensitivity to avoid missing positive cases.
- **Specificity-first**: Fraud detection systems may favour specificity to reduce costly false positives on legitimate transactions.
- **Reporting**: Present sensitivity and specificity alongside Accuracy so stakeholders can judge trade-offs explicitly.

---

## Key takeaways
- Sensitivity tracks missed positives; specificity tracks false alarms on negatives.
- Moving the decision threshold trades one for the other—align the balance with business costs.
- Use tools like the Youden Index to complement Accuracy and surface risks that a single metric would hide.
