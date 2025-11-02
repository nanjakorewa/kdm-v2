---
title: "Choosing Averaging Strategies for Classification Metrics"
linkTitle: "Averaging Strategies"
seo_title: "Averaging Strategies | Multi-class and Multi-label Evaluation"
pre: "4.3.14 "
weight: 14
---

{{< lead >}}
When you compute Precision, Recall, F1, and other metrics for multi-class or multi-label classification, the `average` argument in scikit-learn controls how scores are aggregated. This page compares the four most common strategies with Python 3.13 examples.
{{< /lead >}}

---

## 1. Main averaging options
| average    | How it is computed                                         | When to use it                                                  |
| ---------- | ---------------------------------------------------------- | ---------------------------------------------------------------- |
| `micro`    | Sum TP/FP/FN over all samples, then compute the metric     | Emphasises overall correctness regardless of class distribution |
| `macro`    | Compute the metric per class, then take the unweighted mean | Gives every class the same weight; highlights minority classes   |
| `weighted` | Compute the metric per class, then take a support-weighted mean | Preserves class ratios; behaves closer to Accuracy                |
| `samples`  | Multi-label only. Average metrics per sample               | For cases where each sample can have multiple labels             |

---

## 2. Comparing in Python 3.13
```bash
python --version  # e.g. Python 3.13.0
pip install scikit-learn matplotlib
```

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=30_000,
    n_features=20,
    n_informative=6,
    weights=[0.85, 0.1, 0.05],  # Imbalanced classes
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, multi_class="ovr"),
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
for avg in ["micro", "macro", "weighted"]:
    print(f"F1 ({avg}):", f1_score(y_test, y_pred, average=avg))
```

`classification_report` prints per-class metrics as well as `macro avg`, `weighted avg`, and `micro avg`, making it easy to compare the different strategies side by side.

---

## 3. Picking the right strategy
- **micro** – Best when you care about overall correctness and every prediction carries the same importance.
- **macro** – Use when minority classes matter; it treats every class equally and penalises poor recall on rare labels.
- **weighted** – Useful when you want to stay close to the real class distribution while still reporting Precision/Recall/F1.
- **samples** – The default choice for multi-label tasks where each sample can have several ground-truth labels.

---

## Takeaways
- The `average` parameter changes the meaning of the resulting metric drastically; match it with your task and business goal.
- Remember: `macro` treats classes equally, `micro` focuses on overall ratios, `weighted` preserves class balance, and `samples` targets multi-label use cases.
- scikit-learn lets you compute several averages in one go, so report multiple views to avoid misinterpreting model quality.
