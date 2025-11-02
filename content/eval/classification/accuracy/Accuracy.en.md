---

title: "Accuracy | Fundamentals and pitfalls in Python 3.13"

linkTitle: "Accuracy"

seo_title: "Accuracy | Fundamentals and pitfalls in Python 3.13"

pre: "4.3.1 "

weight: 1

---



{{< lead >}}

Accuracy tells us what fraction of samples a classifier labels correctly. It is the first metric teams reach for, yet on imbalanced datasets it can hide serious mistakes. The sections below show how to compute and visualise it in Python 3.13, and when to complement it with other scores.

{{< /lead >}}



---



## 1. Definition



Using the confusion-matrix entries (true positive TP, false positive FP, false negative FN, true negative TN), accuracy is defined as:



$$

\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

$$



It measures the overall hit rate, but by itself says nothing about class imbalance. Always pair it with other metrics when positive and negative samples appear at very different frequencies.



---



## 2. Implementation and visualisation on Python 3.13



Confirm the interpreter and install the required packages:



```bash

python --version        # e.g. Python 3.13.0

pip install scikit-learn matplotlib

```



The script below trains a random forest on the breast-cancer dataset, computes Accuracy and Balanced Accuracy, and plots both as a bar chart. A `Pipeline` with `StandardScaler` keeps the preprocessing consistent. Images are saved under `static/images/eval/...` so that `generate_eval_assets.py` can refresh them automatically.



```python

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=42, stratify=y

)



pipeline = make_pipeline(

    StandardScaler(),

    RandomForestClassifier(random_state=42, n_estimators=300),

)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



acc = accuracy_score(y_test, y_pred)

bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}, Balanced Accuracy: {bal_acc:.3f}")



fig, ax = plt.subplots(figsize=(5, 4))

scores = np.array([acc, bal_acc])

labels = ["Accuracy", "Balanced Accuracy"]

colors = ["#2563eb", "#f97316"]

bars = ax.bar(labels, scores, color=colors)

ax.set_ylim(0, 1.05)

for bar, score in zip(bars, scores):

    ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.3f}", ha="center", va="bottom")

ax.set_ylabel("Score")

ax.set_title("Accuracy vs. Balanced Accuracy (Breast Cancer Dataset)")

ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

output_dir = Path("static/images/eval/classification/accuracy")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "accuracy_vs_balanced.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/accuracy/accuracy_vs_balanced.png" alt="Accuracy compared to Balanced Accuracy" caption="Balanced Accuracy exposes hidden errors when classes are imbalanced." >}}



---



## 3. Handling class imbalance



Accuracy does not differentiate the cost of false negatives vs. false positives. On skewed datasets, supplement it with:



- **Precision / Recall / F1** — to understand false alarms versus misses.

- **Balanced Accuracy** — averages recall per class, making minority classes visible.

- **Confusion Matrix** — shows which classes dominate the mistakes.

- **ROC-AUC / PR curves** — inspect probability thresholds and trade-offs.



Balanced Accuracy equals the mean recall of each class and is a good default when outcomes are skewed or when compliance requires a fairness-aware score.



---



## 4. Operational checklist



1. **Align with business cost** – check the confusion matrix and confirm that the “99 % accuracy” claim does not mask critical misses.

2. **Explore thresholding** – analyse ROC-AUC or PR curves to see how accuracy changes when you adjust the decision threshold.

3. **Report multiple metrics** – include Precision, Recall, F1, and Balanced Accuracy in dashboards so stakeholders recognise trade-offs.

4. **Keep reproducible notebooks** – store the evaluation in a Python 3.13 notebook to re-run it quickly after model updates.



---



## Summary



- Accuracy is a convenient headline metric, but can mislead on imbalanced data.

- A scikit-learn pipeline with scaling makes the calculation reproducible in Python 3.13.

- Combine Accuracy with Balanced Accuracy and class-wise metrics to build a trustworthy evaluation narrative.

