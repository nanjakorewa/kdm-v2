---

title: "Balanced Accuracy | Evaluating imbalanced datasets"

linkTitle: "Balanced Accuracy"

seo_title: "Balanced Accuracy | Evaluating imbalanced datasets"

pre: "4.3.6 "

weight: 6

---



{{< lead >}}

Balanced Accuracy averages the recall of each class, so it remains informative when your data are skewed. The example below, runnable on Python 3.13, shows how it differs from plain Accuracy and when you should prefer it.

{{< /lead >}}



---



## 1. Definition



Balanced Accuracy is the mean of the true-positive rate (TPR) and the true-negative rate (TNR):





\mathrm{Balanced\ Accuracy} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)





For multiclass problems you average the recall of each class in the same spirit.



---



## 2. Implementation in Python 3.13



```bash

python --version        # e.g. Python 3.13.0

pip install scikit-learn matplotlib

```



We reuse the random-forest classifier from the Accuracy article and print both metrics side by side. The bar chart is saved at static/images/eval/classification/accuracy/accuracy_vs_balanced.png, so generate_eval_assets.py can regenerate it whenever you update the notebook.



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

```



{{< figure src="/images/eval/classification/accuracy/accuracy_vs_balanced.png" alt="Accuracy vs Balanced Accuracy" caption="Balanced Accuracy weights each class equally by averaging the recall per class." >}}



---



## 3. When to prefer Balanced Accuracy



- **Strong class imbalance** – plain Accuracy only reflects the majority class, while Balanced Accuracy keeps minority recall visible.

- **Model comparison** – when benchmark teams submit models on skewed data, Balanced Accuracy makes their performance differences more honest.

- **Threshold tuning** – combine it with precision/recall plots to see whether both classes remain detectable at your chosen threshold.



---



## 4. Companion metrics



| Metric | Measures | Caveat on imbalanced data |

| --- | --- | --- |

| Accuracy | Overall hit rate | Dominated by the majority class |

| Recall / Sensitivity | Detection rate per class | Requires separate reporting for each class |

| **Balanced Accuracy** | Mean recall across classes | Highlights minority-class recall loss |

| Macro F1 | Harmonic mean of precision & recall (per class) | Useful when precision also matters |



---



## Summary



- Balanced Accuracy is the average of per-class recall, making it well suited to imbalanced datasets.

- In Python 3.13, alanced_accuracy_score gives you the value in one line; compare it with Accuracy to show stakeholders the difference.

- Combine it with precision, recall, and F1 metrics to decide how much weight to give each class when evaluating models.

---

