---

title: "Matthews Correlation Coefficient (MCC) | A balanced binary metric"

linkTitle: "MCC"

seo_title: "Matthews Correlation Coefficient (MCC) | A balanced binary metric"

pre: "4.3.5 "

weight: 5

---



{{< lead >}}

The Matthews Correlation Coefficient (MCC) captures all four entries of the confusion matrix—TP, FP, FN, TN—in a single value between −1 and 1. It stays informative even on skewed datasets, making it a trustworthy companion to Accuracy and F1.

{{< /lead >}}



---



## 1. Definition



For binary classification:





\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}.





- **1** → perfect prediction

- **0** → no better than random

- **−1** → total disagreement



Multiclass MCC generalises this formula using the complete confusion matrix.



---



## 2. Computing MCC in Python 3.13



```bash

python --version        # e.g. Python 3.13.0

pip install scikit-learn matplotlib

```



```python

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import matthews_corrcoef, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



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



model = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, class_weight="balanced"),

)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(confusion_matrix(y_test, y_pred))

print("MCC:", matthews_corrcoef(y_test, y_pred))

```



class_weight="balanced" helps the minority class contribute to the coefficient.



---



## 3. Threshold analysis



{{< figure src="/images/eval/classification/mcc/mcc_vs_threshold.png" alt="MCC vs threshold" caption="Find the threshold where MCC peaks to strike the best balance between all confusion-matrix cells." >}}



Unlike F1, MCC rewards correct negatives as well. Evaluating it over thresholds reveals the operating point with the highest correlation.



---



## 4. Practical use cases



- **Sanity-check Accuracy** – a high Accuracy but low MCC signals that one class is being ignored.

- **Model selection** – use make_scorer(matthews_corrcoef) in GridSearchCV to optimise directly for MCC.

- **Combine with ROC/PR curves** – MCC highlights overall balance while ROC-AUC or PR curves focus on ranking/recall trade-offs.



---



## Summary



- MCC delivers a single, balanced view of classification performance from −1 to 1.

- In Python 3.13, compute it with matthews_corrcoef and visualise how it changes with the threshold.

- Report MCC alongside Accuracy, F1, and PR metrics to avoid misleading conclusions on imbalanced datasets.

---

