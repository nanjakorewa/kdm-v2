---

title: "F1 Score | The harmonic mean of precision and recall"

linkTitle: "F1 Score"

seo_title: "F1 Score | The harmonic mean of precision and recall"

pre: "4.3.8 "

weight: 8

---



{{< lead >}}

The F1 score is the harmonic mean of precision and recall. It answers the question “how good is the model when both false alarms and misses matter?” Below we compute it in Python 3.13, inspect how it changes with the decision threshold, and see when to use other Fβ variants.

{{< /lead >}}



---



## 1. Definition



With precision \(P\) and recall \(R\), F1 is defined as





F_1 = 2 \cdot \frac{P \cdot R}{P + R}.





The general Fβ score gives more weight to recall (\(\beta > 1\)) or precision (\(\beta < 1\)):





F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}.





---



## 2. Computing F1 in Python 3.13



```bash

python --version        # e.g. Python 3.13.0

pip install scikit-learn matplotlib

```



```python

import numpy as np

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, f1_score, fbeta_score

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



print(classification_report(y_test, y_pred, digits=3))

print("F1:", f1_score(y_test, y_pred))

print("F0.5:", fbeta_score(y_test, y_pred, beta=0.5))

print("F2:", fbeta_score(y_test, y_pred, beta=2.0))

```



classification_report displays precision, recall, and F1 per class in one table.



---



## 3. How F1 varies with the threshold



Using probability outputs we can plot how F1 evolves as the decision threshold changes.



```python

from sklearn.metrics import f1_score, precision_recall_curve



proba = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, proba)

thresholds = np.append(thresholds, 1.0)

f1_scores = [

    f1_score(y_test, (proba >= t).astype(int))

    for t in thresholds

]

```



{{< figure src="/images/eval/classification/f1-score/f1_vs_threshold.png" alt="F1 score vs threshold" caption="Use the curve to locate the threshold that maximises F1, or to trade precision for recall as requirements change." >}}



- The peak indicates the best trade-off between precision and recall when both are equally important.

- Use F0.5 or F2 when you want to bias the trade-off toward precision or recall respectively.



---



## 4. Averaging strategies for multiclass



scikit-learn’s verage parameter lets you aggregate F1 for multiclass or multilabel data:



- macro — compute F1 per class and take the (unweighted) mean.

- weighted — average per-class F1 weighted by class support.

- micro — pool all predictions and recompute from the global confusion matrix.



```python

from sklearn.metrics import f1_score



f1_macro = f1_score(y_test, y_pred, average="macro")

f1_weighted = f1_score(y_test, y_pred, average="weighted")

```



For multilabel problems verage="samples" reports the mean per sample.



---



## Summary



- F1 balances precision and recall; plotting it across thresholds helps you choose the operating point.

- Fβ scores adapt the balance when either recall (β>1) or precision (β<1) must dominate.

- On multiclass tasks, specify the averaging strategy and review precision, recall, F1, and PR curves together to understand the classifier’s behaviour.

---

