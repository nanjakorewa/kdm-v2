---

title: "Confusion Matrix | Interpreting Classification Performance"

linkTitle: "Confusion Matrix"

seo_title: "Confusion Matrix | Interpreting Classification Performance"

pre: "4.3.0 "

weight: 0

---



{{< lead >}}

A confusion matrix provides a compact snapshot of how a classifier labels each class. Reading it alongside accuracy, precision, recall, and F1 score makes it easier to debug misclassifications.

{{< /lead >}}



---



## 1. Anatomy of a confusion matrix



For binary classification the matrix is a 2×2 table:



|              | Predicted: Negative | Predicted: Positive |

| ------------ | ------------------ | ------------------- |

| **Actual: Negative** | True Negative (TN)  | False Positive (FP)  |

| **Actual: Positive** | False Negative (FN) | True Positive (TP)   |



- Rows represent the ground truth, columns the model prediction.

- Inspecting TP / FP / FN / TN reveals whether the model is biased toward a specific class.



---



## 2. End-to-end example on Python 3.13



Make sure you are running **Python 3.13** and install the required libraries:



```bash

python --version  # e.g. Python 3.13.0

pip install scikit-learn matplotlib

```



The script below trains a logistic regression model on the breast cancer dataset, then prints and plots the confusion matrix. A `Pipeline` with `StandardScaler` keeps the optimisation stable and avoids convergence warnings.



```python

from pathlib import Path



import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)



pipeline = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=1000, solver="lbfgs"),

)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



cm = confusion_matrix(y_test, y_pred)

print(cm)



disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues", colorbar=False)

plt.tight_layout()

plt.show()

```



{{< figure src="/images/eval/confusion-matrix/binary_matrix.png" alt="Confusion matrix for the breast cancer dataset" caption="Confusion matrix rendered with scikit-learn (Python 3.13)" >}}



---



## 3. Normalising the matrix



When the dataset is imbalanced, normalising by row (actual labels) helps you compare error rates.



```python

cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

print(cm_norm)



disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)

disp_norm.plot(cmap="Blues", values_format=".2f", colorbar=False)

plt.tight_layout()

plt.show()

```



- `normalize="true"`: ratio within each actual class  

- `normalize="pred"`: ratio within each predicted class  

- `normalize="all"`: ratio over all observations  



---



## 4. Extending to multiclass problems



`ConfusionMatrixDisplay.from_predictions` automatically builds the matrix for multiclass tasks and adds axis labels.



```python

ConfusionMatrixDisplay.from_predictions(

    y_true=ground_truth_labels,

    y_pred=model_outputs,

    normalize="true",

    values_format=".2f",

    cmap="Blues",

)

plt.tight_layout()

plt.show()

```



---



## 5. Practical checkpoints



- **False negatives vs. false positives**: decide which error is more costly (e.g., medical diagnosis vs. fraud detection) and monitor the relevant cells closely.

- **Pair with heatmaps**: visual inspection highlights skewed classes and makes cross-team discussions easier.

- **Derive other metrics**: accuracy, precision, recall, and F1 can all be computed from the same matrix. Compare them with ROC-AUC or PR curves for a fuller picture.

- **Keep notebooks reproducible**: packaging the analysis in a Python 3.13 notebook enables fast iteration when you tune or retrain the model.



---



## Summary



- A confusion matrix summarises TP / FP / FN / TN and exposes the bias of a classifier.

- Normalising the matrix reveals error ratios when classes are imbalanced.

- Combine the matrix with derived metrics and business requirements to define actionable evaluation criteria.

