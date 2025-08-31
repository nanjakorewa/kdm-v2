---
title: "ROC-AUC"
pre: "4.3.1 "
weight: 1
searchtitle: "plot ROC-AUC graph in python"
---

The area under the ROC curve is called AUC (Area Under the Curve) and is used as an evaluation index for classification models; the best is when the AUC is 1, and 0.5 for random and totally invalid models.

- ROC-AUC is a typical example of binary classification evaluation index
- 1 is the best, 0.5 is close to a completely random prediction
- Below 0.5 can be when the prediction is the opposite of the correct answer
- Plotting the ROC curve can help determine what the classification threshold should be


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
```

## Plot ROC Curve
{{% notice document %}}
[sklearn.metrics.roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
{{% /notice %}}

### Function to plot ROC Curve

```python
def plot_roc_curve(test_y, pred_y):
    """Plot ROC Curve from correct answers and predictions

    Args:
        test_y (ndarray of shape (n_samples,)): y
        pred_y (ndarray of shape (n_samples,)): Predicted value for y
    """
    # False Positive Rate, True Positive Rate
    fprs, tprs, thresholds = roc_curve(test_y, pred_y)

    # plot ROC
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle="-", c="k", alpha=0.2, label="ROC-AUC=0.5")
    plt.plot(fprs, tprs, color="orange", label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Fill in the area corresponding to the ROC-AUC score
    y_zeros = [0 for _ in tprs]
    plt.fill_between(fprs, y_zeros, tprs, color="orange", alpha=0.3, label="ROC-AUC")
    plt.legend()
    plt.show()
```

### Create a model and plot ROC Curve against sample data


```python
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    n_informative=4,
    n_clusters_per_class=3,
    random_state=RND,
)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.33, random_state=RND
)

model = RandomForestClassifier(max_depth=5)
model.fit(train_X, train_y)
pred_y = model.predict_proba(test_X)[:, 1]
plot_roc_curve(test_y, pred_y)
```


    
![png](/images/eval/classification/ROC-AUC_files/ROC-AUC_6_0.png)
    


### Calculate ROC-AUC
{{% notice document %}}
[sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
{{% /notice %}}


```python
from sklearn.metrics import roc_auc_score

roc_auc_score(test_y, pred_y)
```




    0.89069793083171


