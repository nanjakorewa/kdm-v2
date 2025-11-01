---
title: "AdaBoost (Classification) | Intuition, formulas, and practice"
linkTitle: "AdaBoost (Classification)"
seo_title: "AdaBoost (Classification) | Intuition, formulas, and practice"
pre: "2.4.3 "
weight: 3
title_suffix: "Intuition, formulas, and practice"
---

{{% youtube "1K-h4YzrnsY" %}}

<div class="pagetop-box">
  <p><b>AdaBoost</b> is a <b>boosting</b> method that focuses more on <b>hard</b> samples by reweighting misclassified points and fitting the next weak learner accordingly.</p>
</div>

{{% notice document %}}
- [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
{{% /notice %}}

## Algorithm (with formulas)
For rounds \\(t=1,\dots,T\\), fit a weak learner \\(h_t(x)\\) with weights \\(w_i^{(t)}\\). Weighted error:

\\(\displaystyle \varepsilon_t = \frac{\sum_i w_i^{(t)} \, \mathbf{1}[y_i \ne h_t(x_i)]}{\sum_i w_i^{(t)}}\\),
 coefficient \\(\displaystyle \alpha_t = \tfrac{1}{2}\ln \frac{1-\varepsilon_t}{\varepsilon_t}\\).

Update sample weights:
\\(\displaystyle w_i^{(t+1)} = w_i^{(t)} \exp( \alpha_t\, \mathbf{1}[y_i \ne h_t(x_i)] )\\) (then normalize).

Final classifier: \\(\displaystyle H(x) = \operatorname{sign}\big( \sum_{t=1}^T \alpha_t h_t(x) \big)\\)

---

## Training on synthetic data
```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

n_features = 20
X, y = make_classification(
    n_samples=2500,
    n_features=n_features,
    n_informative=10,
    n_classes=2,
    n_redundant=4,
    n_clusters_per_class=5,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

ab = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=10,
    learning_rate=1.0,
    random_state=117117,
)
ab.fit(X_train, y_train)
y_pred = ab.predict(X_test)
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
```

![png](/images/basic/ensemble/Adaboost_Classification_files/Adaboost_Classification_10_0.png)

---

## Hyperparameters

### learning_rate
Smaller → slower but steadier; larger → may hinder convergence. Trade-off with <b>n_estimators</b>.
```python
scores = []
learning_rate_list = np.linspace(0.01, 1, 50)
for lr in learning_rate_list:
    clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=10,
        learning_rate=lr,
        random_state=117117,
    ).fit(X_train, y_train)
    scores.append(roc_auc_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(5, 5))
plt.plot(learning_rate_list, scores)
plt.xlabel("learning rate")
plt.ylabel("ROC-AUC")
plt.grid()
plt.show()
```

![png](/images/basic/ensemble/Adaboost_Classification_files/Adaboost_Classification_13_0.png)

### n_estimators
More rounds increase capacity but risk overfitting and cost.
```python
scores = []
n_estimators_list = [int(ne) for ne in np.linspace(5, 70, 20)]
for ne in n_estimators_list:
    clf = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=ne,
        learning_rate=0.6,
        random_state=117117,
    ).fit(X_train, y_train)
    scores.append(roc_auc_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(5, 5))
plt.plot(n_estimators_list, scores)
plt.xlabel("n_estimators")
plt.ylabel("ROC-AUC")
plt.grid()
plt.show()
```

![png](/images/basic/ensemble/Adaboost_Classification_files/Adaboost_Classification_16_0.png)

### base_estimator
Compare decision trees with different depths.
```python
scores = []
bases = [DecisionTreeClassifier(max_depth=md) for md in [2,3,4,5,6]]
for base in bases:
    clf = AdaBoostClassifier(
        base_estimator=base, n_estimators=10, learning_rate=0.5, random_state=117117
    ).fit(X_train, y_train)
    scores.append(roc_auc_score(y_test, clf.predict(X_test)))

plt.figure(figsize=(6, 5))
idx = range(len(bases))
plt.bar(list(idx), scores)
plt.xticks(list(idx), [str(b) for b in bases], rotation=90)
plt.xlabel("base_estimator")
plt.ylabel("ROC-AUC")
plt.tight_layout()
plt.show()
```

---

## Takeaways
- Reweights hard samples to improve them in later rounds.
- Balance <b>learning_rate × n_estimators</b>.
- Weak learners are typically shallow trees; overly strong bases weaken boosting effects.

