---
title: "Softmax Classification"
pre: "2.2.2 "
weight: 2
title_suffix: "Multiclass in Python"
---

<div class="pagetop-box">
  <p><b>Softmax classification</b> extends logistic regression to <b>multiclass</b> problems. For two classes it reduces to logistic regression; for three or more classes it yields a valid probability over classes.</p>
</div>

---

## 1. Softmax function

Given scores (logits) \\(z=(z_1,\dots,z_K)\\), the softmax converts them to a probability vector:

$$
\mathrm{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)} \quad (i=1,\dots,K)
$$

- Output is in <b>[0,1]</b>  
- Sums to <b>1</b> across classes  
- Can be interpreted as class probabilities

---

## 2. Model

For input \\(x\\), class-\\(k\\) score is

$$
z_k = w_k^\top x + b_k
$$

Softmax yields

$$
P(y=k\mid x) = \frac{\exp(w_k^\top x + b_k)}{\sum_{j=1}^{K} \exp(w_j^\top x + b_j)}
$$

Training typically minimizes multinomial cross-entropy.

---

## 3. Try it in Python

Use scikit-learn’s `LogisticRegression` with `multi_class="multinomial"`.

{{% notice document %}}
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# synthetic 3-class data
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_classes=3,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
clf.fit(X, y)

# decision regions
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.title("Softmax classification (multinomial logistic)")
plt.show()
```

---

## 4. Notes and cautions

- Outputs are probabilities, suitable for expected-utility decisions  
- Linear decision boundaries in feature space  
- Sensitive to feature scaling; add interactions/nonlinear transforms if needed

---

## 5. Typical uses

- <b>Text classification</b> (multiclass topics)  
- <b>Image classification</b> (digits 0–9, etc.)  
- <b>User intent</b> (select one of several intents)

---

## Summary

- Softmax generalizes logistic regression to multiclass.  
- Produces a valid probability distribution.  
- In scikit-learn, set `multi_class="multinomial"` for the true softmax model.  
- Simple yet strong baseline for many multiclass tasks.

---

