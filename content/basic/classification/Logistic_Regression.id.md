---
title: "Regresi Logistik"
pre: "2.2.1 "
weight: 1
searchtitle: "Regresi Logistik in python"
---

<div class="pagetop-box">
    <p>Regresi logistik adalah model untuk klasifikasi dua kelas. Metode ini mencoba untuk mengubah nilai numerik yang dihasilkan oleh model regresi linier menjadi probabilitas sehingga masalah klasifikasi dapat diselesaikan.
Pada halaman ini, saya akan menjalankan regresi logistik yang diimplementasikan di sikit-learn dan menggambar batas keputusannya (batas yang menjadi dasar klasifikasi).</p>
</div>

```python
import matplotlib.pyplot as plt
import numpy as np
```

## Visualisasi Fungsi Logit

```python
fig = plt.figure(figsize=(4, 8))
p = np.linspace(0.01, 0.999, num=100)
y = np.log(p / (1 - p))
plt.plot(p, y)

plt.xlabel("p")
plt.ylabel("y")
plt.axhline(y=0, color="k")
plt.ylim(-3, 3)
plt.grid()
plt.show()
```


    
![png](/images/basic/classification/Logistic_Regression_files/Logistic_Regression_4_0.png)
    


## Regresi Logistik

Pada bagian ini, saya mencoba melatih regresi Logistik pada data yang dibuat secara artifisial.

{{% notice document %}}
- [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
{{% /notice %}}


```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

## Dataset
X, Y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=1,
    random_state=2,
    n_clusters_per_class=1,
)

# Pelatihan LogisticRegression
clf = LogisticRegression()
clf.fit(X, Y)

# Temukan batas keputusan
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
b_1 = -w1 / w2
b_0 = -b / w2
xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
xd = np.array([xmin, xmax])
yd = b_1 * xd + b_0

# Plot Grafik
plt.figure(figsize=(10, 10))
plt.plot(xd, yd, "k", lw=1, ls="-")
plt.scatter(*X[Y == 0].T, marker="o")
plt.scatter(*X[Y == 1].T, marker="x")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()
```


![png](/images/basic/classification/Logistic_Regression_files/Logistic_Regression_6_0.png)
