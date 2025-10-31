---
title: "Regresi Softmax"
pre: "2.2.2 "
weight: 2
title_suffix: "Memperkirakan probabilitas setiap kelas sekaligus"
---

{{% summary %}}
- Regresi softmax menggeneralisasi regresi logistik ke kasus multikelas dengan menghasilkan probabilitas untuk semua kelas secara bersamaan.
- Keluaran berada di rentang \\([0, 1]\\) dan jumlahnya 1, sehingga mudah dipakai dalam penentuan ambang, aturan berbasis biaya, maupun pipeline lanjutan.
- Pelatihan meminimalkan loss entropi silang, langsung mengoreksi perbedaan antara distribusi yang diprediksi dan distribusi sebenarnya.
- Di scikit-learn, `LogisticRegression(multi_class="multinomial")` menyediakan implementasi regresi softmax dan mendukung regularisasi L1/L2.
{{% /summary %}}

## Intuisi
Untuk klasifikasi biner, fungsi sigmoid memberikan probabilitas kelas 1. Pada multikelas kita membutuhkan probabilitas semua kelas sekaligus. Regresi softmax menghitung skor linear tiap kelas, mengeksponensialkannya, lalu menormalkan agar membentuk distribusi probabilitas. Skor tinggi diperbesar, skor rendah dikecilkan.

## Formulasi matematis
Misalkan terdapat \\(K\\) kelas, dengan parameter \\(\mathbf{w}_k\\) dan \\(b_k\\) untuk kelas \\(k\\). Maka

$$
P(y = k \mid \mathbf{x}) =
\frac{\exp\left(\mathbf{w}_k^\top \mathbf{x} + b_k\right)}
{\sum_{j=1}^{K} \exp\left(\mathbf{w}_j^\top \mathbf{x} + b_j\right)}.
$$

Fungsi objektifnya adalah loss entropi silang

$$
L = - \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}(y_i = k) \log P(y = k \mid \mathbf{x}_i),
$$

dengan opsi menambahkan regularisasi untuk menekan overfitting.

## Eksperimen dengan Python
Cuplikan berikut melatih regresi softmax pada data sintetis tiga kelas dan memvisualisasikan wilayah keputusannya. Mengaktifkan parameter `multi_class="multinomial"` sudah cukup untuk memakai formulasi softmax.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Membuat dataset 3 kelas
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_classes=3,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

# Regresi softmax (regresi logistik multikelas)
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
clf.fit(X, y)

# Menyusun grid untuk visualisasi
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200),
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Visualisasi
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.title("Wilayah keputusan regresi softmax")
plt.xlabel("fitur 1")
plt.ylabel("fitur 2")
plt.show()
```

![softmax block 1](/images/basic/classification/softmax_block01.svg)

## Referensi
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
