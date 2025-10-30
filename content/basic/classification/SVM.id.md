---
title: "Support Vector Machine (SVM)"
pre: "2.2.5 "
weight: 5
title_suffix: "Meningkatkan generalisasi dengan margin maksimum"
---

{{% summary %}}
- SVM mempelajari batas keputusan yang memaksimalkan margin antar kelas, sehingga menekankan kemampuan generalisasi.
- Margin lunak memperkenalkan variabel slack; parameter \(C\) mengatur kompromi antara lebar margin dan jumlah kesalahan yang diizinkan.
- Kernel trick mengganti hasil kali dalam dengan fungsi kernel, memungkinkan batas keputusan nonlinier tanpa ekspansi fitur eksplisit.
- Penyetaraan fitur dan pencarian hiperparameter (\(C\), \(\gamma\), dsb.) penting untuk kinerja yang baik.
{{% /summary %}}

## Intuisi
Di antara semua hiperplan yang memisahkan kelas, SVM memilih yang memberikan margin terlebar terhadap sampel pelatihan. Titik yang menyentuh margin disebut support vector; hanya mereka yang menentukan batas akhir, sehingga model cukup tahan terhadap noise.

## Formulasi matematis
Untuk data yang dapat dipisahkan secara linear, kita menyelesaikan

$$
\min_{\mathbf{w}, b} \ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1.
$$

Pada praktiknya digunakan SVM margin lunak dengan variabel slack \(\xi_i \ge 0\):

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}}
\ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 + C \sum_{i=1}^{n} \xi_i
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i.
$$

Mengganti hasil kali \(\mathbf{x}_i^\top \mathbf{x}_j\) dengan kernel \(K(\mathbf{x}_i, \mathbf{x}_j)\) memungkinkan batas keputusan nonlinier.

## Eksperimen dengan Python
Cuplikan berikut melatih SVM dengan kernel linear dan kernel RBF pada data `make_moons` yang tidak separabel secara linear. Kernel RBF jauh lebih mampu menangkap batas melengkung.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Data nonlinier
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)

# Kernel linear
linear_clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
linear_clf.fit(X, y)

# Kernel RBF
rbf_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5.0, gamma=0.5))
rbf_clf.fit(X, y)

print("Statistik kernel linear:")
print(classification_report(y, linear_clf.predict(X)))

print("Statistik kernel RBF:")
print(classification_report(y, rbf_clf.predict(X)))

# Visualisasi batas keputusan RBF
grid_x, grid_y = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200),
)
grid = np.c_[grid_x.ravel(), grid_y.ravel()]

rbf_scores = rbf_clf.predict(grid).reshape(grid_x.shape)

plt.figure(figsize=(6, 5))
plt.contourf(grid_x, grid_y, rbf_scores, alpha=0.2, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
plt.title("Batas keputusan SVM dengan kernel RBF")
plt.xlabel("fitur 1")
plt.ylabel("fitur 2")
plt.tight_layout()
plt.show()
```

![svm block 1](/images/basic/classification/svm_block01.svg)

## Referensi
{{% references %}}
<li>Vapnik, V. (1998). <i>Statistical Learning Theory</i>. Wiley.</li>
<li>Smola, A. J., &amp; Schölkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199–222.</li>
{{% /references %}}
