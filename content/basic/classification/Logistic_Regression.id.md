---
title: "Regresi Logistik"
pre: "2.2.1 "
weight: 1
title_suffix: "Menghitung probabilitas dengan sigmoid"
---

{{% summary %}}
- Regresi logistik memasukkan kombinasi linear fitur ke fungsi sigmoid untuk memperkirakan probabilitas bahwa label bernilai 1.
- Keluaran berada di rentang \\([0, 1]\\), sehingga ambang keputusan dapat diatur fleksibel dan koefisien dapat dibaca sebagai kontribusi terhadap log-odds.
- Pelatihan meminimalkan loss entropi silang (sama dengan memaksimalkan log-likelihood); regularisasi L1/L2 membantu menekan overfitting.
- Dengan `LogisticRegression` dari scikit-learn, pra-pemrosesan, pelatihan, hingga visualisasi garis keputusan dapat diselesaikan dalam beberapa baris kode.
{{% /summary %}}

## Intuisi
Regresi linear menghasilkan bilangan real, namun pada klasifikasi kita sering membutuhkan probabilitas “seberapa besar kemungkinan kelas 1?”. Regresi logistik menyelesaikannya dengan memasukkan skor linear \\(z = \mathbf{w}^\top \mathbf{x} + b\\) ke fungsi sigmoid \\(\sigma(z) = 1 / (1 + e^{-z})\\), sehingga keluar nilai yang dapat diinterpretasikan sebagai probabilitas. Dengan probabilitas tersebut, aturan sederhana seperti “prediksi 1 jika \\(P(y=1 \mid \mathbf{x}) > 0.5\\)” sudah cukup untuk mengklasifikasikan.

## Formulasi matematis
Probabilitas kelas 1 diberikan oleh

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + \exp\left(-(\mathbf{w}^\top \mathbf{x} + b)\right)}.
$$

Model dilatih dengan memaksimalkan log-likelihood

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^{n} \Bigl[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Bigr], \quad p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b),
$$

atau, secara ekuivalen, meminimalkan entropi silang negatif. Regularisasi L2 menahan koefisien agar tidak terlalu besar, sedangkan L1 dapat meniadakan fitur yang tidak relevan.

## Eksperimen dengan Python
Contoh berikut menyesuaikan regresi logistik pada data sintetis dua dimensi dan memvisualisasikan garis keputusan yang dihasilkan. Berkat scikit-learn, seluruh proses pelatihan dan plotting hanya membutuhkan sedikit kode.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Membuat dataset klasifikasi 2D
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=1,
    random_state=2,
    n_clusters_per_class=1,
)

# Melatih model
clf = LogisticRegression()
clf.fit(X, y)

# Menghitung garis keputusan
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
slope = -w1 / w2
intercept = -b / w2

xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
xd = np.array([xmin, xmax])
yd = slope * xd + intercept

# Visualisasi
plt.figure(figsize=(8, 8))
plt.plot(xd, yd, "k-", lw=1, label="garis keputusan")
plt.scatter(*X[y == 0].T, marker="o", label="kelas 0")
plt.scatter(*X[y == 1].T, marker="x", label="kelas 1")
plt.legend()
plt.title("Garis keputusan regresi logistik")
plt.show()
```

![logistic-regression block 2](/images/basic/classification/logistic-regression_block02.svg)

## Referensi
{{% references %}}
<li>Agresti, A. (2015). <i>Foundations of Linear and Generalized Linear Models</i>. Wiley.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
