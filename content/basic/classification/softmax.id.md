---
title: "Klasifikasi Softmax"
pre: "2.2.2 "
weight: 2
title_suffix: "Multikelas di Python"
---

<div class="pagetop-box">
  <p><b>Softmax</b> memperluas regresi logistik ke masalah <b>multikelas</b>. Untuk dua kelas setara dengan logistik; untuk tiga atau lebih menghasilkan probabilitas yang valid.</p>
</div>

---

## 1. Fungsi softmax

Diberikan \(z=(z_1,\dots,z_K)\):

$$
\mathrm{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)} \quad (i=1,\dots,K)
$$

- Keluaran di <b>[0,1]</b>  
- Menjumlah <b>1</b> antar kelas  
- Dapat ditafsir sebagai probabilitas

---

## 2. Model

Untuk \(x\), skor kelas \(k\):

$$
z_k = w_k^\top x + b_k
$$

Probabilitas:

$$
P(y=k\mid x) = \frac{\exp(w_k^\top x + b_k)}{\sum_{j=1}^{K} \exp(w_j^\top x + b_j)}
$$

Dilatih dengan cross-entropy multinomial.

---

## 3. Coba di Python

Gunakan `LogisticRegression` dengan `multi_class="multinomial"`.

{{% notice document %}}
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

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

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.title("Klasifikasi softmax (logistik multinomial)")
plt.show()
```

---

## 4. Catatan

- Probabilitas yang cocok untuk pengambilan keputusan  
- Batas keputusan linear di ruang fitur  
- Skala fitur penting; tambahkan transformasi nonlinier bila perlu

---

## 5. Penggunaan

- <b>Teks</b> (topik multikelas)  
- <b>Gambar</b> (digit 0–9, dll.)  
- <b>Niat pengguna</b> (satu dari beberapa)

---

## Ringkasan

- Softmax menggeneralisasi logistik ke multikelas.  
- Menghasilkan distribusi probabilitas yang valid.  
- Di scikit-learn, `multi_class="multinomial"`.  
- Baseline sederhana namun kuat untuk banyak tugas.

---

