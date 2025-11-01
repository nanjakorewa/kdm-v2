---
title: "Klasifikasi dengan pohon keputusan | Membagi data berdasarkan gain informasi"
linkTitle: "Pohon keputusan"
seo_title: "Klasifikasi dengan pohon keputusan | Membagi data berdasarkan gain informasi"
pre: "2.3.1 "
weight: 1
searchtitle: "Menjalankan pohon keputusan (klasifikasi) dalam python"
---

<div class="pagetop-box">
    <p>Pohon keputusan (klasifikasi) adalah jenis model yang menggunakan kombinasi aturan untuk mengklasifikasikan. Kumpulan aturan diwakili oleh grafik berbentuk pohon (struktur pohon), yang mudah diinterpretasikan. Halaman ini melakukan klasifikasi pohon keputusan dan selanjutnya memvisualisasikan pohon yang dihasilkan.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

## Membuat data sampel
Buat data sampel untuk klasifikasi 2 kelas.


```python
n_classes = 2
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_classes=n_classes,
    n_clusters_per_class=1,
)
```

## Membuat pohon keputusan
Latih model dengan `DecisionTreeClassifier(criterion="gini").fit(X, y)` untuk memvisualisasikan batas-batas keputusan dari pohon yang dibuat.
`kriteria="gini"` adalah opsi untuk menentukan indikator untuk menentukan percabangan.

{{% notice document %}}
[sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
{{% /notice %}}


```python
# Melatih Pengklasifikasi Pohon Keputusan
clf = DecisionTreeClassifier(criterion="gini").fit(X, y)

# Data untuk peta warna batas keputusan
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Memvisualisasikan batas-batas keputusan
plt.figure(figsize=(8, 8))
plt.tight_layout()
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)
plt.xlabel("x1")
plt.ylabel("x2")

# Plot warna terpisah untuk setiap label
for i, color, label_name in zip(range(n_classes), ["r", "b"], ["A", "B"]):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=label_name, cmap=plt.cm.Pastel1)

plt.legend()
plt.show()
```


    
![png](/images/basic/tree/Decision_Tree_Classifier_files/Decision_Tree_Classifier_7_0.png)
    


## Keluaran struktur pohon keputusan sebagai gambar

{{% notice document %}}
[sklearn.tree.plot_tree — scikit-learn 1.0.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
{{% /notice %}}


```python
plt.figure()
clf = DecisionTreeClassifier(criterion="gini").fit(X, y)
plt.figure(figsize=(12, 12))
plot_tree(clf, filled=True)
plt.show()
```

    
![png](/images/basic/tree/Decision_Tree_Classifier_files/Decision_Tree_Classifier_9_1.png)
    

