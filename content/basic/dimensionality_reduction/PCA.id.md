---
title: "Principal Component Analysis | Menangkap variasi dengan sumbu ortogonal"
linkTitle: "PCA"
seo_title: "Principal Component Analysis | Menangkap variasi dengan sumbu ortogonal"
pre: "2.6.1 "
weight: 1
searchtitle: "Cara menggunakan PCA dalam python untuk reduksi dimensionalitas"
---

<div class="pagetop-box">
    <p>PCA (Principal Component Analysis) adalah algoritma tanpa pengawasan yang dapat dipandang sebagai algoritma reduksi dimensionalitas. Hal ini dapat dilihat sebagai metode penggabungan informasi ke dalam sejumlah kecil variabel agar tidak kehilangan informasi pada banyak variabel sebanyak mungkin.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
```

## Dataset untuk eksperimen

{{% notice document %}}
[sklearn.datasets.make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
{{% /notice %}}

```python
X, y = make_blobs(n_samples=600, n_features=3, random_state=117117)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
```


    
![png](/images/basic/dimensionality_reduction/PCA_files/PCA_4_1.png)
    


## Pengurangan dimensi menjadi dua dimensi dengan PCA

{{% notice document %}}
[sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
{{% /notice %}}


```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

fig = plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
```


![png](/images/basic/dimensionality_reduction/PCA_files/PCA_6_1.png)
    


## Lihat efek normalisasi

### Jumlah klaster 3, klaster yang tumpang tindih


```python
X, y = make_blobs(
    n_samples=200, n_features=3, random_state=11711, centers=3, cluster_std=2.0
)
X[:, 1] = X[:, 1] * 1000
X[:, 2] = X[:, 2] * 0.01
X_ss = StandardScaler().fit_transform(X)

# Plot sebaran data sebelum reduksi dimensionalitas
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.title("Plot sebaran data sebelum PCA")
plt.show()

# PCA tanpa normalisasi
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

# PCA dengan normalisasi
pca_ss = PCA(n_components=2).fit(X_ss)
X_pca_ss = pca_ss.transform(X_ss)

fig = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("PCA tanpa normalisasi")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, marker="x", alpha=0.6)
plt.subplot(122)
plt.title("PCA dengan normalisasi")
plt.scatter(X_pca_ss[:, 0], X_pca_ss[:, 1], c=y, marker="x", alpha=0.6)
```


    
![png](/images/basic/dimensionality_reduction/PCA_files/PCA_8_0.png)
    


    
![png](/images/basic/dimensionality_reduction/PCA_files/PCA_8_2.png)
    
