---
title: k-means
weight: 1
pre: "2.5.1 "
searchtitle: "Menjalankan k-means dalam python"
---

<div class="pagetop-box">
    <p>k-means adalah jenis algoritma pengelompokan. Untuk mengelompokkan data yang diberikan ke dalam k klaster, proses penghitungan rata-rata setiap klaster dan menetapkan data ke titik representatif terdekat diulang.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
```

## contoh pengelompokan k-means
### membuat data sampel
Buat data sampel dengan `make_blobs` dan terapkan clustering padanya.

```python
n_samples = 1000
random_state = 117117
n_clusters = 4
X, y = make_blobs(
    n_samples=n_samples, random_state=random_state, cluster_std=1.5, centers=8
)

# Nilai awal centroid dari setiap klaster k-means
init = X[np.where(X[:, 1] < -8)][:n_clusters]

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c="k", marker="x", label="kumpulan data pelatihan")
plt.legend()
plt.show()
```


    
![png](/images/basic/clustering/k-means1_files/k-means1_5_0.png)
    

### Perhitungan Centroid

Kita bisa menghitung ulang centroid sebanyak empat kali dan melihat bahwa datanya terpisah dengan bersih pada keempat kalinya.

```python
plt.figure(figsize=(12, 12))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    max_iter = 1 + i
    km = KMeans(
        n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=1, random_state=1
    ).fit(X)
    cluster_centers = km.cluster_centers_
    y_pred = km.predict(X)

    plt.title(f"max_iter={max_iter}")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker="x")
    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c="r",
        marker="o",
        label="Cluster centroid",
    )
plt.tight_layout()
plt.show()
```


    
![png](/images/basic/clustering/k-means1_files/k-means1_7_0.png)
    


## Ketika cluster tumpang tindih

Periksa hasil pengelompokan ketika data cluster tumpang tindih.


```python
n_samples = 1000
random_state = 117117

plt.figure(figsize=(8, 8))
for i in range(4):
    cluster_std = (i + 1) * 1.5
    X, y = make_blobs(
        n_samples=n_samples, random_state=random_state, cluster_std=cluster_std
    )
    y_pred = KMeans(n_clusters=2, random_state=random_state, init="random").fit_predict(
        X
    )
    plt.subplot(2, 2, i + 1)
    plt.title(f"cluster_std={cluster_std}")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker="x")

plt.tight_layout()
plt.show()
```


    
![png](/images/basic/clustering/k-means1_files/k-means1_9_0.png)
    


## Influence of the number of k


```python
n_samples = 1000
random_state = 117117

plt.figure(figsize=(8, 8))
for i in range(4):
    n_clusters = (i + 1) * 2
    X, y = make_blobs(
        n_samples=n_samples, random_state=random_state, cluster_std=1, centers=5
    )
    y_pred = KMeans(
        n_clusters=n_clusters, random_state=random_state, init="random"
    ).fit_predict(X)
    plt.subplot(2, 2, i + 1)
    plt.title(f"#cluster={n_clusters}")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker="x")

plt.tight_layout()
plt.show()
```


    
![png](/images/basic/clustering/k-means1_files/k-means1_11_0.png)
    

