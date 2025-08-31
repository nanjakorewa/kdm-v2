---
title: k-means++
weight: 2
pre: "2.5.2 "
searchtitle: "Menjalankan k-means++ dalam python"
---

<div class="pagetop-box">
    <p>k-means adalah jenis algoritma pengelompokan. Untuk mengelompokkan data yang diberikan ke dalam k klaster, rata-rata setiap klaster dihitung ↔ data berulang kali ditugaskan ke titik perwakilan terdekat. k-means bervariasi dalam cara konvergensinya tergantung pada nilai awal dari k titik perwakilan.
    Konvergensi k-means tergantung pada nilai awal dari k titik representatif. k-means++ dapat digunakan untuk mendapatkan hasil yang lebih stabil dengan mengambil nilai awal dengan cara yang berbeda.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
```

## Membandingkan hasil pengelompokan antara k-means++ dan k-means (pemilihan nilai awal acak)

Bandingkan hasil pengelompokan antara random dan kmeans++.
Anda dapat melihat bahwa k-means++ dapat membagi cluster kiri atas menjadi dua cluster.
*Dalam contoh ini, kita berani menentukan `max_iter=1`.


```python
n_samples = 3000
random_state = 11711
X, y = make_blobs(
    n_samples=n_samples, random_state=random_state, cluster_std=1.5, centers=8
)

plt.figure(figsize=(8, 8))
for i in range(10):
    # Bandingkan k-means dan k-means++
    rand_index = np.random.permutation(1000)
    X_rand = X[rand_index]
    y_pred_rnd = KMeans(
        n_clusters=5, random_state=random_state, init="random", max_iter=1, n_init=1
    ).fit_predict(X_rand)
    y_pred_kpp = KMeans(
        n_clusters=5, random_state=random_state, init="k-means++", max_iter=1, n_init=1
    ).fit_predict(X_rand)

    plt.figure(figsize=(10, 2))
    plt.subplot(1, 2, 1)
    plt.title(f"random")
    plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_pred_rnd, marker="x")
    plt.subplot(1, 2, 2)
    plt.title(f"k-means++")
    plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_pred_kpp, marker="x")
    plt.show()

plt.tight_layout()
plt.show()
```


    
![png](/images/basic/clustering/k-means2_files/k-means2_5_1.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_2.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_3.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_4.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_5.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_6.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_7.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_8.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_9.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_10.png)
    
