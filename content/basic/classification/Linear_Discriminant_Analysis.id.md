---
title: "Analisis Diskriminan Linier"
pre: "2.2.2 "
weight: 2
searchtitle: "Analisis Diskriminan Linier dalam python"
---

<div class="pagetop-box">
    <p>Linear Discriminant Analysis (LDA) adalah metode penarikan batas yang dapat mendiskriminasikan antar kelas berdasarkan derajat kohesi data antar kelas dan derajat variabilitas data antar kelas untuk dua kelas data. Hal ini juga memungkinkan dilakukannya reduksi dimensionalitas data berdasarkan hasil yang diperoleh. Pada halaman ini, kita akan memvisualisasikan batas-batas keputusan yang diperoleh LDA dan memvisualisasikan hasil pengurangan dimensionalitas.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

## Generate dataset


```python
n_samples = 200
X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=2)
X[:, 0] -= np.mean(X[:, 0])
X[:, 1] -= np.mean(X[:, 1])

fig = plt.figure(figsize=(7, 7))
plt.title("Plot sebaran data", fontsize=20)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```


    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_5_0.png)
    


## Temukan batas-batas keputusan dengan analisis diskriminan Linear
{{% notice document %}}
[sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
{{% /notice %}}


```python
# Temukan batas keputusan
clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X, y)

# Periksa batas keputusan
w = clf.coef_[0]
wt = -1 / (w[1] / w[0])  ## Temukan kemiringan yang tegak lurus terhadap w
xs = np.linspace(-10, 10, 100)
ys_w = [(w[1] / w[0]) * xi for xi in xs]
ys_wt = [wt * xi for xi in xs]

fig = plt.figure(figsize=(7, 7))
plt.title("Visualisasi kemiringan batas keputusan.", fontsize=20)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.plot(xs, ys_w, "-.", color="k", alpha=0.5)  # orientasi dari W
plt.plot(xs, ys_wt, "--", color="k")  # Orientasi tegak lurus terhadap w

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

# Lokasi data ketika memproyeksikan data dalam satu dimensi.
X_1d = clf.transform(X).reshape(1, -1)[0]
fig = plt.figure(figsize=(7, 7))
plt.title("Lokasi data ketika memproyeksikan data dalam satu dimensi.", fontsize=15)
plt.scatter(X_1d, [0 for _ in range(n_samples)], c=y)
plt.show()
```


    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_7_0.png)
    



    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_7_1.png)
    


## Contoh dengan data dalam lebih dari dua dimensi.


```python
X_3d, y_3d = make_blobs(n_samples=200, centers=3, n_features=3, cluster_std=3)

# Distribusi data sampel
fig = plt.figure(figsize=(7, 7))
plt.title("Distribusi data sampel", fontsize=20)
ax = fig.add_subplot(projection="3d")
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_3d)
plt.show()

# Terapkan LDA
clf_3d = LinearDiscriminantAnalysis()
clf_3d.fit(X_3d, y_3d)
X_2d = clf_3d.transform(X_3d)

# Lokasi data ketika data diproyeksikan dalam dua dimensi
fig = plt.figure(figsize=(7, 7))

plt.title("Lokasi data ketika data diproyeksikan dalam dua dimensi", fontsize=15)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_3d)
plt.show()
```


    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_9_0.png)
    



    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_9_1.png)
    

