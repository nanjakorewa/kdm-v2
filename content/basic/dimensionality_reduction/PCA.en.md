---
title: "Principal Component Analysis | Capturing Variance with Orthogonal Axes"
linkTitle: "PCA"
seo_title: "Principal Component Analysis | Capturing Variance with Orthogonal Axes"
pre: "2.6.1 "
weight: 1
searchtitle: "Using PCA in python for dimensionality reduction"
---

<div class="pagetop-box">
    <p>PCA (Principal Component Analysis) is an unsupervised algorithm that can be viewed as a dimensionality reduction algorithm. It can be seen as a method of combining information into a small number of variables so as not to lose information on many variables as much as possible.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
```

## Dataset for experiments

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
    


## Dimension reduction to two dimensions with PCA

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
    


## See the effect of normalization

### Number of clusters 3, overlapping clusters


```python
X, y = make_blobs(
    n_samples=200, n_features=3, random_state=11711, centers=3, cluster_std=2.0
)
X[:, 1] = X[:, 1] * 1000
X[:, 2] = X[:, 2] * 0.01
X_ss = StandardScaler().fit_transform(X)

# Scatter plots of data before dimensionality reduction
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.title("Scatter plots of data before PCA")
plt.show()

# PCA without normalization
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

# PCA with normalization
pca_ss = PCA(n_components=2).fit(X_ss)
X_pca_ss = pca_ss.transform(X_ss)

fig = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("without normalization")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, marker="x", alpha=0.6)
plt.subplot(122)
plt.title("with normalization")
plt.scatter(X_pca_ss[:, 0], X_pca_ss[:, 1], c=y, marker="x", alpha=0.6)
```


    
![png](/images/basic/dimensionality_reduction/PCA_files/PCA_8_0.png)
    


    
![png](/images/basic/dimensionality_reduction/PCA_files/PCA_8_2.png)
    
