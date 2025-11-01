---
title: "LDA | dijalankan dengan Python"
linkTitle: "LDA"
seo_title: "LDA | dijalankan dengan Python"
pre: "2.6.3 "
weight: 3
title_suffix: "dijalankan dengan Python"
---

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs

## Data untuk Eksperimen
{{% notice document %}}
[sklearn.datasets.make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
{{% /notice %}}

```python
X, y = make_blobs(
    n_samples=600, n_features=3, random_state=11711, cluster_std=4, centers=3
)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y)
ax.set_xlabel("\\(x_1\\)")
ax.set_ylabel("\\(x_2\\)")
ax.set_zlabel("\\(x_3\\)")
```


    
![png](/images/basic/dimensionality_reduction/LDA_files/LDA_4_1.png)
    


## Reduksi Dimensi ke Dua Dimensi dengan LDA

{{% notice document %}}
[sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
{{% /notice %}}


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2).fit(X, y)
X_lda = lda.transform(X)

fig = plt.figure(figsize=(8, 8))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, alpha=0.5)
```



    
![png](/images/basic/dimensionality_reduction/LDA_files/LDA_6_1.png)
    


## PCAとLDAの比較


```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

fig = plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.5)
```


    
![png](/images/basic/dimensionality_reduction/LDA_files/LDA_8_1.png)
    

