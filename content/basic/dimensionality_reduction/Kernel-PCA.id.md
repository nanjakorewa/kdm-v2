---
title: "Kernel-PCA | dijalankan dengan Python"
linkTitle: "Kernel-PCA"
seo_title: "Kernel-PCA | dijalankan dengan Python"
pre: "2.6.4 "
weight: 4
title_suffix: "dijalankan dengan Python"
---


```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_circles
```
## 実験用のデータ
{{% notice document %}}
[sklearn.datasets.make_circles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)
{{% /notice %}}


```python
X, y = make_circles(n_samples=400, factor=0.3, noise=0.15)
fig = plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y)
```



    <matplotlib.collections.PathCollection at 0x7fc8e1c9bbe0>




    
![png](/images/basic/dimensionality_reduction/Kernel-PCA_files/Kernel-PCA_4_1.png)
    


## KernelPCA
{{% notice document %}}
[sklearn.decomposition.KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html#sklearn.decomposition.KernelPCA)
{{% /notice %}}

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(kernel="rbf", degree=2, gamma=3)
X_kpca = kpca.fit_transform(X)

fig = plt.figure(figsize=(8, 8))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
```




    <matplotlib.collections.PathCollection at 0x7fc8e28323a0>




    
![png](/images/basic/dimensionality_reduction/Kernel-PCA_files/Kernel-PCA_6_1.png)
    

