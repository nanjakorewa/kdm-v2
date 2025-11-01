---
title: "Análisis de componentes principales | Capturar la varianza con ejes ortogonales"
linkTitle: "PCA"
seo_title: "Análisis de componentes principales | Capturar la varianza con ejes ortogonales"
pre: "2.6.1 "
weight: 1
title_suffix: "Explicación del funcionamiento"
---

{{% youtube "9sn0b3tml50" %}}

<div class="pagetop-box">
    <p><b>PCA (Análisis de Componentes Principales)</b> es uno de los algoritmos de aprendizaje no supervisado y puede considerarse un algoritmo de reducción de dimensionalidad. Es un método que consolida la información de muchas variables en un número reducido de variables, intentando perder la menor cantidad de información posible.</p>
</div>


```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
```

## Datos para experimentos

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
    


## Reducir la dimensionalidad a dos dimensiones con PCA

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
    

## Observar el efecto de la normalización
### Número de clústeres: 3, con superposición entre clústeres


```python
# Datos experimentales
X, y = make_blobs(
    n_samples=200, n_features=3, random_state=11711, centers=3, cluster_std=2.0
)
X[:, 1] = X[:, 1] * 1000
X[:, 2] = X[:, 2] * 0.01
X_ss = StandardScaler().fit_transform(X)

# Graficar los datos originales
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.title("Datos experimentales")
plt.show()

# PCA sin normalización
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

# PCA con normalización
pca_ss = PCA(n_components=2).fit(X_ss)
X_pca_ss = pca_ss.transform(X_ss)

fig = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Sin normalización")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, marker="x", alpha=0.6)
plt.subplot(122)
plt.title("Con normalización")
plt.scatter(X_pca_ss[:, 0], X_pca_ss[:, 1], c=y, marker="x", alpha=0.6)

```


    
![png](/images/basic/dimensionality_reduction/PCA_files/PCA_8_0.png)
    


    
![png](/images/basic/dimensionality_reduction/PCA_files/PCA_8_2.png)
    

### Número de clústeres: 6, sin superposición entre clústeres

```python
# Datos experimentales
X, y = make_blobs(
    n_samples=500, n_features=3, random_state=11711, centers=6, cluster_std=0.4
)
X[:, 1] = X[:, 1] * 1000
X[:, 2] = X[:, 2] * 0.01
X_ss = StandardScaler().fit_transform(X)

# Graficar los datos originales
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
plt.title("Datos experimentales")
plt.show()

# PCA sin normalización
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

# PCA con normalización
pca_ss = PCA(n_components=2).fit(X_ss)
X_pca_ss = pca_ss.transform(X_ss)

fig = plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Sin normalización")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, marker="x", alpha=0.6)
plt.subplot(122)
plt.title("Con normalización")
plt.scatter(X_pca_ss[:, 0], X_pca_ss[:, 1], c=y, marker="x", alpha=0.6)

```


    
![png](/images/basic/dimensionality_reduction/PCA_files/PCA_10_0.png)
    

![png](/images/basic/dimensionality_reduction/PCA_files/PCA_10_2.png)
    

