---
title: k-means
weight: 1
pre: "2.5.1 "
searchtitle: "Ejecutando k-means en Python"
---

<div class="pagetop-box">
    <p>k-means es un tipo de algoritmo de agrupamiento. Para agrupar los datos proporcionados en k clústeres, el algoritmo calcula repetidamente el promedio de cada clúster y asigna los datos al punto representativo más cercano.</p>
</div>


```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
```

## Ejemplo de agrupamiento k-means
### Crear datos de muestra
Crea datos de muestra con `make_blobs` y aplícalos al agrupamiento.

```python
n_samples = 1000
random_state = 117117
n_clusters = 4
X, y = make_blobs(
    n_samples=n_samples, random_state=random_state, cluster_std=1.5, centers=8
)

# Valor inicial del centroide de cada clúster de k-means
init = X[np.where(X[:, 1] < -8)][:n_clusters]

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c="k", marker="x", label="training dataset")
plt.legend()
plt.show()
```


    
![png](/images/basic/clustering/k-means1_files/k-means1_5_0.png)
    


### Cálculo del Centroide
Podemos recalcular el centroide cuatro veces y observar que los datos están claramente separados en la cuarta iteración.


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
    


## Cuando los clústeres se superponen
Verifica los resultados del agrupamiento cuando los datos de los clústeres se superponen.


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
    

## Influencia del número de k

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
    

