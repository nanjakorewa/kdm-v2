---
title: k-means++
weight: 2
pre: "2.5.2 "
searchtitle: "Ejecutando k-means++ en Python"
---

<div class="pagetop-box">
    <p>k-means es un tipo de algoritmo de agrupamiento. Para agrupar los datos proporcionados en k clústeres, se calcula el promedio de cada clúster ↔ y los datos se asignan repetidamente al punto representativo más cercano. k-means varía en la forma en que converge dependiendo del valor inicial de los k puntos representativos.
    La convergencia de k-means depende de los valores iniciales de los puntos representativos. k-means++ puede usarse para obtener resultados más estables al elegir los valores iniciales de una manera diferente.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
```

## Comparar k-means y k-means++

k-means y k-means++ son métodos de agrupamiento similares, pero difieren en cómo seleccionan los valores iniciales de los centroides. 

- **k-means**: Selecciona aleatoriamente los valores iniciales de los centroides, lo que puede llevar a resultados inconsistentes o inestables dependiendo de las inicializaciones. Algunas ejecuciones pueden converger a soluciones subóptimas.
  
- **k-means++**: Mejora la selección inicial de los centroides al elegir puntos iniciales que están más espaciados, lo que generalmente conduce a una convergencia más rápida y resultados más estables. Reduce la probabilidad de encontrar soluciones subóptimas.


```python
n_samples = 3000
random_state = 11711
X, y = make_blobs(
    n_samples=n_samples, random_state=random_state, cluster_std=1.5, centers=8
)

plt.figure(figsize=(8, 8))
for i in range(10):
    # Compare k-means and k-means++
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
    
