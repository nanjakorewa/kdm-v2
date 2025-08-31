---
title: "Análisis discriminante lineal"
pre: "2.2.2 "
weight: 2
---

<div class="pagetop-box">
    <p>El Análisis Discriminante Lineal (LDA) es un método para trazar una frontera que puede discriminar entre clases basándose en el grado de cohesión de los datos entre las clases y el grado de variabilidad de los datos entre las clases para dos clases de datos. También permite reducir la dimensionalidad de los datos en función de los resultados obtenidos. En esta página, visualizaremos las fronteras de decisión obtenidas por el LDA y visualizaremos los resultados de la reducción de la dimensionalidad.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

## Generar conjunto de datos


```python
n_samples = 200
X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=2)
X[:, 0] -= np.mean(X[:, 0])
X[:, 1] -= np.mean(X[:, 1])

fig = plt.figure(figsize=(7, 7))
plt.title("Scatter plots of data", fontsize=20)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```


    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_5_0.png)
    


## Encontrar los límites de decisión mediante el análisis lineal discriminanteEncontrar los límites de decisión mediante el análisis lineal discriminante

{{% notice document %}}
[sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
{{% /notice %}}


```python
# Encontrar el límite de decisión
clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X, y)

# Comprobar el límite de decisión
w = clf.coef_[0]
wt = -1 / (w[1] / w[0])  ## Encuentra la pendiente perpendicular a w
xs = np.linspace(-10, 10, 100)
ys_w = [(w[1] / w[0]) * xi for xi in xs]
ys_wt = [wt * xi for xi in xs]

fig = plt.figure(figsize=(7, 7))
plt.title("Visualize the slope of the decision boundary", fontsize=20)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.plot(xs, ys_w, "-.", color="k", alpha=0.5)  # orientación de w
plt.plot(xs, ys_wt, "--", color="k")  # Orientación perpendicular a w

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

# El resultado de la transferencia de datos a una dimensión a partir del vector w obtenido
X_1d = clf.transform(X).reshape(1, -1)[0]
fig = plt.figure(figsize=(7, 7))
plt.title("Ubicación de los datos cuando se proyectan en una dimensión", fontsize=15)
plt.scatter(X_1d, [0 for _ in range(n_samples)], c=y)
plt.show()
```


    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_7_0.png)
    



    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_7_1.png)
    


## Ejemplo con datos de más de 2 dimensiones


```python
X_3d, y_3d = make_blobs(n_samples=200, centers=3, n_features=3, cluster_std=3)

# Distribución de los datos de la muestra
fig = plt.figure(figsize=(7, 7))
plt.title("Gráficos de dispersión de datos", fontsize=20)
ax = fig.add_subplot(projection="3d")
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_3d)
plt.show()

# Aplicar LDA
clf_3d = LinearDiscriminantAnalysis()
clf_3d.fit(X_3d, y_3d)
X_2d = clf_3d.transform(X_3d)

# Ubicación de los datos cuando se proyectan en dos dimensiones
fig = plt.figure(figsize=(7, 7))

plt.title("Ubicación de los datos cuando se proyectan en dos dimensiones", fontsize=15)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_3d)
plt.show()
```


    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_9_0.png)
    



    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_9_1.png)
    

