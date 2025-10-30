---
title: "Máquinas de vectores de soporte (SVM)"
pre: "2.2.5 "
weight: 5
title_suffix: "Mejorar la generalización maximizando el margen"
---

{{% summary %}}
- SVM aprende una frontera de decisión que maximiza el margen entre clases, priorizando la capacidad de generalización.
- El margen blando introduce variables de holgura; el parámetro \(C\) controla el equilibrio entre anchura del margen y errores permitidos.
- El truco del kernel reemplaza productos internos por funciones kernel, permitiendo fronteras no lineales sin expandir explícitamente las características.
- La estandarización de características y la búsqueda de hiperparámetros (\(C\), \(\gamma\), etc.) son claves para un buen rendimiento.
{{% /summary %}}

## Intuición
Entre todas las hiperplanos que separan las clases, SVM elige el que deja el margen más ancho. Los puntos que tocan el margen son los vectores soporte: solo ellos determinan la frontera final, lo que aporta robustez ante ruido suave.

## Formulación matemática
Si los datos son separables linealmente, resolvemos

$$
\min_{\mathbf{w}, b} \ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1.
$$

En la práctica usamos la variante de margen blando con variables de holgura \(\xi_i \ge 0\):

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}}
\ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 + C \sum_{i=1}^{n} \xi_i
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i.
$$

Sustituir los productos internos \(\mathbf{x}_i^\top \mathbf{x}_j\) por un kernel \(K(\mathbf{x}_i, \mathbf{x}_j)\) permite modelar fronteras no lineales.

## Experimentos con Python
El código siguiente entrena SVM con kernel lineal y con kernel RBF sobre datos generados por `make_moons`, que no son separables linealmente. El kernel RBF captura la frontera curva con mayor precisión.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Datos no lineales
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)

# Kernel lineal
linear_clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
linear_clf.fit(X, y)

# Kernel RBF
rbf_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5.0, gamma=0.5))
rbf_clf.fit(X, y)

print("Estadísticas kernel lineal:")
print(classification_report(y, linear_clf.predict(X)))

print("Estadísticas kernel RBF:")
print(classification_report(y, rbf_clf.predict(X)))

# Visualizar la frontera con RBF
grid_x, grid_y = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200),
)
grid = np.c_[grid_x.ravel(), grid_y.ravel()]

rbf_scores = rbf_clf.predict(grid).reshape(grid_x.shape)

plt.figure(figsize=(6, 5))
plt.contourf(grid_x, grid_y, rbf_scores, alpha=0.2, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
plt.title("Frontera de decisión con SVM RBF")
plt.xlabel("característica 1")
plt.ylabel("característica 2")
plt.tight_layout()
plt.show()
```

![svm block 1](/images/basic/classification/svm_block01.svg)

## Referencias
{{% references %}}
<li>Vapnik, V. (1998). <i>Statistical Learning Theory</i>. Wiley.</li>
<li>Smola, A. J., &amp; Schölkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199–222.</li>
{{% /references %}}
