---
title: "Regresión softmax"
pre: "2.2.2 "
weight: 2
title_suffix: "Predecir todas las probabilidades a la vez"
---

{{% summary %}}
- La regresión softmax generaliza la regresión logística al caso multiclase, produciendo simultáneamente la probabilidad de cada clase.
- Las salidas están entre 0 y 1 y suman 1, por lo que se integran fácilmente en umbrales de decisión, reglas con costes o pipelines posteriores.
- El entrenamiento minimiza la entropía cruzada, corrigiendo directamente la discrepancia entre la distribución predicha y la verdadera.
- En scikit-learn, `LogisticRegression(multi_class="multinomial")` implementa la regresión softmax y admite regularización L1/L2.
{{% /summary %}}

## Intuición
En el caso binario, la sigmoide entrega la probabilidad de la clase 1. Con varias clases necesitamos todas las probabilidades a la vez. La regresión softmax calcula un puntaje lineal por clase, lo exponentia y lo normaliza para obtener una verdadera distribución de probabilidad: puntajes altos se realzan y los bajos se atenúan.

## Formulación matemática
Sea \(K\) el número de clases, \(\mathbf{w}_k\) y \(b_k\) los parámetros de la clase \(k\). Entonces

$$
P(y = k \mid \mathbf{x}) =
\frac{\exp\left(\mathbf{w}_k^\top \mathbf{x} + b_k\right)}
{\sum_{j=1}^{K} \exp\left(\mathbf{w}_j^\top \mathbf{x} + b_j\right)}.
$$

La función objetivo es la entropía cruzada

$$
L = - \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}(y_i = k) \log P(y = k \mid \mathbf{x}_i),
$$

con la posibilidad de añadir regularización para evitar sobreajuste.

## Experimentos con Python
El siguiente código entrena una regresión softmax sobre un conjunto sintético de tres clases y dibuja las regiones de decisión. Basta con indicar `multi_class="multinomial"` para activar la formulación softmax.

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generar un conjunto de 3 clases
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_classes=3,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

# Regresión softmax (regresión logística multiclase)
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
clf.fit(X, y)

# Crear una malla para visualizar
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200),
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Mostrar
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.title("Regiones de decisión de la regresión softmax")
plt.xlabel("característica 1")
plt.ylabel("característica 2")
plt.show()
```

![softmax block 1](/images/basic/classification/softmax_block01.svg)

## Referencias
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
