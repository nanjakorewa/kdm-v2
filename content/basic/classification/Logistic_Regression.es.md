---
title: "Regresión logística"
pre: "2.2.1 "
weight: 1
title_suffix: "Estimar probabilidades con la sigmoide"
---

{{% summary %}}
- La regresión logística aplica una combinación lineal de las entradas a la función sigmoide para predecir la probabilidad de que la etiqueta sea 1.
- La salida está en \([0, 1]\), lo que permite fijar umbrales de decisión con flexibilidad y leer los coeficientes como contribuciones al logit.
- El entrenamiento minimiza la entropía cruzada (equivale a maximizar la verosimilitud); la regularización L1/L2 ayuda a evitar el sobreajuste.
- Con `LogisticRegression` de scikit-learn se cubren el preprocesamiento, el ajuste y la visualización de la frontera de decisión en pocas líneas.
{{% /summary %}}

## Intuición
La regresión lineal produce valores reales, pero en clasificación suele interesar “¿cuál es la probabilidad de la clase 1?”. La regresión logística aborda el problema pasando el puntaje lineal \(z = \mathbf{w}^\top \mathbf{x} + b\) por la función sigmoide \(\sigma(z) = 1 / (1 + e^{-z})\), obteniendo valores con interpretación probabilística. Una regla simple, como “predecir 1 si \(P(y=1 \mid \mathbf{x}) > 0.5\)”, basta para clasificar.

## Formulación matemática
La probabilidad de la clase 1 dada \(\mathbf{x}\) es

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + \exp\left(-(\mathbf{w}^\top \mathbf{x} + b)\right)}.
$$

El aprendizaje maximiza la log-verosimilitud

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^{n} \Bigl[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Bigr], \quad p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b),
$$

o, de forma equivalente, minimiza la entropía cruzada negativa. Agregar regularización L2 evita coeficientes inestables, mientras que L1 puede anular características irrelevantes.

## Experimentos con Python
El siguiente ejemplo ajusta la regresión logística a un conjunto sintético bidimensional y visualiza la frontera resultante. Gracias a scikit-learn, entrenar y trazar la frontera requiere pocas líneas.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generar un conjunto de clasificación 2D
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=1,
    random_state=2,
    n_clusters_per_class=1,
)

# Entrenar el modelo
clf = LogisticRegression()
clf.fit(X, y)

# Calcular la frontera de decisión
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
slope = -w1 / w2
intercept = -b / w2

xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
xd = np.array([xmin, xmax])
yd = slope * xd + intercept

# Visualizar
plt.figure(figsize=(8, 8))
plt.plot(xd, yd, "k-", lw=1, label="frontera de decisión")
plt.scatter(*X[y == 0].T, marker="o", label="clase 0")
plt.scatter(*X[y == 1].T, marker="x", label="clase 1")
plt.legend()
plt.title("Frontera de la regresión logística")
plt.show()
```

![logistic-regression block 2](/images/basic/classification/logistic-regression_block02.svg)

## Referencias
{{% references %}}
<li>Agresti, A. (2015). <i>Foundations of Linear and Generalized Linear Models</i>. Wiley.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
