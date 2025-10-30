---
title: "Análisis discriminante lineal (LDA)"
pre: "2.2.4 "
weight: 4
title_suffix: "Aprender direcciones que separan clases"
---

{{% summary %}}
- LDA busca direcciones que maximizan la razón entre la varianza entre clases y la varianza intraclase, por lo que sirve tanto para clasificar como para reducir la dimensionalidad.
- La frontera de decisión es de la forma \(\mathbf{w}^\top \mathbf{x} + b = 0\); en 2D es una recta y en 3D un plano, lo que facilita su interpretación geométrica.
- Si cada clase sigue una distribución gaussiana con igual matriz de covarianza, LDA se aproxima al clasificador bayesiano óptimo.
- Con `LinearDiscriminantAnalysis` de scikit-learn es sencillo visualizar la frontera de decisión y examinar las características proyectadas.
{{% /summary %}}

## Intuición
LDA busca direcciones que mantengan cerca a los puntos de la misma clase y separen a los de clases distintas. Al proyectar los datos sobre esa dirección, las clases se vuelven más distinguibles, por lo que LDA puede utilizarse directamente para clasificar o como paso previo de reducción de dimensionalidad.

## Formulación matemática
Para dos clases, la dirección de proyección \(\mathbf{w}\) maximiza

$$
J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}},
$$

donde \(\mathbf{S}_B\) es la matriz de dispersión entre clases y \(\mathbf{S}_W\) la matriz de dispersión intraclase. En el caso multiclase se obtienen hasta \(K-1\) direcciones, útiles para reducir la dimensionalidad.

## Experimentos con Python
El código siguiente aplica LDA a un conjunto sintético de dos clases, dibuja la frontera de decisión y muestra la proyección a una dimensión. Con `transform` podemos obtener directamente los datos proyectados.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

rs = 42
X, y = make_blobs(
    n_samples=200, centers=2, n_features=2, cluster_std=2, random_state=rs
)

clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X, y)

w = clf.coef_[0]
b = clf.intercept_[0]

xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
ys_boundary = -(w[0] / w[1]) * xs - b / w[1]

scale = 3.0
origin = X.mean(axis=0)
arrow_end = origin + scale * (w / np.linalg.norm(w))

plt.figure(figsize=(7, 7))
plt.title("Frontera de decisión y dirección de proyección de LDA", fontsize=16)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.8, label="muestras")
plt.plot(xs, ys_boundary, "k--", lw=1.2, label=r"$\mathbf{w}^\top \mathbf{x} + b = 0$")
plt.arrow(
    origin[0],
    origin[1],
    (arrow_end - origin)[0],
    (arrow_end - origin)[1],
    head_width=0.5,
    length_includes_head=True,
    color="k",
    alpha=0.7,
    label="dirección $\mathbf{w}$",
)
plt.xlabel("característica 1")
plt.ylabel("característica 2")
plt.legend()
plt.xlim(X[:, 0].min() - 2, X[:, 0].max() + 2)
plt.ylim(X[:, 1].min() - 2, X[:, 1].max() + 2)
plt.grid(alpha=0.25)
plt.show()

X_1d = clf.transform(X)[:, 0]

plt.figure(figsize=(8, 4))
plt.title("Proyección unidimensional con LDA", fontsize=15)
plt.hist(X_1d[y == 0], bins=20, alpha=0.7, label="clase 0")
plt.hist(X_1d[y == 1], bins=20, alpha=0.7, label="clase 1")
plt.xlabel("característica proyectada")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

![linear-discriminant-analysis block 2](/images/basic/classification/linear-discriminant-analysis_block02.svg)

## Referencias
{{% references %}}
<li>Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic Problems. <i>Annals of Eugenics</i>, 7(2), 179–188.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
