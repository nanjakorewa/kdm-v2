---
title: "Mﾃ｡quinas de vectores de soporte (SVM) | Mejorar la generalizaciﾃｳn maximizando el margen"
linkTitle: "Mﾃ｡quinas de vectores de soporte (SVM)"
seo_title: "Mﾃ｡quinas de vectores de soporte (SVM) | Mejorar la generalizaciﾃｳn maximizando el margen"
pre: "2.2.5 "
weight: 5
title_suffix: "Mejorar la generalizaciﾃｳn maximizando el margen"
---

{{% summary %}}
- SVM aprende una frontera de decisiﾃｳn que maximiza el margen entre clases, priorizando la capacidad de generalizaciﾃｳn.
- El margen blando introduce variables de holgura; el parﾃ｡metro \\(C\\) controla el equilibrio entre anchura del margen y errores permitidos.
- El truco del kernel reemplaza productos internos por funciones kernel, permitiendo fronteras no lineales sin expandir explﾃｭcitamente las caracterﾃｭsticas.
- La estandarizaciﾃｳn de caracterﾃｭsticas y la bﾃｺsqueda de hiperparﾃ｡metros (\\(C\\), \\(\gamma\\), etc.) son claves para un buen rendimiento.
{{% /summary %}}

## Intuiciﾃｳn
Entre todas las hiperplanos que separan las clases, SVM elige el que deja el margen mﾃ｡s ancho. Los puntos que tocan el margen son los vectores soporte: solo ellos determinan la frontera final, lo que aporta robustez ante ruido suave.

## Formulaciﾃｳn matemﾃ｡tica
Si los datos son separables linealmente, resolvemos

$$
\min_{\mathbf{w}, b} \ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1.
$$

En la prﾃ｡ctica usamos la variante de margen blando con variables de holgura \\(\xi_i \ge 0\\):

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}}
\ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 + C \sum_{i=1}^{n} \xi_i
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i.
$$

Sustituir los productos internos \\(\mathbf{x}_i^\top \mathbf{x}_j\\) por un kernel \\(K(\mathbf{x}_i, \mathbf{x}_j)\\) permite modelar fronteras no lineales.

## Experimentos con Python
El cﾃｳdigo siguiente entrena SVM con kernel lineal y con kernel RBF sobre datos generados por `make_moons`, que no son separables linealmente. El kernel RBF captura la frontera curva con mayor precisiﾃｳn.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def run_svm_demo(
    n_samples: int = 400,
    noise: float = 0.25,
    random_state: int = 42,
    title: str = "Regiﾃｳn de decisiﾃｳn del SVM con RBF",
    xlabel: str = "caracterﾃｭstica 1",
    ylabel: str = "caracterﾃｭstica 2",
) -> dict[str, float]:
    """Train linear and RBF SVMs and plot the RBF decision boundary."""
    japanize_matplotlib.japanize()
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)

    linear_clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
    linear_clf.fit(X, y)

    rbf_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5.0, gamma=0.5))
    rbf_clf.fit(X, y)

    linear_acc = float(accuracy_score(y, linear_clf.predict(X)))
    rbf_acc = float(accuracy_score(y, rbf_clf.predict(X)))

    grid_x, grid_y = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 400),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 400),
    )
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    rbf_scores = rbf_clf.predict(grid).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(grid_x, grid_y, rbf_scores, alpha=0.2, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()

    return {"linear_accuracy": linear_acc, "rbf_accuracy": rbf_acc}


metrics = run_svm_demo(
    title="Regiﾃｳn de decisiﾃｳn del SVM con RBF",
    xlabel="caracterﾃｭstica 1",
    ylabel="caracterﾃｭstica 2",
)
print(f"Precisiﾃｳn con kernel lineal: {metrics['linear_accuracy']:.3f}")
print(f"Precisiﾃｳn con kernel RBF: {metrics['rbf_accuracy']:.3f}")

```


![svm block 1](/images/basic/classification/svm_block01_es.png)

## Referencias
{{% references %}}
<li>Vapnik, V. (1998). <i>Statistical Learning Theory</i>. Wiley.</li>
<li>Smola, A. J., &amp; Schﾃｶlkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199窶・22.</li>
{{% /references %}}

