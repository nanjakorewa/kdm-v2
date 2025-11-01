---
title: "Regresión softmax | Modelos multiclase con entropía cruzada"
linkTitle: "Regresión softmax"
seo_title: "Regresión softmax | Modelos multiclase con entropía cruzada"
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
Sea \\(K\\) el número de clases, \\(\mathbf{w}_k\\) y \\(b_k\\) los parámetros de la clase \\(k\\). Entonces

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
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run_softmax_regression_demo(
    n_samples: int = 300,
    n_classes: int = 3,
    random_state: int = 42,
    label_title: str = "Región de decisión de regresión softmax",
    xlabel: str = "característica 1",
    ylabel: str = "característica 2",
) -> dict[str, float]:
    """Train a softmax regression model and visualise decision regions."""
    japanize_matplotlib.japanize()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state,
    )

    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
    clf.fit(X, y)

    accuracy = float(accuracy_score(y, clf.predict(X)))

    x1_min, x1_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    x2_min, x2_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    grid_x1, grid_x2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 400),
        np.linspace(x2_min, x2_max, 400),
    )
    grid_points = np.c_[grid_x1.ravel(), grid_x2.ravel()]
    preds = clf.predict(grid_points).reshape(grid_x1.shape)

    cmap = ListedColormap(["#ff9896", "#98df8a", "#aec7e8", "#f7b6d2", "#c5b0d5"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(grid_x1, grid_x2, preds, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, n_classes + 0.5, 1))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(label_title)
    legend = ax.legend(*scatter.legend_elements(), title="classes", loc="best")
    ax.add_artist(legend)
    fig.tight_layout()
    plt.show()

    return {"accuracy": accuracy}


metrics = run_softmax_regression_demo(
    label_title="Región de decisión de regresión softmax",
    xlabel="característica 1",
    ylabel="característica 2",
)
print(f"Precisión de entrenamiento: {metrics['accuracy']:.3f}")

```


![softmax regression demo](/images/basic/classification/softmax_block01_es.png)

## Referencias
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
