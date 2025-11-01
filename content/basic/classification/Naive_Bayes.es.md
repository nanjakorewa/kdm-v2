---
title: "Naive Bayes | Inferencia rﾃ｡pida con independencia condicional"
linkTitle: "Naive Bayes"
seo_title: "Naive Bayes | Inferencia rﾃ｡pida con independencia condicional"
pre: "2.2.6 "
weight: 6
title_suffix: "Inferencia rﾃ｡pida con independencia condicional"
---

{{% summary %}}
- Naive Bayes asume independencia condicional entre caracterﾃｭsticas y combina la probabilidad a priori con la verosimilitud mediante el teorema de Bayes.
- El entrenamiento y la inferencia son muy rﾃ｡pidos, lo que lo vuelve una potente lﾃｭnea base para datos dispersos y de alta dimensiﾃｳn como texto o spam.
- El suavizado de Laplace y las caracterﾃｭsticas TF-IDF ayudan frente a palabras no vistas y diferencias de frecuencia.
- Cuando la suposiciﾃｳn de independencia es demasiado fuerte, conviene aplicar selecciﾃｳn de caracterﾃｭsticas o ensamblarlo con otros modelos.
{{% /summary %}}

## Intuiciﾃｳn
El teorema de Bayes afirma que 窶徘rior ﾃ・verosimilitud 竏・posterior窶・ Si las caracterﾃｭsticas son condicionalmente independientes, la verosimilitud se factoriza como el producto de probabilidades individuales. Naive Bayes aprovecha esta aproximaciﾃｳn y ofrece estimaciones sﾃｳlidas incluso con pocos datos de entrenamiento.

## Formulaciﾃｳn matemﾃ｡tica
Para una clase \\(y\\) y un vector de caracterﾃｭsticas \\(\mathbf{x} = (x_1, \ldots, x_d)\\),

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y).
$$

Existen distintas variantes segﾃｺn el tipo de datos: el modelo multinomial para frecuencias de palabras, el bernoulli para presencia/ausencia y el gaussiano para valores continuos.

## Experimentos con Python
El ejemplo siguiente entrena un clasificador Naive Bayes multinomial sobre un subconjunto del conjunto 20 Newsgroups usando TF-IDF. Aun con miles de caracterﾃｭsticas el entrenamiento es veloz, y el informe de clasificaciﾃｳn resume el desempeﾃｱo.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB


def run_naive_bayes_demo(
    n_samples: int = 600,
    n_classes: int = 3,
    random_state: int = 0,
    title: str = "Regiones de decisiﾃｳn del Naive Bayes gaussiano",
    xlabel: str = "caracterﾃｭstica 1",
    ylabel: str = "caracterﾃｭstica 2",
) -> dict[str, float]:
    """Train Gaussian Naive Bayes on synthetic data and plot decision regions."""
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

    clf = GaussianNB()
    clf.fit(X, y)

    accuracy = float(accuracy_score(y, clf.predict(X)))
    conf = confusion_matrix(y, clf.predict(X))

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]
    preds = clf.predict(grid).reshape(grid_x.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(grid_x, grid_y, preds, alpha=0.25, cmap="coolwarm", levels=np.arange(-0.5, n_classes + 0.5, 1))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    plt.show()

    return {"accuracy": accuracy, "confusion": conf}


metrics = run_naive_bayes_demo(
    title="Regiones de decisiﾃｳn del Naive Bayes gaussiano",
    xlabel="caracterﾃｭstica 1",
    ylabel="caracterﾃｭstica 2",
)
print(f"Precisiﾃｳn de entrenamiento: {metrics['accuracy']:.3f}")
print("Matriz de confusiﾃｳn:")
print(metrics['confusion'])

```


## Referencias
{{% references %}}
<li>Manning, C. D., Raghavan, P., &amp; Schﾃｼtze, H. (2008). <i>Introduction to Information Retrieval</i>. Cambridge University Press.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}

