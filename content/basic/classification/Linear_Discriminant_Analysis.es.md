---
title: "Anﾃ｡lisis discriminante lineal (LDA) | Aprender direcciones que separan clases"
linkTitle: "Anﾃ｡lisis discriminante lineal (LDA)"
seo_title: "Anﾃ｡lisis discriminante lineal (LDA) | Aprender direcciones que separan clases"
pre: "2.2.4 "
weight: 4
title_suffix: "Aprender direcciones que separan clases"
---

{{% summary %}}
- LDA busca direcciones que maximizan la razﾃｳn entre la varianza entre clases y la varianza intraclase, por lo que sirve tanto para clasificar como para reducir la dimensionalidad.
- La frontera de decisiﾃｳn es de la forma \\(\mathbf{w}^\top \mathbf{x} + b = 0\\); en 2D es una recta y en 3D un plano, lo que facilita su interpretaciﾃｳn geomﾃｩtrica.
- Si cada clase sigue una distribuciﾃｳn gaussiana con igual matriz de covarianza, LDA se aproxima al clasificador bayesiano ﾃｳptimo.
- Con `LinearDiscriminantAnalysis` de scikit-learn es sencillo visualizar la frontera de decisiﾃｳn y examinar las caracterﾃｭsticas proyectadas.
{{% /summary %}}

## Intuiciﾃｳn
LDA busca direcciones que mantengan cerca a los puntos de la misma clase y separen a los de clases distintas. Al proyectar los datos sobre esa direcciﾃｳn, las clases se vuelven mﾃ｡s distinguibles, por lo que LDA puede utilizarse directamente para clasificar o como paso previo de reducciﾃｳn de dimensionalidad.

## Formulaciﾃｳn matemﾃ｡tica
Para dos clases, la direcciﾃｳn de proyecciﾃｳn \\(\mathbf{w}\\) maximiza

$$
J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}},
$$

donde \\(\mathbf{S}_B\\) es la matriz de dispersiﾃｳn entre clases y \\(\mathbf{S}_W\\) la matriz de dispersiﾃｳn intraclase. En el caso multiclase se obtienen hasta \\(K-1\\) direcciones, ﾃｺtiles para reducir la dimensionalidad.

## Experimentos con Python
El cﾃｳdigo siguiente aplica LDA a un conjunto sintﾃｩtico de dos clases, dibuja la frontera de decisiﾃｳn y muestra la proyecciﾃｳn a una dimensiﾃｳn. Con `transform` podemos obtener directamente los datos proyectados.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


def run_lda_demo(
    n_samples: int = 200,
    random_state: int = 42,
    title_boundary: str = "Frontera de decisiﾃｳn de LDA",
    title_projection: str = "Proyecciﾃｳn unidimensional con LDA",
    xlabel: str = "caracterﾃｭstica 1",
    ylabel: str = "caracterﾃｭstica 2",
    hist_xlabel: str = "caracterﾃｭstica proyectada",
    class0_label: str = "clase 0",
    class1_label: str = "clase 1",
) -> dict[str, float]:
    """Train LDA on synthetic blobs and plot boundary plus projection."""
    japanize_matplotlib.japanize()
    X, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        n_features=2,
        cluster_std=2.0,
        random_state=random_state,
    )

    clf = LinearDiscriminantAnalysis(store_covariance=True)
    clf.fit(X, y)

    accuracy = float(accuracy_score(y, clf.predict(X)))
    w = clf.coef_[0]
    b = float(clf.intercept_[0])

    xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300)
    ys_boundary = -(w[0] / w[1]) * xs - b / w[1]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(title_boundary)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.8)
    ax.plot(xs, ys_boundary, "k--", lw=1.2, label="w^T x + b = 0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()

    X_proj = clf.transform(X)[:, 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(title_projection)
    ax.hist(X_proj[y == 0], bins=20, alpha=0.7, label=class0_label)
    ax.hist(X_proj[y == 1], bins=20, alpha=0.7, label=class1_label)
    ax.set_xlabel(hist_xlabel)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    plt.show()

    return {"accuracy": accuracy}


metrics = run_lda_demo(
    title_boundary="Frontera de decisiﾃｳn de LDA",
    title_projection="Proyecciﾃｳn unidimensional con LDA",
    xlabel="caracterﾃｭstica 1",
    ylabel="caracterﾃｭstica 2",
    hist_xlabel="caracterﾃｭstica proyectada",
    class0_label="clase 0",
    class1_label="clase 1",
)
print(f"Precisiﾃｳn de entrenamiento: {metrics['accuracy']:.3f}")

```


![linear-discriminant-analysis block 2](/images/basic/classification/linear-discriminant-analysis_block01_es.png)

## Referencias
{{% references %}}
<li>Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic Problems. <i>Annals of Eugenics</i>, 7(2), 179窶・88.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}

