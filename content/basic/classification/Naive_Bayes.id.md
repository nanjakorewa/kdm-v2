---
title: "Naive Bayes"
pre: "2.2.6 "
weight: 6
title_suffix: "Inferensi cepat dengan asumsi independensi"
---

{{% summary %}}
- Naive Bayes mengasumsikan fitur saling independen secara kondisional dan menggabungkan prior dengan likelihood menggunakan teorema Bayes.
- Pelatihan dan inferensinya sangat cepat, menjadikannya baseline kuat untuk data berdimensi tinggi dan jarang seperti teks atau deteksi spam.
- Pelapisan Laplace dan fitur TF-IDF membantu menghadapi kata yang belum pernah muncul serta ketimpangan frekuensi.
- Jika asumsi independensi terlalu kuat, gunakan seleksi fitur atau kombinasikan Naive Bayes dengan model lain.
{{% /summary %}}

## Intuisi
Teorema Bayes menyatakan bahwa 窶徘rior ﾃ・likelihood 竏・posterior窶・ Jika fitur independen secara kondisional, likelihood dapat dipisah menjadi hasil kali probabilitas per fitur. Naive Bayes memanfaatkan pendekatan ini sehingga tetap stabil meski data latih sedikit.

## Formulasi matematis
Untuk kelas \\(y\\) dan vektor fitur \\(\mathbf{x} = (x_1, \ldots, x_d)\\),

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y).
$$

Model likelihood berbeda cocok untuk jenis data berbeda: multinomial untuk frekuensi kata, bernoulli untuk hadir/tidak hadir, dan gaussian untuk nilai kontinu.

## Eksperimen dengan Python
Contoh berikut melatih Naive Bayes multinomial pada subset data 20 Newsgroups dengan fitur TF-IDF. Walaupun jumlah fiturnya besar, pelatihan tetap cepat dan laporan klasifikasi merangkum performanya.

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
    title: str = "Wilayah keputusan Naive Bayes Gaussian",
    xlabel: str = "fitur 1",
    ylabel: str = "fitur 2",
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
    title="Wilayah keputusan Naive Bayes Gaussian",
    xlabel="fitur 1",
    ylabel="fitur 2",
)
print(f"Akurasi pelatihan: {metrics['accuracy']:.3f}")
print("Matriks kebingungan:")
print(metrics['confusion'])

```


## Referensi
{{% references %}}
<li>Manning, C. D., Raghavan, P., &amp; Schﾃｼtze, H. (2008). <i>Introduction to Information Retrieval</i>. Cambridge University Press.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}

