---
title: "Support Vector Machine (SVM)"
pre: "2.2.5 "
weight: 5
title_suffix: "Meningkatkan generalisasi dengan margin maksimum"
---

{{% summary %}}
- SVM mempelajari batas keputusan yang memaksimalkan margin antar kelas, sehingga menekankan kemampuan generalisasi.
- Margin lunak memperkenalkan variabel slack; parameter \\(C\\) mengatur kompromi antara lebar margin dan jumlah kesalahan yang diizinkan.
- Kernel trick mengganti hasil kali dalam dengan fungsi kernel, memungkinkan batas keputusan nonlinier tanpa ekspansi fitur eksplisit.
- Penyetaraan fitur dan pencarian hiperparameter (\\(C\\), \\(\gamma\\), dsb.) penting untuk kinerja yang baik.
{{% /summary %}}

## Intuisi
Di antara semua hiperplan yang memisahkan kelas, SVM memilih yang memberikan margin terlebar terhadap sampel pelatihan. Titik yang menyentuh margin disebut support vector; hanya mereka yang menentukan batas akhir, sehingga model cukup tahan terhadap noise.

## Formulasi matematis
Untuk data yang dapat dipisahkan secara linear, kita menyelesaikan

$$
\min_{\mathbf{w}, b} \ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1.
$$

Pada praktiknya digunakan SVM margin lunak dengan variabel slack \\(\xi_i \ge 0\\):

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}}
\ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 + C \sum_{i=1}^{n} \xi_i
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i.
$$

Mengganti hasil kali \\(\mathbf{x}_i^\top \mathbf{x}_j\\) dengan kernel \\(K(\mathbf{x}_i, \mathbf{x}_j)\\) memungkinkan batas keputusan nonlinier.

## Eksperimen dengan Python
Cuplikan berikut melatih SVM dengan kernel linear dan kernel RBF pada data `make_moons` yang tidak separabel secara linear. Kernel RBF jauh lebih mampu menangkap batas melengkung.

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
    title: str = "Wilayah keputusan SVM RBF",
    xlabel: str = "fitur 1",
    ylabel: str = "fitur 2",
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
    title="Wilayah keputusan SVM RBF",
    xlabel="fitur 1",
    ylabel="fitur 2",
)
print(f"Akurasi kernel linier: {metrics['linear_accuracy']:.3f}")
print(f"Akurasi kernel RBF: {metrics['rbf_accuracy']:.3f}")

```


![svm block 1](/images/basic/classification/svm_block01_id.png)

## Referensi
{{% references %}}
<li>Vapnik, V. (1998). <i>Statistical Learning Theory</i>. Wiley.</li>
<li>Smola, A. J., &amp; Schﾃｶlkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199窶・22.</li>
{{% /references %}}

