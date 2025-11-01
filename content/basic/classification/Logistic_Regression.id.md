---
title: "Regresi logistik | Memodelkan probabilitas kelas dengan fungsi sigmoid"
linkTitle: "Regresi logistik"
seo_title: "Regresi logistik | Memodelkan probabilitas kelas dengan fungsi sigmoid"
pre: "2.2.1 "
weight: 1
title_suffix: "Menghitung probabilitas dengan sigmoid"
---

{{% summary %}}
- Regresi logistik memasukkan kombinasi linear fitur ke fungsi sigmoid untuk memperkirakan probabilitas bahwa label bernilai 1.
- Keluaran berada di rentang \\([0, 1]\\), sehingga ambang keputusan dapat diatur fleksibel dan koefisien dapat dibaca sebagai kontribusi terhadap log-odds.
- Pelatihan meminimalkan loss entropi silang (sama dengan memaksimalkan log-likelihood); regularisasi L1/L2 membantu menekan overfitting.
- Dengan `LogisticRegression` dari scikit-learn, pra-pemrosesan, pelatihan, hingga visualisasi garis keputusan dapat diselesaikan dalam beberapa baris kode.
{{% /summary %}}

## Intuisi
Regresi linear menghasilkan bilangan real, namun pada klasifikasi kita sering membutuhkan probabilitas “seberapa besar kemungkinan kelas 1?”. Regresi logistik menyelesaikannya dengan memasukkan skor linear \\(z = \mathbf{w}^\top \mathbf{x} + b\\) ke fungsi sigmoid \\(\sigma(z) = 1 / (1 + e^{-z})\\), sehingga keluar nilai yang dapat diinterpretasikan sebagai probabilitas. Dengan probabilitas tersebut, aturan sederhana seperti “prediksi 1 jika \\(P(y=1 \mid \mathbf{x}) > 0.5\\)” sudah cukup untuk mengklasifikasikan.

## Formulasi matematis
Probabilitas kelas 1 diberikan oleh

$$
P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + \exp\left(-(\mathbf{w}^\top \mathbf{x} + b)\right)}.
$$

Model dilatih dengan memaksimalkan log-likelihood

$$
\ell(\mathbf{w}, b) = \sum_{i=1}^{n} \Bigl[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \Bigr], \quad p_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b),
$$

atau, secara ekuivalen, meminimalkan entropi silang negatif. Regularisasi L2 menahan koefisien agar tidak terlalu besar, sedangkan L1 dapat meniadakan fitur yang tidak relevan.

## Eksperimen dengan Python
Contoh berikut menyesuaikan regresi logistik pada data sintetis dua dimensi dan memvisualisasikan garis keputusan yang dihasilkan. Berkat scikit-learn, seluruh proses pelatihan dan plotting hanya membutuhkan sedikit kode.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run_logistic_regression_demo(
    n_samples: int = 300,
    random_state: int = 2,
    label_class0: str = "kelas 0",
    label_class1: str = "kelas 1",
    label_boundary: str = "batas keputusan",
    title: str = "Batas regresi logistik",
) -> dict[str, float]:
    """Train logistic regression on a synthetic 2D dataset and visualise the boundary.

    Args:
        n_samples: Number of samples to generate.
        random_state: Seed for reproducible sampling.
        label_class0: Legend label for class 0.
        label_class1: Legend label for class 1.
        label_boundary: Legend label for the separating line.
        title: Title for the plot.

    Returns:
        Dictionary containing training accuracy and coefficients.
    """
    japanize_matplotlib.japanize()
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=random_state,
        n_clusters_per_class=1,
    )

    clf = LogisticRegression()
    clf.fit(X, y)

    accuracy = float(accuracy_score(y, clf.predict(X)))
    coef = clf.coef_[0]
    intercept = float(clf.intercept_[0])

    x1, x2 = X[:, 0], X[:, 1]
    grid_x1, grid_x2 = np.meshgrid(
        np.linspace(x1.min() - 1.0, x1.max() + 1.0, 200),
        np.linspace(x2.min() - 1.0, x2.max() + 1.0, 200),
    )
    grid = np.c_[grid_x1.ravel(), grid_x2.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(grid_x1.shape)

    cmap = ListedColormap(["#aec7e8", "#ffbb78"])
    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(grid_x1, grid_x2, probs, levels=20, cmap=cmap, alpha=0.4)
    ax.contour(grid_x1, grid_x2, probs, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(x1[y == 0], x2[y == 0], marker="o", edgecolor="k", label=label_class0)
    ax.scatter(x1[y == 1], x2[y == 1], marker="x", color="k", label=label_class1)
    ax.set_xlabel("fitur 1")
    ax.set_ylabel("fitur 2")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.colorbar(contour, ax=ax, label="P(class = 1)")
    fig.tight_layout()
    plt.show()

    return {
        "accuracy": accuracy,
        "coef_0": float(coef[0]),
        "coef_1": float(coef[1]),
        "intercept": intercept,
    }


metrics = run_logistic_regression_demo(
    label_class0="kelas 0",
    label_class1="kelas 1",
    label_boundary="batas keputusan",
    title="Batas regresi logistik",
)
print(f"Akurasi pelatihan: {metrics['accuracy']:.3f}")
print(f"Koefisien fitur 1: {metrics['coef_0']:.3f}")
print(f"Koefisien fitur 2: {metrics['coef_1']:.3f}")
print(f"Intersep: {metrics['intercept']:.3f}")

```


![logistic-regression demo](/images/basic/classification/logistic-regression_block01_id.png)

## Referensi
{{% references %}}
<li>Agresti, A. (2015). <i>Foundations of Linear and Generalized Linear Models</i>. Wiley.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
