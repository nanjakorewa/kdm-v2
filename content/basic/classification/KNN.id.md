---
title: "k tetangga terdekat (k-NN)"
pre: "2.2.7 "
weight: 7
title_suffix: "Pembelajaran malas berbasis jarak"
---

{{% summary %}}
- k-NN menyimpan data latih dan memprediksi lewat voting mayoritas di antara \\(k\\) tetangga terdekat dari titik uji.
- Hiperparameter utamanya adalah jumlah tetangga \\(k\\) dan skema pembobotan jarak, yang relatif mudah ditelusuri.
- Secara alamiah mampu memodelkan batas keputusan nonlinier, tetapi kontras jarak menurun di dimensi tinggi (“kutukan dimensi”).
- Menyetarakan fitur atau memilih fitur penting membuat perhitungan jarak lebih stabil.
{{% /summary %}}

## Intuisi
Dengan asumsi “sampel yang berdekatan cenderung berbagi label”, k-NN mencari \\(k\\) contoh latih terdekat dan memutuskan label melalui voting (dapat diberi bobot sesuai jarak). Karena tidak membangun model eksplisit sebelumnya, metode ini disebut pembelajaran malas.

## Formulasi matematis
Untuk titik uji \\(\mathbf{x}\\), misalkan \\(\mathcal{N}_k(\mathbf{x})\\) adalah kumpulan \\(k\\) tetangga terdekat. Suara untuk kelas \\(c\\) dihitung sebagai

$$
v_c = \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \,\mathbb{1}(y_i = c),
$$

di mana bobot \\(w_i\\) bisa seragam atau bergantung pada jarak (misalnya kebalikan jarak). Kelas dengan suara terbanyak menjadi prediksi akhir.

## Eksperimen dengan Python
Kode berikut mengevaluasi beberapa nilai \\(k\\) menggunakan data validasi dan menggambarkan wilayah keputusan dari model terbaik.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_knn_demo(
    n_samples: int = 600,
    random_state: int = 7,
    weights: str = "distance",
    k_values: tuple[int, ...] = (1, 3, 5, 7, 11),
    validation_ratio: float = 0.3,
    title: str = "Wilayah keputusan k-NN",
    xlabel: str = "fitur 1",
    ylabel: str = "fitur 2",
    class_label_prefix: str = "kelas",
) -> dict[str, object]:
    """Evaluate k-NN for several neighbour counts and plot decision regions.

    Args:
        n_samples: Number of synthetic samples to draw.
        random_state: Seed for reproducible sampling.
        weights: Weighting scheme handed to KNeighborsClassifier.
        k_values: Candidate neighbour counts to evaluate.
        validation_ratio: Fraction of the data reserved for validation.
        title: Title for the generated figure.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        class_label_prefix: Prefix used when labelling the classes.

    Returns:
        Dictionary with validation scores per k and the best-performing k.
    """
    japanize_matplotlib.japanize()
    X, y = make_blobs(
        n_samples=n_samples,
        centers=3,
        cluster_std=[1.1, 1.0, 1.2],
        random_state=random_state,
    )

    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(X))
    split = int(len(X) * (1.0 - validation_ratio))
    train_idx, valid_idx = indices[:split], indices[split:]
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    scores: dict[int, float] = {}
    for k in k_values:
        model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=k, weights=weights),
        )
        model.fit(X_train, y_train)
        scores[k] = float(model.score(X_valid, y_valid))

    best_k = max(scores, key=scores.get)
    best_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=best_k, weights=weights),
    )
    best_model.fit(X, y)

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1.5, X[:, 0].max() + 1.5, 300),
        np.linspace(X[:, 1].min() - 1.5, X[:, 1].max() + 1.5, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    predictions = best_model.predict(grid).reshape(xx.shape)

    unique_classes = np.unique(y)
    levels = np.arange(unique_classes.min(), unique_classes.max() + 2) - 0.5
    cmap = ListedColormap(["#fee0d2", "#deebf7", "#c7e9c0"])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    contour = ax.contourf(xx, yy, predictions, levels=levels, cmap=cmap, alpha=0.85)
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="Set1",
        edgecolor="#1f2937",
        linewidth=0.6,
    )
    ax.set_title(f"{title} (k={best_k}, weights={weights})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.grid(alpha=0.15)

    legend = ax.legend(
        handles=scatter.legend_elements()[0],
        labels=[f"{class_label_prefix} {cls}" for cls in unique_classes],
        loc="upper right",
        frameon=True,
    )
    legend.get_frame().set_alpha(0.9)
    fig.colorbar(contour, ax=ax, label="Kelas prediksi")
    fig.tight_layout()
    plt.show()

    return {"scores": scores, "best_k": int(best_k), "validation_accuracy": scores[best_k]}


metrics = run_knn_demo(
    title="Wilayah keputusan k-NN",
    xlabel="fitur 1",
    ylabel="fitur 2",
    class_label_prefix="kelas",
)
print(f"k terbaik: {metrics['best_k']}")
print(f"Akurasi validasi (k terbaik): {metrics['validation_accuracy']:.3f}")
for candidate_k, score in metrics["scores"].items():
    print(f"k={candidate_k}: akurasi validasi={score:.3f}")

```


![Wilayah keputusan k-NN](/images/basic/classification/knn_block01_id.png)

## Referensi
{{% references %}}
<li>Cover, T. M., &amp; Hart, P. E. (1967). Nearest Neighbor Pattern Classification. <i>IEEE Transactions on Information Theory</i>, 13(1), 21–27.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
