---
title: "k tetangga terdekat (k-NN)"
pre: "2.2.7 "
weight: 7
title_suffix: "Pembelajaran malas berbasis jarak"
---

{{% summary %}}
- k-NN menyimpan data latih dan memprediksi lewat voting mayoritas di antara \(k\) tetangga terdekat dari titik uji.
- Hiperparameter utamanya adalah jumlah tetangga \(k\) dan skema pembobotan jarak, yang relatif mudah ditelusuri.
- Secara alamiah mampu memodelkan batas keputusan nonlinier, tetapi kontras jarak menurun di dimensi tinggi (“kutukan dimensi”).
- Menyetarakan fitur atau memilih fitur penting membuat perhitungan jarak lebih stabil.
{{% /summary %}}

## Intuisi
Dengan asumsi “sampel yang berdekatan cenderung berbagi label”, k-NN mencari \(k\) contoh latih terdekat dan memutuskan label melalui voting (dapat diberi bobot sesuai jarak). Karena tidak membangun model eksplisit sebelumnya, metode ini disebut pembelajaran malas.

## Formulasi matematis
Untuk titik uji \(\mathbf{x}\), misalkan \(\mathcal{N}_k(\mathbf{x})\) adalah kumpulan \(k\) tetangga terdekat. Suara untuk kelas \(c\) dihitung sebagai

$$
v_c = \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \,\mathbb{1}(y_i = c),
$$

di mana bobot \(w_i\) bisa seragam atau bergantung pada jarak (misalnya kebalikan jarak). Kelas dengan suara terbanyak menjadi prediksi akhir.

## Eksperimen dengan Python
Kode berikut membandingkan akurasi validasi silang untuk beberapa nilai \(k\) dan memvisualisasikan batas keputusan pada data dua dimensi.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Dampak variasi k terhadap akurasi
X_full, y_full = make_blobs(
    n_samples=600,
    centers=3,
    cluster_std=[1.1, 1.0, 1.2],
    random_state=7,
)
ks = [1, 3, 5, 7, 11]
for k in ks:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
    scores = cross_val_score(model, X_full, y_full, cv=5)
    print(f"k={k}: akurasi CV={scores.mean():.3f} +/- {scores.std():.3f}")

# Visualisasi batas keputusan 2D
X_vis, y_vis = make_blobs(
    n_samples=450,
    centers=[(-2, 3), (1.8, 2.2), (0.8, -2.5)],
    cluster_std=[1.0, 0.9, 1.1],
    random_state=42,
)
vis_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))
vis_model.fit(X_vis, y_vis)

fig, ax = plt.subplots(figsize=(6, 4.5))
xx, yy = np.meshgrid(
    np.linspace(X_vis[:, 0].min() - 1.5, X_vis[:, 0].max() + 1.5, 300),
    np.linspace(X_vis[:, 1].min() - 1.5, X_vis[:, 1].max() + 1.5, 300),
)
grid = np.column_stack([xx.ravel(), yy.ravel()])
pred = vis_model.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, pred, levels=np.arange(0, 4) - 0.5, cmap="Pastel1", alpha=0.9)

scatter = ax.scatter(
    X_vis[:, 0],
    X_vis[:, 1],
    c=y_vis,
    cmap="Set1",
    edgecolor="#1f2937",
    linewidth=0.6,
)
ax.set_title("Batas keputusan k-NN (k=5, bobot jarak)")
ax.set_xlabel("fitur 1")
ax.set_ylabel("fitur 2")
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(alpha=0.15)

legend = ax.legend(
    handles=scatter.legend_elements()[0],
    labels=[f"kelas {i}" for i in range(len(np.unique(y_vis)))],
    loc="upper right",
    frameon=True,
)
legend.get_frame().set_alpha(0.9)

fig.tight_layout()
```

![knn block 1](/images/basic/classification/knn_block01.svg)

## Referensi
{{% references %}}
<li>Cover, T. M., &amp; Hart, P. E. (1967). Nearest Neighbor Pattern Classification. <i>IEEE Transactions on Information Theory</i>, 13(1), 21–27.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
