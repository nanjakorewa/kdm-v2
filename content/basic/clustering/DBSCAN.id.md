---
title: "DBSCAN"
pre: "2.5.4 "
weight: 4
title_suffix: "Clustering berbasis kepadatan"
searchtitle: "Contoh DBSCAN dengan Python"
---

{{% summary %}}
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) menganggap area padat sebagai klaster dan area renggang sebagai noise.
- Dua parameter, `eps` (radius pencarian) dan `min_samples` (jumlah tetangga minimal), membedakan titik inti, batas, dan noise.
- Tidak perlu menentukan jumlah klaster di awal dan cocok untuk bentuk klaster yang tidak konveks maupun data dengan noise.
- Grafik jarak terurut membantu memilih `eps`, sementara standardisasi fitur menjaga skala jarak tetap konsisten.
{{% /summary %}}

## Intuisi
DBSCAN menilai setiap titik berdasarkan banyaknya tetangga di radius `eps`.

- **Titik inti**: memiliki sedikitnya `min_samples` tetangga dalam radius `eps`.
- **Titik batas**: berada di sekitar titik inti tetapi tidak memenuhi syarat tetangga.
- **Noise**: bukan inti maupun batas.

Titik inti yang saling terhubung karena kepadatan membentuk suatu klaster. Titik batas menempel pada klaster terdekat, sedangkan titik tersisa diberi label noise.

## Rumusan
Untuk titik \\(x_i\\), tetangga \\(\varepsilon\\)-radius didefinisikan sebagai

$$
\mathcal{N}_\varepsilon(x_i) = \{\, x_j \in \mathcal{X} \mid \lVert x_i - x_j \rVert \le \varepsilon \,\}.
$$

Seorang titik menjadi inti bila

$$
|\mathcal{N}_\varepsilon(x_i)| \ge \texttt{min\_samples}.
$$

Konektivitas kepadatan mengelompokkan titik inti, titik batas mengikuti klaster terkait, dan sisanya tetap sebagai noise.

## Demonstrasi Python
Kita terapkan DBSCAN pada data berbentuk bulan sabit, lalu hitung jumlah klaster dan titik noise.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def jalankan_demo_dbscan(
    n_samples: int = 600,
    noise: float = 0.08,
    eps: float = 0.3,
    min_samples: int = 10,
    random_state: int = 0,
) -> dict[str, int]:
    """Menjalankan DBSCAN pada data bulan sabit dan mengembalikan jumlah klaster serta noise."""
    features, _ = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )
    features = StandardScaler().fit_transform(features)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)

    unique_labels = sorted(np.unique(labels))
    cluster_ids = [label for label in unique_labels if label != -1]
    noise_count = int(np.sum(labels == -1))

    core_mask = np.zeros(labels.shape[0], dtype=bool)
    if hasattr(model, "core_sample_indices_"):
        core_mask[model.core_sample_indices_] = True

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    palette = plt.cm.get_cmap("tab10", max(len(cluster_ids), 1))

    for order, cluster_id in enumerate(cluster_ids):
        mask = labels == cluster_id
        colour = palette(order)
        if np.any(mask & core_mask):
            ax.scatter(
                features[mask & core_mask, 0],
                features[mask & core_mask, 1],
                c=[colour],
                s=36,
                edgecolor="white",
                linewidth=0.2,
                label=f"cluster {cluster_id} (inti)",
            )
        if np.any(mask & ~core_mask):
            ax.scatter(
                features[mask & ~core_mask, 0],
                features[mask & ~core_mask, 1],
                c=[colour],
                s=24,
                edgecolor="white",
                linewidth=0.2,
                marker="o",
                label=f"cluster {cluster_id} (batas)",
            )

    if noise_count:
        noise_mask = labels == -1
        ax.scatter(
            features[noise_mask, 0],
            features[noise_mask, 1],
            c="#9ca3af",
            marker="x",
            s=28,
            linewidth=0.8,
            label="noise",
        )

    ax.set_title("Clustering DBSCAN")
    ax.set_xlabel("fitur 1")
    ax.set_ylabel("fitur 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    plt.show()

    return {"n_clusters": len(cluster_ids), "n_noise": noise_count}


hasil = jalankan_demo_dbscan()
print(f"Jumlah klaster yang ditemukan: {hasil['n_clusters']}")
print(f"Jumlah titik noise: {hasil['n_noise']}")
```


![Hasil clustering DBSCAN](/images/basic/clustering/dbscan_block01_id.png)

## Referensi
{{% references %}}
<li>Ester, M., Kriegel, H.-P., Sander, J., &amp; Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. <i>KDD</i>.</li>
<li>Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., &amp; Xu, X. (2017). DBSCAN Revisited, Revisited. <i>ACM Transactions on Database Systems</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
