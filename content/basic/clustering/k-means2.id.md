---
title: "k-means++ | Inisialisasi centroid cerdas untuk klastering k-means"
linkTitle: "k-means++"
seo_title: "k-means++ | Inisialisasi centroid cerdas untuk klastering k-means"
weight: 2
pre: "2.5.2 "
title_suffix: "Memilih centroid awal secara cerdas"
searchtitle: "Inisialisasi k-means++ dengan Python"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means++ menyebarkan centroid awal sehingga k-means lebih jarang berhenti pada solusi lokal yang buruk.
- Setiap centroid baru dipilih dengan peluang sebanding kuadrat jaraknya terhadap centroid yang sudah ada, sehingga seed tidak terkumpul di satu area.
- Dengan `KMeans(init="k-means++")` di `scikit-learn`, perbandingan dengan inisialisasi acak menjadi sangat mudah.
- Turunan skala besar seperti mini-batch k-means dibangun dari ide k-means++ dan lazim dipakai pada data streaming maupun dataset raksasa.
{{% /summary %}}

## Intuisi
Ketika semua centroid awal jatuh di wilayah padat yang sama, k-means mungkin konvergen cepat tetapi kualitas klaster tetap buruk (WCSS besar). k-means++ mengatasi hal tersebut: centroid pertama dipilih acak, sedangkan berikutnya dipilih lebih sering jika jauh dari centroid yang sudah ada.

## Formulasi matematis
Dengan centroid terpilih \\(\{\mu_1, \dots, \mu_m\}\\), probabilitas sebuah titik \\(x\\) menjadi centroid berikutnya adalah

$$
P(x) = \frac{D(x)^2}{\sum_{x' \in \mathcal{X}} D(x')^2}, \qquad
D(x) = \min_{1 \le j \le m} \lVert x - \mu_j \rVert.
$$

Titik yang jauh dari semua centroid eksisting (\\(D(x)\\) besar) memiliki kesempatan lebih tinggi, sehingga awalannya menyebar dengan baik (Arthur & Vassilvitskii, 2007).

## Demonstrasi Python
Contoh berikut membandingkan tiga percobaan dengan inisialisasi acak dan k-means++ hanya setelah satu iterasi k-means, agar perbedaannya jelas.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def buat_data_blobs(
    n_samples: int = 3000,
    n_centers: int = 8,
    cluster_std: float = 1.5,
    random_state: int = 11711,
) -> tuple[np.ndarray, np.ndarray]:
    """Membangun data sintetis untuk eksperimen inisialisasi."""
    return make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )


def bandingkan_inisialisasi(
    data: np.ndarray,
    n_clusters: int = 5,
    subset_size: int = 1000,
    n_trials: int = 3,
    random_state: int = 11711,
) -> dict[str, list[float]]:
    """Membandingkan inisialisasi acak vs k-means++ dan menampilkan hasilnya.

    Args:
        data: Matriks fitur yang akan diklaster.
        n_clusters: Jumlah klaster.
        subset_size: Banyaknya sampel per percobaan.
        n_trials: Jumlah percobaan komparatif.
        random_state: Seed untuk reproducibility.

    Returns:
        Kamus berisi nilai WCSS untuk kedua mode inisialisasi.
    """
    rng = np.random.default_rng(random_state)
    inersia_acak: list[float] = []
    inersia_kpp: list[float] = []

    fig, axes = plt.subplots(
        n_trials,
        2,
        figsize=(10, 3.2 * n_trials),
        sharex=True,
        sharey=True,
    )

    for percobaan in range(n_trials):
        indeks = rng.choice(len(data), size=subset_size, replace=False)
        subset = data[indeks]

        model_acak = KMeans(
            n_clusters=n_clusters,
            init="random",
            n_init=1,
            max_iter=1,
            random_state=random_state + percobaan,
        ).fit(subset)
        model_kpp = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=1,
            random_state=random_state + percobaan,
        ).fit(subset)

        inersia_acak.append(float(model_acak.inertia_))
        inersia_kpp.append(float(model_kpp.inertia_))

        ax_acak = axes[percobaan, 0] if n_trials > 1 else axes[0]
        ax_kpp = axes[percobaan, 1] if n_trials > 1 else axes[1]

        ax_acak.scatter(subset[:, 0], subset[:, 1], c=model_acak.labels_, s=10)
        ax_acak.set_title(f"Inisialisasi acak (percobaan {percobaan + 1})")
        ax_acak.grid(alpha=0.2)

        ax_kpp.scatter(subset[:, 0], subset[:, 1], c=model_kpp.labels_, s=10)
        ax_kpp.set_title(f"k-means++ (percobaan {percobaan + 1})")
        ax_kpp.grid(alpha=0.2)

    fig.suptitle("Label setelah satu iterasi (acak vs k-means++)")
    fig.tight_layout()
    plt.show()

    return {"acak": inersia_acak, "k-means++": inersia_kpp}


FITUR, _ = buat_data_blobs()
hasil = bandingkan_inisialisasi(
    data=FITUR,
    n_clusters=5,
    subset_size=1000,
    n_trials=3,
    random_state=2024,
)
for metode, nilai in hasil.items():
    print(f"Rata-rata WCSS {metode}: {np.mean(nilai):.1f}")
```


![Perbandingan inisialisasi](/images/basic/clustering/k-means2_block01_id.png)

## Referensi
{{% references %}}
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>Bahmani, B., Moseley, B., Vattani, A., Kumar, R., &amp; Vassilvitskii, S. (2012). Scalable k-means++. <i>VLDB</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
