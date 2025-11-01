---
title: "X-means | Mengestimasi jumlah klaster secara otomatis"
linkTitle: "X-means"
seo_title: "X-means | Mengestimasi jumlah klaster secara otomatis"
weight: 3
pre: "2.5.3 "
title_suffix: "Mengestimasi jumlah klaster secara otomatis"
searchtitle: "Clustering X-means dengan Python"
---

{{< katex />}}
{{% youtube "2hkyJcWctUA" %}}

{{% summary %}}
- X-means memperluas k-means dengan memilih centroid awal yang saling berjauhan dan hanya membagi klaster ketika kriteria model mendukungnya.
- Setiap percobaan pembagian membandingkan nilai BIC; pembagian hanya diterima jika nilainya meningkat.
- Dengan bantuan `KMeans` dari scikit-learn dan fungsi BIC sederhana kita dapat menulis versi ringan X-means.
- Metode ini berguna ketika kita tidak mengetahui \\(k\\) yang tepat: jumlah klaster bertambah hanya saat data memberikan alasan kuat.
{{% /summary %}}

## Intuisi
Menentukan \\(k\\) merupakan kelemahan utama k-means. Jika terlalu kecil, klaster bercampur; jika terlalu besar, klaster terpecah. X-means memulai dari \\(k\\) kecil, menjalankan k-means, lalu mencoba membagi setiap klaster. Jika skor BIC meningkat, pembagian dipertahankan; jika tidak, klaster tetap apa adanya. Iterasi proses ini menghasilkan \\(k\\) yang seimbang antara akurasi dan kompleksitas.

## Pengingat matematis
Untuk klaster \\(\{C_1, \dots, C_k\}\\) dengan pusat \\(\{\mu_1, \dots, \mu_k\}\\), BIC didefinisikan sebagai

$$
\mathrm{BIC} = \ln L - \tfrac{p}{2} \ln n,
$$

di mana \\(L\\) adalah likelihood model, \\(p\\) jumlah parameter bebas, dan \\(n\\) jumlah sampel. Saat kita mencoba membagi klaster menjadi dua, BIC kedua model dibandingkan dan yang lebih besar (lebih baik) dipilih.

## Demonstrasi Python
Berikut implementasi ringkas X-means yang dibandingkan dengan k-means ber-\\(k=4\\).

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def buat_dataset(
    n_samples: int = 1200,
    n_pusat: int = 9,
    cluster_std: float = 1.1,
    random_state: int = 1707,
) -> NDArray[np.float64]:
    """Membuat data sintetis untuk eksperimen clustering."""
    fitur, _ = make_blobs(
        n_samples=n_samples,
        centers=n_pusat,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return fitur


def hitung_bic(
    data: NDArray[np.float64],
    label: NDArray[np.int64],
    pusat: NDArray[np.float64],
) -> float:
    """Menghitung skor BIC untuk klaster gaya k-means."""
    n_sampel, n_fitur = data.shape
    n_clust = pusat.shape[0]
    if n_clust <= 0:
        raise ValueError("Jumlah klaster harus positif.")

    ukuran = np.bincount(label, minlength=n_clust)
    sse = 0.0
    for idx, center in enumerate(pusat):
        if ukuran[idx] == 0:
            continue
        diff = data[label == idx] - center
        sse += float((diff**2).sum())

    varian = sse / max(n_sampel - n_clust, 1)
    if varian <= 0:
        varian = 1e-6

    log_likelihood = 0.0
    konstanta = -0.5 * n_fitur * np.log(2 * np.pi * varian)
    for size in ukuran:
        if size > 0:
            log_likelihood += size * konstanta - 0.5 * (size - 1)

    n_param = n_clust * (n_fitur + 1)
    bic = log_likelihood - 0.5 * n_param * np.log(n_sampel)
    return float(bic)


def xmeans_split(
    data: NDArray[np.float64],
    k_max: int = 20,
    random_state: int = 42,
) -> NDArray[np.int64]:
    """Mendekati X-means dengan membagi klaster ketika BIC meningkat."""
    rng = np.random.default_rng(random_state)
    antrean = [np.arange(len(data))]
    hasil: list[NDArray[np.int64]] = []

    while antrean:
        indeks = antrean.pop()
        subset = data[indeks]

        if len(indeks) <= 2:
            hasil.append(indeks)
            continue

        label_induk = np.zeros(len(indeks), dtype=int)
        pusat_induk = subset.mean(axis=0, keepdims=True)
        bic_induk = hitung_bic(subset, label_induk, pusat_induk)

        model_split = KMeans(
            n_clusters=2,
            n_init=10,
            max_iter=200,
            random_state=rng.integers(0, 1_000_000),
        ).fit(subset)
        bic_anak = hitung_bic(
            subset,
            model_split.labels_,
            model_split.cluster_centers_,
        )

        if bic_anak > bic_induk and (len(hasil) + len(antrean) + 1) < k_max:
            antrean.append(indeks[model_split.labels_ == 0])
            antrean.append(indeks[model_split.labels_ == 1])
        else:
            hasil.append(indeks)

    label_final = np.empty(len(data), dtype=int)
    for idx_clust, idx_sampel in enumerate(hasil):
        label_final[idx_sampel] = idx_clust
    return label_final


def jalankan_demo_xmeans(
    random_state: int = 2024,
    k_awal: int = 4,
    k_max: int = 18,
) -> dict[str, object]:
    """Membandingkan k-means biasa dan versi X-means sederhana."""
    data = buat_dataset(random_state=random_state)

    label_kmeans = KMeans(
        n_clusters=k_awal,
        n_init=10,
        random_state=random_state,
    ).fit_predict(data)
    label_xmeans = xmeans_split(data, k_max=k_max, random_state=random_state + 99)

    klaster_unik = np.unique(label_xmeans)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(data[:, 0], data[:, 1], c=label_kmeans, cmap="tab20", s=10)
    axes[0].set_title(f"k-means (k={k_awal})")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(data[:, 0], data[:, 1], c=label_xmeans, cmap="tab20", s=10)
    axes[1].set_title(f"X-means (k={len(klaster_unik)})")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Perbandingan penugasan k-means vs X-means")
    fig.tight_layout()
    plt.show()

    return {
        "kmeans_clusters": int(k_awal),
        "xmeans_clusters": int(len(klaster_unik)),
        "cluster_sizes": np.bincount(label_xmeans).tolist(),
    }


hasil = jalankan_demo_xmeans()
print(f"Jumlah klaster yang diminta k-means: {hasil['kmeans_clusters']}")
print(f"Jumlah klaster yang ditemukan X-means: {hasil['xmeans_clusters']}")
print(f"Ukuran setiap klaster X-means: {hasil['cluster_sizes']}")
```


![Perbandingan k-means vs X-means](/images/basic/clustering/x-means_block01_id.png)

## Referensi
{{% references %}}
<li>Pelleg, D., &amp; Moore, A. W. (2000). X-means: Extending k-means with Efficient Estimation of the Number of Clusters. <i>ICML</i>.</li>
<li>Bahmani, B., Moseley, B., Vattani, A., Kumar, R., &amp; Vassilvitskii, S. (2012). Scalable k-means++. <i>VLDB</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
