---
title: "Klastering k-means | Membagi data dengan penyempurnaan centroid"
linkTitle: "k-means"
seo_title: "Klastering k-means | Membagi data dengan penyempurnaan centroid"
weight: 1
pre: "2.5.1 "
title_suffix: "Memperbarui centroid secara iteratif"
searchtitle: "Belajar k-means dengan Python"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means mengelompokkan titik yang saling berdekatan dengan memperbarui centroid (titik wakil) dan penugasan sampel secara bergantian hingga stabil.
- Fungsi objektifnya adalah jumlah kuadrat dalam klaster (WCSS), yaitu jarak kuadrat antara setiap sampel dan centroid klasternya.
- `KMeans` dari `scikit-learn` memudahkan kita memvisualisasikan konvergensi, mencoba berbagai inisialisasi, dan melihat perubahan penugasan.
- Pemilihan \\(k\\) lazimnya memadukan diagnosis seperti metode siku atau skor siluet dengan pertimbangan kebutuhan bisnis.
{{% /summary %}}

## Intuisi
Setelah jumlah klaster \\(k\\) ditentukan, k-means mengulang dua langkah sederhana:

1. Tugaskan setiap sampel ke centroid terdekat.
2. Hitung ulang tiap centroid sebagai rata-rata sampel yang ditugaskan kepadanya.

Konvergensi tercapai ketika kedua langkah tersebut tidak lagi mengubah penugasan secara berarti. Karena hasilnya peka terhadap nilai awal dan outlier, praktik umum adalah mencoba beberapa seed acak atau memakai inisialisasi k-means++.

## Bentuk matematis
Untuk data \\(\mathcal{X} = \{x_1, \dots, x_n\}\\) dan klaster \\(\{C_1, \dots, C_k\}\\), k-means meminimalkan

$$
\min_{C_1, \dots, C_k} \sum_{j=1}^k \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2,
$$

dengan \\(\mu_j = |C_j|^{-1} \sum_{x_i \in C_j} x_i\\) sebagai centroid klaster \\(j\\). Secara intuitif, kita ingin jarak kuadrat tiap titik ke “pusat massa” klasternya sekecil mungkin.

## Eksperimen Python
Di bawah ini kita ulangi contoh dalam bahasa Jepang, kali ini dengan penjelasan dalam Bahasa Indonesia.

### 1. Membuat data dan meninjau inisialisasi
```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def buat_dataset(
    n_samples: int = 1000,
    random_state: int = 117_117,
    cluster_std: float = 1.5,
    n_centers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Membuat dataset sintetis untuk demonstrasi k-means."""
    return make_blobs(
        n_samples=n_samples,
        random_state=random_state,
        cluster_std=cluster_std,
        centers=n_centers,
    )


def pilih_centroid_awal(
    data: np.ndarray,
    n_clusters: int,
    ambang: float = -8.0,
) -> np.ndarray:
    """Memilih centroid awal secara deterministik di bawah ambang tertentu."""
    kandidat = data[data[:, 1] < ambang]
    if len(kandidat) < n_clusters:
        raise ValueError("Jumlah kandidat tidak cukup untuk centroid yang diminta.")
    return kandidat[:n_clusters]


def plot_konfigurasi_awal(
    data: np.ndarray,
    centroid: np.ndarray,
    figsize: tuple[float, float] = (7.5, 7.5),
) -> None:
    """Menampilkan dataset dan menyorot centroid awal."""
    japanize_matplotlib.japanize()
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(data[:, 0], data[:, 1], c="#4b5563", marker="x", label="sampel")
    ax.scatter(
        centroid[:, 0],
        centroid[:, 1],
        c="#ef4444",
        marker="o",
        s=80,
        label="centroid awal",
    )
    ax.set_title("Data awal dan centroid inisialisasi")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


DATA_X, DATA_Y = buat_dataset()
CENTROID_AWAL = pilih_centroid_awal(DATA_X, n_clusters=4)
plot_konfigurasi_awal(DATA_X, CENTROID_AWAL)
```


![Konfigurasi awal](/images/basic/clustering/k-means1_block01_id.png)

### 2. Melihat konvergensi centroid
```python
from typing import Sequence

from sklearn.cluster import KMeans


def plot_konvergensi(
    data: np.ndarray,
    centroid_awal: np.ndarray,
    max_iters: Sequence[int] = (1, 2, 3, 10),
    random_state: int = 1,
) -> dict[int, float]:
    """Menjalankan k-means dengan berbagai batas iterasi dan menggambarkan hasilnya."""
    japanize_matplotlib.japanize()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    inersia_per_iterasi: dict[int, float] = {}

    for ax, max_iter in zip(axes.ravel(), max_iters, strict=False):
        model = KMeans(
            n_clusters=len(centroid_awal),
            init=centroid_awal,
            max_iter=max_iter,
            n_init=1,
            random_state=random_state,
        )
        model.fit(data)
        label = model.predict(data)

        ax.scatter(data[:, 0], data[:, 1], c=label, cmap="tab10", s=10)
        ax.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            c="#dc2626",
            marker="o",
            s=80,
            label="centroid",
        )
        ax.set_title(f"max_iter = {max_iter}")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        inersia_per_iterasi[max_iter] = float(model.inertia_)

    fig.suptitle("Perilaku konvergensi terhadap batas iterasi")
    fig.tight_layout()
    plt.show()
    return inersia_per_iterasi


STAT_KONVERGENSI = plot_konvergensi(DATA_X, CENTROID_AWAL)
for iterasi, inersia in STAT_KONVERGENSI.items():
    print(f"max_iter={iterasi}: inersia={inersia:,.1f}")
```


![Perbandingan konvergensi](/images/basic/clustering/k-means1_block02_id.png)

### 3. Menambah tumpang tindih klaster
```python
def plot_pengaruh_tumpang_tindih(
    random_state_dasar: int = 117_117,
    deviasi: Sequence[float] = (1.0, 2.0, 3.0, 4.5),
) -> None:
    """Menunjukkan bagaimana tumpang tindih memperumit penugasan k-means."""
    japanize_matplotlib.japanize()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for ax, std in zip(axes.ravel(), deviasi, strict=False):
        fitur, _ = make_blobs(
            n_samples=1_000,
            random_state=random_state_dasar,
            cluster_std=std,
        )
        penugasan = KMeans(n_clusters=2, random_state=random_state_dasar).fit_predict(
            fitur
        )
        ax.scatter(fitur[:, 0], fitur[:, 1], c=penugasan, cmap="tab10", s=10)
        ax.set_title(f"cluster_std = {std}")
        ax.grid(alpha=0.2)

    fig.suptitle("Dampak tumpang tindih terhadap penugasan")
    fig.tight_layout()
    plt.show()


plot_pengaruh_tumpang_tindih()
```


![Perbandingan tumpang tindih](/images/basic/clustering/k-means1_block03_id.png)

### 4. Membandingkan metrik pemilihan \\(k\\)
```python
from sklearn.metrics import silhouette_score


def evaluasi_jumlah_klaster(
    data: np.ndarray,
    rentang_k: Sequence[int] = range(2, 11),
) -> dict[str, list[float]]:
    """Menghitung WCSS dan skor siluet untuk berbagai nilai k."""
    inersia: list[float] = []
    siluet: list[float] = []

    for k in rentang_k:
        model = KMeans(n_clusters=k, random_state=117_117).fit(data)
        inersia.append(float(model.inertia_))
        siluet.append(float(silhouette_score(data, model.labels_)))

    return {"inertia": inersia, "silhouette": siluet}


def plot_metrik_k(
    metrik: dict[str, list[float]],
    rentang_k: Sequence[int],
) -> None:
    """Menggambarkan metode siku dan skor siluet secara berdampingan."""
    japanize_matplotlib.japanize()
    ks = list(rentang_k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, metrik["inertia"], marker="o")
    axes[0].set_title("Metode siku (WCSS)")
    axes[0].set_xlabel("Jumlah klaster k")
    axes[0].set_ylabel("WCSS")
    axes[0].grid(alpha=0.2)

    axes[1].plot(ks, metrik["silhouette"], marker="o", color="#ea580c")
    axes[1].set_title("Skor siluet")
    axes[1].set_xlabel("Jumlah klaster k")
    axes[1].set_ylabel("Skor")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    plt.show()


METRIK_K = evaluasi_jumlah_klaster(DATA_X, range(2, 11))
plot_metrik_k(METRIK_K, range(2, 11))

nilai_k_optimal = int(
    range(2, 11)[
        max(
            range(len(METRIK_K["silhouette"])),
            key=METRIK_K["silhouette"].__getitem__,
        )
    ]
)
print(f"Skor siluet tertinggi dicapai pada k={nilai_k_optimal}")
```


![Metrik pemilihan k](/images/basic/clustering/k-means1_block04_id.png)

## Referensi
{{% references %}}
<li>MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. <i>Proceedings of the Fifth Berkeley Symposium</i>.</li>
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
