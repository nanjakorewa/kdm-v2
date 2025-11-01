---
title: "Gaussian Mixture Model (GMM)"
pre: "2.5.5 "
weight: 5
title_suffix: "Clustering probabilistik dengan penugasan lunak"
searchtitle: "Clustering dengan Gaussian Mixture Model di Python"
---

{{% summary %}}
- GMM memodelkan data sebagai gabungan berbobot dari distribusi normal multivariat.
- Matriks tanggung jawab (responsibility) menunjukkan seberapa kuat setiap komponen menjelaskan tiap sampel.
- Parameter diperkirakan menggunakan algoritma EM; bentuk kovarians dapat dipilih dari `full`, `tied`, `diag`, atau `spherical`.
- Pemilihan model biasanya menggabungkan BIC/AIC dengan beberapa inisialisasi acak agar hasil stabil.
{{% /summary %}}

## Intuisi
Bayangkan data berasal dari \\(K\\) sumber Gaussian. Setiap komponen memiliki vektor rata-rata dan matriks kovarians sehingga klasternya berbentuk elips. Tidak seperti k-means yang memberi label keras, GMM memberikan penugasan lunak: untuk setiap sampel \\(x_i\\) dan komponen \\(k\\) kita memperoleh \\(\gamma_{ik}\\), probabilitas bahwa komponen tersebut menghasilkan \\(x_i\\).

## Rumusan
Kerapatan \\(\mathbf{x}\\) dinyatakan sebagai

$$
p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k),
$$

dengan bobot campuran \\(\pi_k\\) (tidak negatif dan berjumlah 1). Algoritma EM melakukan:

- **E-step**: menghitung tanggung jawab \\(\gamma_{ik}\\).
  $$
  \gamma_{ik} = \frac{\pi_k \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}
  {\sum_{j=1}^K \pi_j \, \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.
  $$
- **M-step**: memperbarui \\(\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\\) menggunakan \\(\gamma_{ik}\\) sebagai bobot.

Log-likelihood meningkat monoton hingga mencapai optimum lokal.

## Contoh Python
Kita akan menyesuaikan GMM pada data sintetis 2D, menggambar penugasan keras, serta melaporkan bobot campuran dan ukuran matriks tanggung jawab.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


def jalankan_demo_gmm(
    n_samples: int = 600,
    n_komponen: int = 3,
    cluster_std: list[float] | tuple[float, ...] = (1.0, 1.4, 0.8),
    covariance_type: str = "full",
    random_state: int = 7,
    n_init: int = 8,
) -> dict[str, object]:
    """Melatih GMM dan memvisualisasikan pusat komponen serta label prediksi."""
    fitur, _ = make_blobs(
        n_samples=n_samples,
        centers=n_komponen,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    gmm = GaussianMixture(
        n_components=n_komponen,
        covariance_type=covariance_type,
        random_state=random_state,
        n_init=n_init,
    )
    gmm.fit(fitur)

    label_keras = gmm.predict(fitur)
    tanggung_jawab = gmm.predict_proba(fitur)
    log_likelihood = float(gmm.score(fitur))
    bobot = gmm.weights_

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sebar = ax.scatter(
        fitur[:, 0],
        fitur[:, 1],
        c=label_keras,
        cmap="viridis",
        s=30,
        edgecolor="white",
        linewidth=0.2,
        alpha=0.85,
    )
    ax.scatter(
        gmm.means_[:, 0],
        gmm.means_[:, 1],
        marker="x",
        c="red",
        s=140,
        linewidth=2.0,
        label="Pusat komponen",
    )
    ax.set_title("Clustering dengan Gaussian Mixture Model")
    ax.set_xlabel("fitur 1")
    ax.set_ylabel("fitur 2")
    ax.grid(alpha=0.2)
    handles, _ = sebar.legend_elements()
    label_legenda = [f"cluster {idx}" for idx in range(n_komponen)]
    ax.legend(handles, label_legenda, title="Label prediksi", loc="upper right")
    fig.tight_layout()
    plt.show()

    return {
        "log_likelihood": log_likelihood,
        "weights": bobot.tolist(),
        "responsibilities_shape": tanggung_jawab.shape,
    }


hasil = jalankan_demo_gmm()
print(f"log-likelihood: {hasil['log_likelihood']:.3f}")
print("bobot campuran:", hasil["weights"])
print("ukuran matriks tanggung jawab:", hasil["responsibilities_shape"])
```


![Hasil clustering GMM](/images/basic/clustering/gaussian-mixture_block01_id.png)

## Referensi
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Dempster, A. P., Laird, N. M., &amp; Rubin, D. B. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm. <i>Journal of the Royal Statistical Society, Series B</i>.</li>
<li>scikit-learn developers. (2024). <i>Gaussian Mixture Models</i>. https://scikit-learn.org/stable/modules/mixture.html</li>
{{% /references %}}
