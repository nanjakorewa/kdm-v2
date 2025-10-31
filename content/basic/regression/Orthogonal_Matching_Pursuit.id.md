---
title: "Orthogonal Matching Pursuit (OMP)"
pre: "2.1.12 "
weight: 12
title_suffix: "Pemilihan rakus untuk koefisien jarang"
---

{{% summary %}}
- Orthogonal Matching Pursuit (OMP) memilih fitur yang paling berkorelasi dengan residual pada setiap langkah untuk membangun model linear yang jarang.
- Algoritme menyelesaikan regresi kuadrat terkecil dengan fitur yang telah dipilih sehingga koefisien mudah ditafsirkan.
- Alih-alih mengatur kekuatan regularisasi, tingkat sparsitas dikendalikan langsung dengan menentukan jumlah fitur yang dipertahankan.
- Menstandarkan fitur dan memeriksa multikolinearitas membantu OMP tetap stabil ketika merekonstruksi sinyal jarang.
{{% /summary %}}

## Intuisi
Ketika hanya sedikit fitur yang benar-benar penting, OMP menambahkan fitur yang paling banyak mengurangi error residual di tiap iterasi. Metode ini menjadi dasar pembelajaran kamus dan sparse coding, serta menghasilkan vektor koefisien ringkas yang menyoroti prediktor paling relevan.

## Formulasi matematis
Inisialisasi residual dengan \(\mathbf{r}^{(0)} = \mathbf{y}\). Untuk setiap iterasi \(t\):

1. Hitung hasil kali dalam antara setiap fitur \(\mathbf{x}_j\) dan residual \(\mathbf{r}^{(t-1)}\), kemudian pilih fitur \(j\) dengan nilai absolut terbesar.
2. Tambahkan \(j\) ke himpunan aktif \(\mathcal{A}_t\).
3. Selesaikan masalah kuadrat terkecil terbatas pada \(\mathcal{A}_t\) untuk memperoleh \(\hat{\boldsymbol\beta}_{\mathcal{A}_t}\).
4. Perbarui residual \(\mathbf{r}^{(t)} = \mathbf{y} - \mathbf{X}_{\mathcal{A}_t} \hat{\boldsymbol\beta}_{\mathcal{A}_t}\).

Berhenti ketika sparsitas yang diinginkan tercapai atau residual sudah cukup kecil.

## Eksperimen dengan Python
Cuplikan berikut membandingkan OMP dan lasso pada data dengan vektor koefisien sebenarnya yang jarang.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error


def run_omp_vs_lasso(
    n_samples: int = 200,
    n_features: int = 40,
    sparsity: int = 4,
    noise_scale: float = 0.5,
    xlabel: str = "feature index",
    ylabel: str = "coefficient",
    label_true: str = "true",
    label_omp: str = "OMP",
    label_lasso: str = "Lasso",
    title: str | None = None,
) -> dict[str, object]:
    """Compare OMP and lasso on synthetic sparse regression data.

    Args:
        n_samples: Number of training samples to generate.
        n_features: Total number of features in the dictionary.
        sparsity: Count of non-zero coefficients in the ground truth.
        noise_scale: Standard deviation of Gaussian noise added to targets.
        xlabel: Label for the coefficient plot x-axis.
        ylabel: Label for the coefficient plot y-axis.
        label_true: Legend label for the ground-truth bars.
        label_omp: Legend label for the OMP bars.
        label_lasso: Legend label for the lasso bars.
        title: Optional title for the bar chart.

    Returns:
        Dictionary containing recovered supports and MSE values.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n_samples, n_features))
    true_coef = np.zeros(n_features)
    true_support = rng.choice(n_features, size=sparsity, replace=False)
    true_coef[true_support] = rng.normal(loc=0.0, scale=3.0, size=sparsity)
    y = X @ true_coef + rng.normal(scale=noise_scale, size=n_samples)

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
    omp.fit(X, y)
    lasso = Lasso(alpha=0.05)
    lasso.fit(X, y)

    omp_pred = omp.predict(X)
    lasso_pred = lasso.predict(X)

    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(n_features)
    ax.bar(indices - 0.3, true_coef, width=0.2, label=label_true, color="#2ca02c")
    ax.bar(indices, omp.coef_, width=0.2, label=label_omp, color="#1f77b4")
    ax.bar(indices + 0.3, lasso.coef_, width=0.2, label=label_lasso, color="#d62728")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "true_support": np.flatnonzero(true_coef),
        "omp_support": np.flatnonzero(omp.coef_),
        "lasso_support": np.flatnonzero(np.abs(lasso.coef_) > 1e-6),
        "omp_mse": float(mean_squared_error(y, omp_pred)),
        "lasso_mse": float(mean_squared_error(y, lasso_pred)),
    }



metrics = run_omp_vs_lasso(
    xlabel="indeks fitur",
    ylabel="koefisien",
    label_true="sebenarnya",
    label_omp="OMP",
    label_lasso="Lasso",
    title="Perbandingan OMP dan Lasso",
)
print("Dukungan sebenarnya:", metrics['true_support'])
print("Dukungan OMP:", metrics['omp_support'])
print("Dukungan Lasso:", metrics['lasso_support'])
print(f"MSE OMP: {metrics['omp_mse']:.4f}")
print(f"MSE Lasso: {metrics['lasso_mse']:.4f}")

```

### Cara membaca hasil
- Jika `n_nonzero_coefs` sama dengan jumlah koefisien tak nol yang sebenarnya, OMP cenderung berhasil menemukan fitur relevan.
- Dibandingkan lasso, OMP memberikan koefisien benar-benar nol pada fitur yang tidak dipilih.
- Ketika fitur saling berkorelasi kuat, urutan pemilihan dapat berubah-ubah, sehingga praproses dan rekayasa fitur tetap penting.

## Referensi
{{% references %}}
<li>Pati, Y. C., Rezaiifar, R., &amp; Krishnaprasad, P. S. (1993). Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition. Dalam <i>Conference Record of the Twenty-Seventh Asilomar Conference on Signals, Systems and Computers</i>.</li>
<li>Tropp, J. A. (2004). Greed is Good: Algorithmic Results for Sparse Approximation. <i>IEEE Transactions on Information Theory</i>, 50(10), 2231–2242.</li>
{{% /references %}}
