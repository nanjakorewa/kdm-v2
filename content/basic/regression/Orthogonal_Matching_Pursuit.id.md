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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
n_samples, n_features = 200, 40
X = rng.normal(size=(n_samples, n_features))
true_coef = np.zeros(n_features)
true_support = [1, 5, 12, 33]
true_coef[true_support] = [2.5, -1.5, 3.0, 1.0]
y = X @ true_coef + rng.normal(scale=0.5, size=n_samples)

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=4).fit(X, y)
lasso = Lasso(alpha=0.05).fit(X, y)

print("Indeks tak nol OMP:", np.flatnonzero(omp.coef_))
print("Indeks tak nol Lasso:", np.flatnonzero(np.abs(lasso.coef_) > 1e-6))

print("MSE OMP:", mean_squared_error(y, omp.predict(X)))
print("MSE Lasso:", mean_squared_error(y, lasso.predict(X)))

coef_df = pd.DataFrame({
    "true": true_coef,
    "omp": omp.coef_,
    "lasso": lasso.coef_,
})
print(coef_df.head(10))
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
