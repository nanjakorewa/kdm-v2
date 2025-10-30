---
title: Regresi linear
weight: 1
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.1 </b>"
---

{{% summary %}}
- Regresi linear menangkap hubungan linier antara masukan dan keluaran serta menjadi fondasi untuk prediksi dan interpretasi.
- Dengan menambahkan regularisasi, metode robust, atau reduksi dimensi, model ini dapat menyesuaikan diri dengan berbagai jenis data.
- Setiap halaman mengikuti alur yang sama—ringkasan, intuisi, formulasi, eksperimen Python, dan referensi—agar pemahaman berkembang dari dasar hingga aplikasi.
{{% /summary %}}

# Regresi linear

## Intuisi
Regresi linear menjawab pertanyaan “berapa banyak keluaran berubah ketika masukan bertambah satu satuan?” Koefisiennya mudah ditafsirkan dan perhitungannya cepat, sehingga sering menjadi model pertama yang dicoba pada proyek pembelajaran mesin.

## Formulasi matematis
Metode kuadrat terkecil ordinari menghitung koefisien dengan meminimalkan jumlah kuadrat kesalahan antara observasi dan prediksi. Untuk matriks \(\mathbf{X}\) dan vektor \(\mathbf{y}\),

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
$$

Halaman-halaman berikut memperluas kerangka ini dengan regularisasi, teknik robust, dan pendekatan lainnya.

## Eksperimen dengan Python
Setiap halaman menyediakan contoh `scikit-learn` yang dapat dijalankan dan mencakup:

- Dasar-dasar: kuadrat terkecil ordinari, Ridge, Lasso, regresi robust  
- Lebih ekspresif: regresi polinomial, Elastic Net, regresi kuantil, regresi linear Bayesian  
- Reduksi dimensi dan kejarangan: principal component regression, PLS, weighted least squares, Orthogonal Matching Pursuit, SVR, dan lainnya

Cobalah kodenya secara langsung untuk memahami perilaku model pada data sintetis maupun nyata.

## Referensi
{{% references %}}
<li>Draper, N. R., &amp; Smith, H. (1998). <i>Applied Regression Analysis</i> (3rd ed.). John Wiley &amp; Sons.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
