---
title: "Support Vector Regression (SVR)"
pre: "2.1.10 "
weight: 10
title_suffix: "Prediksi tangguh dengan tabung ε-insensitive"
---

{{% summary %}}
- SVR memperluas SVM ke regresi dengan memperlakukan kesalahan di dalam tabung ε-insensitive sebagai nol sehingga dampak outlier berkurang.
- Metode kernel memungkinkan hubungan nonlinier yang fleksibel sambil menjaga model tetap ringkas melalui support vector.
- Hiperparameter `C`, `epsilon`, dan `gamma` mengatur keseimbangan antara kemampuan generalisasi dan kehalusan model.
- Penyetaraan skala fitur sangat penting; membungkus praproses dan pembelajaran dalam pipeline memastikan transformasi yang konsisten.
{{% /summary %}}

## Intuisi
SVR menyesuaikan sebuah fungsi yang dikelilingi tabung selebar ε: titik yang berada di dalam tabung tidak dikenai penalti, sedangkan titik di luar tabung dibebani penalti. Hanya titik yang menyentuh atau keluar dari tabung —support vector— yang memengaruhi model akhir, sehingga menghasilkan aproksimasi halus yang tahan terhadap noise.

## Formulasi matematis
Masalah optimisasinya adalah

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
$$

dengan kendala

$$
\begin{aligned}
y_i - (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) &\le \epsilon + \xi_i, \\
(\mathbf{w}^\top \phi(\mathbf{x}_i) + b) - y_i &\le \epsilon + \xi_i^*, \\
\xi_i, \xi_i^* &\ge 0,
\end{aligned}
$$

di mana \(\phi\) memetakan input ke ruang fitur melalui kernel yang dipilih. Menyelesaikan bentuk dual menghasilkan support vector dan koefisiennya.

## Eksperimen dengan Python
Contoh berikut menunjukkan SVR yang dikombinasikan dengan `StandardScaler` dalam sebuah pipeline.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
)

svr.fit(X_train, y_train)
pred = svr.predict(X_test)
```

### Cara membaca hasil
- Pipeline menstandarkan data latih menggunakan mean dan variansnya, lalu menerapkan transformasi yang sama ke data uji.
- `pred` berisi prediksi; pengaturan `epsilon` dan `C` mengubah trade-off antara overfitting dan underfitting.
- Nilai `gamma` yang lebih besar pada kernel RBF menekankan pola lokal, sedangkan nilai kecil menghasilkan fungsi yang lebih halus.

## Referensi
{{% references %}}
<li>Smola, A. J., &amp; Schölkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199–222.</li>
<li>Vapnik, V. (1995). <i>The Nature of Statistical Learning Theory</i>. Springer.</li>
{{% /references %}}
