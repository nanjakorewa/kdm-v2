---
title: "Klasifikasi linear | Dasar pembelajaran mesin"
linkTitle: "Klasifikasi linear"
seo_title: "Klasifikasi linear | Dasar pembelajaran mesin"
weight: 3
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.2 </b>"
---

{{% summary %}}
- Klasifikator linear memisahkan kelas dengan hiperbidang sehingga mudah diinterpretasi dan dilatih dengan cepat.
- Perceptron, regresi logistik, SVM, dan metode terkait berbagi banyak ide sehingga menarik untuk dibandingkan.
- Konsep di sini terhubung dengan regularisasi, kernel, dan pembelajaran berbasis jarak.
{{% /summary %}}

# Klasifikasi linear

## Intuisi
Klasifikator linear menempatkan hiperbidang di ruang fitur dan memberi label berdasarkan sisi mana sampel berada. Mulai dari perceptron sederhana, regresi logistik probabilistik, hingga SVM berbasis margin, perbedaannya terletak pada cara memilih hiperbidang.

## Formulasi matematis
Klasifikator linear umumnya memprediksi \\(\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)\\). Setiap metode menentukan \\(\mathbf{w}\\) dengan pendekatan yang berbeda; versi probabilistik memakai sigmoid atau softmax. Kernel trick cukup mengganti hasil kali dalam dengan kernel untuk memperoleh batas nonlinier.

## Experimen dengan Python
Bab ini menyajikan contoh praktis dengan Python dan scikit-learn:

- Perceptron: aturan pembaruan dan batas keputusan klasifikator linear paling sederhana.
- Regresi logistik: klasifikasi biner probabilistik dengan entropi silang.
- Regresi softmax: perluasan multikelas yang menghasilkan seluruh probabilitas.
- Linear discriminant analysis (LDA): arah proyeksi yang memaksimalkan separasi kelas.
- Support vector machine (SVM): margin maksimum dan kernel trick.
- Naive Bayes: klasifikasi cepat dengan asumsi independensi kondisional.
- k tetangga terdekat: pembelajaran malas berbasis jarak.

Silakan jalankan cuplikan kode dan bereksperimen untuk melihat bagaimana batas keputusan dan probabilitas berubah.

## Referensi
{{% references %}}
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
{{% /references %}}
