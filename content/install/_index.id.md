---
title: "Panduan Penyiapan Lingkungan | Python dan Peralatan"
linkTitle: "Penyiapan"
seo_title: "Panduan Penyiapan Lingkungan | Python dan Peralatan"
weight: 1
pre: "<b>1. </b>"
chapter: true
not_use_colab: true
not_use_twitter: true
---
# Bagian 1
## Pendahuluan

Situs web ini adalah halaman tempat saya meninggalkan studi saya dalam bentuk vlog dan kode.
Sejak dirilis pada awal Oktober 2021, berfokus pada topik yang terkait dengan pemrosesan dan pembelajaran mesin yang diperlukan untuk analisis data.

- Pembelajaran mesin: bagaimana algoritme bekerja dan sistem yang sebenarnya telah saya jalankan.
- Pra-pemrosesan data: tentang pemrosesan data/pra-pemrosesan berbagai jenis data
- Metrik evaluasi: tentang pemilihan model dan evaluasi kinerja model
- Deret waktu: tentang data yang menggabungkan cap waktu dan nilai numerik

Sebagian besar kode telah diuji dalam bahasa Jepang. Harap diperhatikan, bahwa sebagian diagram ditampilkan dalam bahasa Jepang.

## Cara Menjalankan File Python

{{< tabs >}}
{{% tab name="Google Colaboratory(Colab)" %}}
Google Colaboratory (Colab) adalah nama dari lingkungan runtime Python berbasis browser Google dan
Ini adalah layanan yang memungkinkan Anda untuk menjalankan dan berbagi kode dengan mudah tanpa membangun lingkungan. Akun Google diperlukan untuk menggunakan layanan ini.
Klik tombol <a href="#" class="btn colab-btn-border in-text-button">Colab</a> di pojok kanan atas setiap halaman untuk menjalankan kode untuk halaman tersebut di Google Colaboratory.

![](2022-03-30-20-15-24.png)

Lihat juga halaman-halaman seperti 「[Colaboratory FAQs](https://research.google.com/colaboratory/faq.html?hl=id)」tentang Colab.

{{% /tab %}}
{{% tab name="Anaconda" %}}

Setelah membuat lingkungan virtual dengan Anaconda, instal paket pustaka yang diperlukan untuk setiap kode.
Namun demikian, harap dicatat, bahwa Anaconda tidak digunakan sama sekali di situs ini dan belum diuji.

```
conda create -n py38_env python=3.8
conda install nama-perpustakaan
```

{{% /tab %}}
{{% tab name="Poetry" %}}

Setelah menginstal puisi seperti yang dijelaskan di [Poetry | Installation](https://python-poetry.org/docs/), jalankan `poetry install` pada hirarki di mana file `pyproject.toml` berada.

{{% /tab %}}
{{< /tabs >}}