title: "Halaman Utama"
---

## Selamat datang di K_DM Book
K_DM Book adalah panduan praktis yang membantu Anda mempelajari analisis data dan machine learning hingga tahap produksi. Setiap bab menggabungkan penjelasan konseptual dengan notebook serta script yang bisa dijalankan, sehingga Anda dapat belajar dan langsung mencoba.

- **Alur belajar terstruktur** — Materi diurutkan agar Anda dapat naik tingkat dari dasar ke penerapan tanpa kehilangan konteks.
- **Contoh dapat dijalankan** — Hampir semua artikel dilengkapi notebook atau script Python untuk mereplikasi grafik dan eksperimen.
- **Tips bernuansa industri** — Kami merangkum metrik evaluasi, checklist pra-pemrosesan, dan pola visualisasi yang berguna dalam proyek nyata.

## Rekomendasi jalur belajar
1. **Dasar** — Pelajari ulang model inti seperti regresi, klasifikasi, clustering, dan reduksi dimensi.
2. **Prep / Evaluation** — Perdalam teknik rekayasa fitur dan validasi model agar iterasi menjadi lebih yakin.
3. **Timeseries / Finance / WebApp** — Telusuri studi kasus per domain dan ketahui cara merilis solusi.
4. **Visualize** — Kuasai cara menyampaikan temuan melalui grafik, dashboard, dan laporan yang efektif.

## Peta konten
{{< mermaid >}}
flowchart TD
  subgraph fundamentals["Fase Dasar"]
    A1[Dasar Matematika<br>Aljabar Linear / Probabilitas]
    A2[Penyiapan Lingkungan<br>Python & Library]
    A3[Analisis Eksploratif<br>EDA & Visualisasi Dasar]
  end

  subgraph modeling["Fase Pemodelan Dasar"]
    B1[Regresi]
    B2[Klasifikasi]
    B3[Clustering]
    B4[Reduksi Dimensi]
    B5[Metode Ensemble]
  end

  subgraph evaluation["Fase Evaluasi & Tuning"]
    C1[Metrik]
    C2[Penyetelan Hiperparameter]
    C3[Rekayasa Fitur]
    C4[Interpretasi Model]
  end

  subgraph communication["Fase Komunikasi"]
    D1[Katalog Visualisasi]
    D2[Cerita & Laporan]
    D3[Tips Kolaborasi]
  end

  subgraph deployment["Fase Aplikasi & Deploy"]
    E1[Analisis Deret Waktu]
    E2[Kasus Keuangan]
    E3[Implementasi Web<br>Flask / Gradio]
    E4[Monitoring & Operasional]
  end

  fundamentals --> modeling
  modeling --> evaluation
  evaluation --> communication
  evaluation --> deployment
  communication --> deployment

  click A2 "/install/" "Buka panduan penyiapan lingkungan"
  click A3 "/prep/" "Pelajari bab persiapan data"
  click B1 "/basic/regression/" "Masuk ke bagian regresi"
  click B2 "/basic/classification/" "Masuk ke bagian klasifikasi"
  click B3 "/basic/clustering/" "Masuk ke bagian clustering"
  click B4 "/basic/dimensionality_reduction/" "Masuk ke bagian reduksi dimensi"
  click B5 "/basic/ensemble/" "Masuk ke bagian metode ensemble"
  click C1 "/eval/" "Pelajari metrik evaluasi"
  click C3 "/prep/feature_selection/" "Masuk ke rekayasa fitur"
  click D1 "/visualize/" "Jelajahi katalog visualisasi"
  click E1 "/timeseries/" "Lihat materi deret waktu"
  click E2 "/finance/" "Lihat materi keuangan"
  click E3 "/webapp/" "Lihat materi aplikasi web"
{{< /mermaid >}}

## Sumber daya dan dukungan
- **Notebook & script** — Tersedia di `scripts/` dan `data/` untuk mereplikasi tutorial.
- **Informasi terbaru** — Ikuti linimasa di halaman utama atau berlangganan RSS (`/index.xml`).
- **Masukan** — Jika menemukan kesalahan, sampaikan melalui [formulir masukan](https://kdm.hatenablog.jp/entry/issue) atau hubungi kami lewat X / Twitter.

## Komunitas & kanal resmi
- <a href="https://www.youtube.com/@K_DM" style="color:#FF0000;"><i class="fab fa-fw fa-youtube"></i> YouTube — video penjelasan dan sesi langsung</a>
- <a href="https://twitter.com/_K_DM" style="color:#1DA1F2;"><i class="fab fa-fw fa-twitter"></i> X (Twitter) — kabar singkat dan tips cepat</a>

Untuk informasi lebih lanjut, silakan tinjau [kebijakan privasi](https://kdm.hatenablog.jp/privacy-policy). Manfaatkan materi ini untuk proyek Anda sendiri!
