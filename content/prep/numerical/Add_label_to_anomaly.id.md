---
title: "Memberi Label pada Nilai Pencilan①"
pre: "3.3.4 "
weight: 4
title_replace: "Melakukan Metode Hotelling's T² dengan Python"
---

<div class="pagetop-box">
    <p>Metode Hotelling $T^2$ mengasumsikan bahwa data mengikuti distribusi normal, dan memanfaatkan fakta bahwa tingkat keanehan setiap data mengikuti distribusi $\chi^2$ untuk mendeteksi nilai pencilan. Halaman ini akan menunjukkan cara memberi label pada nilai pencilan yang terdapat dalam data yang mengikuti distribusi normal dengan menggunakan Python.</p>
</div>

## Data untuk Eksperimen
Data akan dibuat dari distribusi normal menggunakan `stats.norm.rvs(size=n_samples)`, dan beberapa nilai pencilan akan ditambahkan ke dalamnya.


```pythonfrom scipy import stats
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

# Menetapkan seed untuk menghasilkan angka acak yang konsisten
np.random.seed(seed=100)

# Data untuk eksperimen
n_samples = 1000
x = stats.norm.rvs(size=n_samples)
ind = [i for i in range(n_samples)]

# Indeks nilai pencilan
anom_ind = [
    100,
    200,
    300,
    400,
    500,
]
for an_i in anom_ind:
    x[an_i] += np.random.randint(10, 20)

# Memvisualisasikan data
plt.figure(figsize=(12, 4))
plt.plot(ind, x, ".", label="Data Normal")
plt.plot(anom_ind, x[anom_ind], "x", label="Nilai Pencilan", c="red", markersize=12)
plt.legend()
plt.show()
```


    
![png](/images/prep/numerical/Add_label_to_anomaly_files/Add_label_to_anomaly_1_0.png)
    

## Mendeteksi Nilai Pencilan
Kita menghitung nilai rata-rata dan varians dari sampel, lalu menentukan skor tingkat keanehan untuk setiap sampel. Data dengan tingkat keanehan yang melebihi ambang batas yang telah ditentukan sebelumnya akan terdeteksi sebagai nilai pencilan.

```python
from __future__ import annotations


def get_anomaly_index(X: np.array, threshold: float) -> list[int]:
    """Mendapatkan Indeks Nilai Pencilan untuk Data Satu Dimensi

    Fungsi ini digunakan untuk mendapatkan indeks nilai pencilan dari data satu dimensi.

    Args:
        X (numpy.array): Data target yang akan dianalisis.
        threshold (float): Ambang batas untuk menentukan nilai pencilan.

    Returns:
        list[int]: Daftar indeks dari data yang dianggap sebagai nilai pencilan.

    Examples:
        >>> print(get_anomaly_index(np.array([1, 2, 3, ..., 1]), 0.05))
        [1, ]
    """
    avg = np.average(X)
    var = np.var(X)
    scores = [(x_i - avg) ** 2 / var for x_i in X]
    th = stats.chi2.interval(1 - threshold, 1)[1]
    return [ind for (ind, x) in enumerate(scores) if x > th]

```

## Memastikan Hasil Deteksi
Kita akan memverifikasi apakah nilai pencilan berhasil terdeteksi dengan membandingkan hasil deteksi terhadap data asli yang mengandung nilai pencilan.


```pythondetected_anom_index = get_anomaly_index(x, 0.05)

# Memvisualisasikan hasil deteksi
plt.figure(figsize=(12, 4))
plt.plot(ind, x, ".", label="Data Normal")
plt.plot(anom_ind, x[anom_ind], "x", label="Nilai Pencilan", c="red", markersize=12)
plt.plot(
    detected_anom_index,
    x[detected_anom_index],
    "o",
    label="Nilai Pencilan yang Terdeteksi",
    c="red",
    alpha=0.2,
    markersize=12,
)
plt.legend()
plt.show()
```

    
![png](/images/prep/numerical/Add_label_to_anomaly_files/Add_label_to_anomaly_5_0.png)
    

