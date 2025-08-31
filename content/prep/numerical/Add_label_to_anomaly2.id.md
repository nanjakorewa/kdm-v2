---
title: "Memberi Label pada Nilai Pencilan②"
pre: "3.3.5 "
weight: 5
title_replace: "Mendeteksi Nilai Pencilan dengan IsolationForest"
---

<div class="pagetop-box">
    <p>Isolation Forest adalah salah satu algoritma untuk deteksi anomali. Algoritma ini menetapkan struktur pohon pada data, lalu menghitung skor berdasarkan bagaimana pohon sesuai dengan data tersebut. Nilai pencilan cenderung berada di lokasi yang berbeda dari mayoritas data lainnya, sehingga sering kali menyebabkan pembagian pohon pada tahap awal atau memiliki sedikit batas keputusan di sekitarnya. Pada halaman ini, kita akan menggunakan Isolation Forest untuk memberi label pada nilai pencilan yang terdapat dalam data.</p>
</div>


{{% notice document %}}
[sklearn.ensemble.IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
{{% /notice %}}

## Data untuk Eksperimen


```python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

# Menetapkan seed untuk menghasilkan angka acak yang konsisten
np.random.seed(seed=100)
X, y = make_moons(n_samples=1000, noise=0.1)

# Indeks nilai pencilan
anom_ind = [i * 50 for i in range(18)]
for an_i in anom_ind:
    X[an_i] *= 2.5

# Memvisualisasikan data
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], marker=".", label="Data Normal")
plt.scatter(X[:, 0][anom_ind], X[:, 1][anom_ind], marker="x", s=70, label="Nilai Pencilan")
plt.legend()
plt.show()

```


    
![png](/images/prep/numerical/Add_label_to_anomaly2_files/Add_label_to_anomaly2_1_0.png)
    


## Mendeteksi Nilai Pencilan

> contamination‘auto’ or float, default=’auto’
> 
> The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Used when fitting to define the threshold on the scores of the samples. 

Parameter `contamination` digunakan untuk menentukan proporsi data yang akan dianggap sebagai nilai pencilan. Dengan menentukan nilai parameter ini, kita dapat mengontrol persentase data yang akan dideteksi sebagai pencilan oleh algoritma Isolation Forest.


```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(
    n_estimators=20, max_samples=200, random_state=100, contamination=0.015
)
clf.fit(X)
detected_anom_index_train = np.where(clf.predict(X) < 0)
```

## 実際に検出できたかを確認する


```python
# Memvisualisasikan hasil deteksi
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], marker=".", label="Data Normal")
plt.scatter(X[:, 0][anom_ind], X[:, 1][anom_ind], marker="x", s=70, label="Nilai Pencilan")
plt.scatter(
    X[:, 0][detected_anom_index_train],
    X[:, 1][detected_anom_index_train],
    marker="o",
    s=100,
    label="Nilai Pencilan yang Terdeteksi",
    alpha=0.5,
)
plt.legend()
plt.show()
```

    
![png](/images/prep/numerical/Add_label_to_anomaly2_files/Add_label_to_anomaly2_5_0.png)
    
