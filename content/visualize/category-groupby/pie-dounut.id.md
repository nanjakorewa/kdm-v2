---
title: "Bagan donat"
pre: "6.1.3 "
weight: 3
not_use_colab: true
searchtitle: "Membuat bagan donat dalam python"
---

Bagan donat (doughnut graph) adalah jenis bagan pai yang digunakan untuk menampilkan rasio menurut kategori, dengan ruang kosong di tengahnya. Ruang kosong tidak memiliki arti khusus, tetapi dapat digunakan untuk menampilkan statistik keseluruhan (misalnya, "Total XXX yen"). Diagram donat dibuat dalam python menggunakan [matplotlib.pyplot.pie](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html).


```python
import matplotlib.pyplot as plt

# Data
percent = [40, 20, 20, 10, 10]
explode = [0, 0, 0, 0, 0]
labels = ["米国", "エマージング", "日本", "欧州", "その他"]

percent.reverse()
explode.reverse()
labels.reverse()

# Membuat diagram lingkaranMembuat diagram lingkaran
plt.figure(figsize=(7, 7))
plt.pie(x=percent, labels=labels, explode=explode, autopct="%1.0f%%", startangle=90)

# Tambahkan lingkaran kosong di tengah
background_color = "#fff"
p = plt.gcf()
p.gca().add_artist(plt.Circle((0, 0), 0.8, color=background_color))

plt.show()
```


    
![png](/images/visualize/category-groupby/pie-dounut_files/pie-dounut_1_0.png)
    

