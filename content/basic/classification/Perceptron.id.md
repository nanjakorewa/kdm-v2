---
title: "Perceptron"
pre: "2.2.3 "
weight: 3
title_suffix: "Klasifikator linear paling sederhana"
---

{{% summary %}}
- Perceptron akan konvergen dalam jumlah pembaruan terbatas jika data dapat dipisahkan secara linear, menjadikannya salah satu algoritme klasifikasi tertua.
- Prediksi menggunakan tanda \(\mathbf{w}^\top \mathbf{x} + b\); jika tanda salah, sampel tersebut memperbarui bobot.
- Aturan pembaruan—menambahkan sampel yang salah diklasifikasikan dikalikan laju belajar—memberikan intuisi awal untuk metode berbasis gradien.
- Jika data tidak dapat dipisahkan secara linear, perluasan fitur atau kernel trick menjadi solusi.
{{% /summary %}}

## Intuisi
Perceptron menggeser batas keputusan setiap kali salah memprediksi, mendorongnya ke sisi yang benar. Vektor bobot \(\mathbf{w}\) tegak lurus terhadap batas, sedangkan bias \(b\) mengatur pergeseran. Laju belajar \(\eta\) mengontrol seberapa besar setiap pergeseran.

## Formulasi matematis
Prediksi dihitung sebagai

$$
\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b).
$$

Jika contoh \((\mathbf{x}_i, y_i)\) salah klasifikasi, perbarui parameter dengan

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i\, \mathbf{x}_i,\qquad
b \leftarrow b + \eta\, y_i.
$$

Untuk data yang terpisahkan secara linear, prosedur ini dijamin konvergen.

## Eksperimen dengan Python
Contoh berikut menerapkan perceptron pada data sintetis, melaporkan jumlah kesalahan per epoch, dan menggambar batas keputusan yang dihasilkan.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=0)
y = np.where(y == 0, -1, 1)

w = np.zeros(X.shape[1])
b = 0.0
lr = 0.1
n_epochs = 20
history = []

for epoch in range(n_epochs):
    errors = 0
    for xi, target in zip(X, y):
        update = lr * target if target * (np.dot(w, xi) + b) <= 0 else 0.0
        if update != 0.0:
            w += update * xi
            b += update
            errors += 1
    history.append(errors)
    if errors == 0:
        break

print("Pelatihan selesai w=", w, " b=", b)
print("Jumlah kesalahan per epoch:", history)

# Gambar batas keputusan
xx = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)
yy = -(w[0] * xx + b) / w[1]
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
plt.plot(xx, yy, color="black", linewidth=2, label="batas keputusan")
plt.xlabel("fitur 1")
plt.ylabel("fitur 2")
plt.legend()
plt.tight_layout()
plt.show()
```

![perceptron block 1](/images/basic/classification/perceptron_block01.svg)

## Referensi
{{% references %}}
<li>Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. <i>Psychological Review</i>, 65(6), 386–408.</li>
<li>Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016). <i>Deep Learning</i>. MIT Press.</li>
{{% /references %}}
