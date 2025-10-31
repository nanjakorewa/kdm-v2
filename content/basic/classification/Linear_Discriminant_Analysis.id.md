---
title: "Linear Discriminant Analysis (LDA)"
pre: "2.2.4 "
weight: 4
title_suffix: "Mencari arah yang memisahkan kelas"
---

{{% summary %}}
- LDA mencari arah yang memaksimalkan rasio antara variansi antar kelas dan variansi intra kelas, sehingga berguna untuk klasifikasi sekaligus reduksi dimensi.
- Batas keputusan berbentuk \\(\mathbf{w}^\top \mathbf{x} + b = 0\\); di 2D berupa garis dan di 3D berupa bidang, mudah ditafsirkan secara geometris.
- Dengan asumsi tiap kelas berdistribusi Gaussian dan berbagi kovarians yang sama, LDA mendekati klasifikator Bayes optimal.
- `LinearDiscriminantAnalysis` di scikit-learn memudahkan visualisasi batas keputusan dan inspeksi fitur hasil proyeksi.
{{% /summary %}}

## Intuisi
LDA mencari arah proyeksi yang membuat sampel dalam kelas yang sama tetap rapat, namun mendorong sampel dari kelas berbeda saling menjauh. Setelah diproyeksikan, kelas menjadi lebih mudah dipisahkan, sehingga LDA dapat dipakai langsung untuk klasifikasi atau sebagai langkah reduksi dimensi sebelum model lain.

## Formulasi matematis
Untuk dua kelas, arah proyeksi \\(\mathbf{w}\\) memaksimalkan

$$
J(\mathbf{w}) = \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}},
$$

di mana \\(\mathbf{S}_B\\) adalah matriks sebar antar kelas dan \\(\mathbf{S}_W\\) adalah matriks sebar intra kelas. Pada kasus multikelas diperoleh hingga \\(K-1\\) arah proyeksi yang berguna untuk reduksi dimensi.

## Eksperimen dengan Python
Contoh berikut menerapkan LDA pada data sintetis dua kelas, menggambar batas keputusan, dan menampilkan fitur hasil proyeksi 1D. Dengan `transform`, data hasil proyeksi dapat langsung diperoleh.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

rs = 42
X, y = make_blobs(
    n_samples=200, centers=2, n_features=2, cluster_std=2, random_state=rs
)

clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X, y)

w = clf.coef_[0]
b = clf.intercept_[0]

xs = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
ys_boundary = -(w[0] / w[1]) * xs - b / w[1]

scale = 3.0
origin = X.mean(axis=0)
arrow_end = origin + scale * (w / np.linalg.norm(w))

plt.figure(figsize=(7, 7))
plt.title("Batas keputusan dan arah proyeksi LDA", fontsize=16)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.8, label="sampel")
plt.plot(xs, ys_boundary, "k--", lw=1.2, label=r"$\mathbf{w}^\top \mathbf{x} + b = 0$")
plt.arrow(
    origin[0],
    origin[1],
    (arrow_end - origin)[0],
    (arrow_end - origin)[1],
    head_width=0.5,
    length_includes_head=True,
    color="k",
    alpha=0.7,
    label="arah proyeksi $\mathbf{w}$",
)
plt.xlabel("fitur 1")
plt.ylabel("fitur 2")
plt.legend()
plt.xlim(X[:, 0].min() - 2, X[:, 0].max() + 2)
plt.ylim(X[:, 1].min() - 2, X[:, 1].max() + 2)
plt.grid(alpha=0.25)
plt.show()

X_1d = clf.transform(X)[:, 0]

plt.figure(figsize=(8, 4))
plt.title("Proyeksi satu dimensi dengan LDA", fontsize=15)
plt.hist(X_1d[y == 0], bins=20, alpha=0.7, label="kelas 0")
plt.hist(X_1d[y == 1], bins=20, alpha=0.7, label="kelas 1")
plt.xlabel("fitur terproyeksi")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

![linear-discriminant-analysis block 2](/images/basic/classification/linear-discriminant-analysis_block02.svg)

## Referensi
{{% references %}}
<li>Fisher, R. A. (1936). The Use of Multiple Measurements in Taxonomic Problems. <i>Annals of Eugenics</i>, 7(2), 179–188.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
