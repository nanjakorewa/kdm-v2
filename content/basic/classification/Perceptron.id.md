---
title: "Perceptron | Klasifikator linear paling sederhana"
linkTitle: "Perceptron"
seo_title: "Perceptron | Klasifikator linear paling sederhana"
pre: "2.2.3 "
weight: 3
title_suffix: "Klasifikator linear paling sederhana"
---

{{% summary %}}
- Perceptron akan konvergen dalam jumlah pembaruan terbatas jika data dapat dipisahkan secara linear, menjadikannya salah satu algoritme klasifikasi tertua.
- Prediksi menggunakan tanda \\(\mathbf{w}^\top \mathbf{x} + b\\); jika tanda salah, sampel tersebut memperbarui bobot.
- Aturan pembaruan窶芭enambahkan sampel yang salah diklasifikasikan dikalikan laju belajar窶芭emberikan intuisi awal untuk metode berbasis gradien.
- Jika data tidak dapat dipisahkan secara linear, perluasan fitur atau kernel trick menjadi solusi.
{{% /summary %}}

## Intuisi
Perceptron menggeser batas keputusan setiap kali salah memprediksi, mendorongnya ke sisi yang benar. Vektor bobot \\(\mathbf{w}\\) tegak lurus terhadap batas, sedangkan bias \\(b\\) mengatur pergeseran. Laju belajar \\(\eta\\) mengontrol seberapa besar setiap pergeseran.

## Formulasi matematis
Prediksi dihitung sebagai

$$
\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b).
$$

Jika contoh \\((\mathbf{x}_i, y_i)\\) salah klasifikasi, perbarui parameter dengan

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i\, \mathbf{x}_i,\qquad
b \leftarrow b + \eta\, y_i.
$$

Untuk data yang terpisahkan secara linear, prosedur ini dijamin konvergen.

## Eksperimen dengan Python
Contoh berikut menerapkan perceptron pada data sintetis, melaporkan jumlah kesalahan per epoch, dan menggambar batas keputusan yang dihasilkan.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score


def run_perceptron_demo(
    n_samples: int = 200,
    lr: float = 0.1,
    n_epochs: int = 20,
    random_state: int = 0,
    title: str = "Batas keputusan perceptron",
    xlabel: str = "fitur 1",
    ylabel: str = "fitur 2",
    label_boundary: str = "batas keputusan",
) -> dict[str, object]:
    """Train a perceptron on synthetic blobs and plot the decision boundary."""
    japanize_matplotlib.japanize()
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=1.0, random_state=random_state)
    y_signed = np.where(y == 0, -1, 1)

    w = np.zeros(X.shape[1])
    b = 0.0
    history: list[int] = []

    for _ in range(n_epochs):
        errors = 0
        for xi, target in zip(X, y_signed):
            update = lr * target if target * (np.dot(w, xi) + b) <= 0 else 0.0
            if update != 0.0:
                w += update * xi
                b += update
                errors += 1
        history.append(int(errors))
        if errors == 0:
            break

    preds = np.where(np.dot(X, w) + b >= 0, 1, -1)
    accuracy = float(accuracy_score(y_signed, preds))

    xx = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
    yy = -(w[0] * xx + b) / w[1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
    ax.plot(xx, yy, color="black", linewidth=2, label=label_boundary)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()

    return {"weights": w, "bias": b, "errors": history, "accuracy": accuracy}


metrics = run_perceptron_demo(
    title="Batas keputusan perceptron",
    xlabel="fitur 1",
    ylabel="fitur 2",
    label_boundary="batas keputusan",
)
print(f"Akurasi pelatihan: {metrics['accuracy']:.3f}")
print("Bobot:", metrics['weights'])
print(f"Bias: {metrics['bias']:.3f}")
print("Jumlah kesalahan per epoch:", metrics['errors'])

```


![perceptron block 1](/images/basic/classification/perceptron_block01_id.png)

## Referensi
{{% references %}}
<li>Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. <i>Psychological Review</i>, 65(6), 386窶・08.</li>
<li>Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016). <i>Deep Learning</i>. MIT Press.</li>
{{% /references %}}

