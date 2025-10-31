---
title: "Perceptrﾃｳn"
pre: "2.2.3 "
weight: 3
title_suffix: "El clasificador lineal mﾃ｡s simple"
---

{{% summary %}}
- El perceptrﾃｳn converge en un nﾃｺmero finito de actualizaciones si los datos son linealmente separables, siendo uno de los algoritmos de clasificaciﾃｳn mﾃ｡s antiguos.
- La predicciﾃｳn se basa en el signo de \\(\mathbf{w}^\top \mathbf{x} + b\\); cuando la seﾃｱal es incorrecta, ese ejemplo actualiza los pesos.
- La regla de actualizaciﾃｳn 窶敗umar el ejemplo mal clasificado escalado por la tasa de aprendizaje窶・ofrece una introducciﾃｳn intuitiva a los mﾃｩtodos basados en gradiente.
- Si los datos no son separables linealmente, conviene ampliar caracterﾃｭsticas o recurrir a kernel tricks.
{{% /summary %}}

## Intuiciﾃｳn
El perceptrﾃｳn mueve la frontera de decisiﾃｳn cada vez que se equivoca, desplazﾃ｡ndola hacia el lado correcto. El vector de pesos \\(\mathbf{w}\\) es normal a la frontera, mientras que el sesgo \\(b\\) ajusta el desplazamiento. La tasa de aprendizaje \\(\eta\\) controla la magnitud de cada movimiento.

## Formulaciﾃｳn matemﾃ｡tica
La predicciﾃｳn se calcula como

$$
\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b).
$$

Si un ejemplo \\((\mathbf{x}_i, y_i)\\) queda mal clasificado, se actualiza mediante

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i\, \mathbf{x}_i,\qquad
b \leftarrow b + \eta\, y_i.
$$

Cuando los datos son separables linealmente, este procedimiento converge.

## Experimentos con Python
El siguiente ejemplo aplica el perceptrﾃｳn a datos sintﾃｩticos, muestra el nﾃｺmero de errores por ﾃｩpoca y dibuja la frontera obtenida.

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
    title: str = "Frontera de decisiﾃｳn del perceptrﾃｳn",
    xlabel: str = "caracterﾃｭstica 1",
    ylabel: str = "caracterﾃｭstica 2",
    label_boundary: str = "frontera de decisiﾃｳn",
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
    title="Frontera de decisiﾃｳn del perceptrﾃｳn",
    xlabel="caracterﾃｭstica 1",
    ylabel="caracterﾃｭstica 2",
    label_boundary="frontera de decisiﾃｳn",
)
print(f"Precisiﾃｳn de entrenamiento: {metrics['accuracy']:.3f}")
print("Pesos:", metrics['weights'])
print(f"Sesgo: {metrics['bias']:.3f}")
print("Errores por ﾃｩpoca:", metrics['errors'])

```


![perceptron block 1](/images/basic/classification/perceptron_block01_es.png)

## Referencias
{{% references %}}
<li>Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. <i>Psychological Review</i>, 65(6), 386窶・08.</li>
<li>Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016). <i>Deep Learning</i>. MIT Press.</li>
{{% /references %}}

