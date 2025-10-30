---
title: "Perceptrón"
pre: "2.2.3 "
weight: 3
title_suffix: "El clasificador lineal más simple"
---

{{% summary %}}
- El perceptrón converge en un número finito de actualizaciones si los datos son linealmente separables, siendo uno de los algoritmos de clasificación más antiguos.
- La predicción se basa en el signo de \(\mathbf{w}^\top \mathbf{x} + b\); cuando la señal es incorrecta, ese ejemplo actualiza los pesos.
- La regla de actualización —sumar el ejemplo mal clasificado escalado por la tasa de aprendizaje— ofrece una introducción intuitiva a los métodos basados en gradiente.
- Si los datos no son separables linealmente, conviene ampliar características o recurrir a kernel tricks.
{{% /summary %}}

## Intuición
El perceptrón mueve la frontera de decisión cada vez que se equivoca, desplazándola hacia el lado correcto. El vector de pesos \(\mathbf{w}\) es normal a la frontera, mientras que el sesgo \(b\) ajusta el desplazamiento. La tasa de aprendizaje \(\eta\) controla la magnitud de cada movimiento.

## Formulación matemática
La predicción se calcula como

$$
\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b).
$$

Si un ejemplo \((\mathbf{x}_i, y_i)\) queda mal clasificado, se actualiza mediante

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i\, \mathbf{x}_i,\qquad
b \leftarrow b + \eta\, y_i.
$$

Cuando los datos son separables linealmente, este procedimiento converge.

## Experimentos con Python
El siguiente ejemplo aplica el perceptrón a datos sintéticos, muestra el número de errores por época y dibuja la frontera obtenida.

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

print("Entrenamiento finalizado w=", w, " b=", b)
print("Errores por época:", history)

# Dibujar la frontera de decisión
xx = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)
yy = -(w[0] * xx + b) / w[1]
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
plt.plot(xx, yy, color="black", linewidth=2, label="frontera de decisión")
plt.xlabel("característica 1")
plt.ylabel("característica 2")
plt.legend()
plt.tight_layout()
plt.show()
```

![perceptron block 1](/images/basic/classification/perceptron_block01.svg)

## Referencias
{{% references %}}
<li>Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. <i>Psychological Review</i>, 65(6), 386–408.</li>
<li>Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016). <i>Deep Learning</i>. MIT Press.</li>
{{% /references %}}
