---
title: "k vecinos más cercanos (k-NN)"
pre: "2.2.7 "
weight: 7
title_suffix: "Aprendizaje perezoso basado en distancias"
---

{{% summary %}}
- k-NN almacena los datos de entrenamiento y predice por mayoría entre los \(k\) vecinos más cercanos del punto a clasificar.
- Los principales hiperparámetros son el número de vecinos \(k\) y el esquema de ponderación de distancias, fáciles de explorar.
- Representa de forma natural fronteras de decisión no lineales, aunque las distancias pierden contraste en alta dimensión (“la maldición de la dimensionalidad”).
- Estandarizar las características o seleccionar las más relevantes estabiliza los cálculos de distancia.
{{% /summary %}}

## Intuición
Bajo la idea de que “muestras cercanas comparten etiqueta”, k-NN busca los \(k\) ejemplos de entrenamiento más próximos al punto objetivo y decide la clase por mayoría (opcionalmente ponderada por la distancia). Como no entrena un modelo explícito de antemano, se le conoce como aprendizaje perezoso.

## Formulación matemática
Para un punto de prueba \(\mathbf{x}\), sea \(\mathcal{N}_k(\mathbf{x})\) el conjunto de los \(k\) vecinos más cercanos. El voto para la clase \(c\) es

$$
v_c = \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \,\mathbb{1}(y_i = c),
$$

donde los pesos \(w_i\) pueden ser uniformes o funciones de la distancia (inversa, gaussiana, etc.). La clase con mayor voto se predice como resultado.

## Experimentos con Python
El siguiente código compara la precisión en validación cruzada para distintos valores de \(k\) y visualiza la frontera de decisión en un conjunto bidimensional.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Cómo cambia la precisión al variar k
X_full, y_full = make_blobs(
    n_samples=600,
    centers=3,
    cluster_std=[1.1, 1.0, 1.2],
    random_state=7,
)
ks = [1, 3, 5, 7, 11]
for k in ks:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
    scores = cross_val_score(model, X_full, y_full, cv=5)
    print(f"k={k}: exactitud CV={scores.mean():.3f} +/- {scores.std():.3f}")

# Visualizar la frontera en 2D
X_vis, y_vis = make_blobs(
    n_samples=450,
    centers=[(-2, 3), (1.8, 2.2), (0.8, -2.5)],
    cluster_std=[1.0, 0.9, 1.1],
    random_state=42,
)
vis_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))
vis_model.fit(X_vis, y_vis)

fig, ax = plt.subplots(figsize=(6, 4.5))
xx, yy = np.meshgrid(
    np.linspace(X_vis[:, 0].min() - 1.5, X_vis[:, 0].max() + 1.5, 300),
    np.linspace(X_vis[:, 1].min() - 1.5, X_vis[:, 1].max() + 1.5, 300),
)
grid = np.column_stack([xx.ravel(), yy.ravel()])
pred = vis_model.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, pred, levels=np.arange(0, 4) - 0.5, cmap="Pastel1", alpha=0.9)

scatter = ax.scatter(
    X_vis[:, 0],
    X_vis[:, 1],
    c=y_vis,
    cmap="Set1",
    edgecolor="#1f2937",
    linewidth=0.6,
)
ax.set_title("Frontera de decisión k-NN (k=5, ponderado por distancia)")
ax.set_xlabel("característica 1")
ax.set_ylabel("característica 2")
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(alpha=0.15)

legend = ax.legend(
    handles=scatter.legend_elements()[0],
    labels=[f"clase {i}" for i in range(len(np.unique(y_vis)))],
    loc="upper right",
    frameon=True,
)
legend.get_frame().set_alpha(0.9)

fig.tight_layout()
```

![knn block 1](/images/basic/classification/knn_block01.svg)

## Referencias
{{% references %}}
<li>Cover, T. M., &amp; Hart, P. E. (1967). Nearest Neighbor Pattern Classification. <i>IEEE Transactions on Information Theory</i>, 13(1), 21–27.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
