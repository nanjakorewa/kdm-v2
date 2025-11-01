---
title: "DBSCAN"
pre: "2.5.4 "
weight: 4
title_suffix: "Clustering basado en densidad"
searchtitle: "Ejemplo de DBSCAN con Python"
---

{{% summary %}}
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifica regiones densas como clústeres y deja el resto como ruido.
- Los parámetros `eps` (radio de búsqueda) y `min_samples` (vecinos mínimos) clasifican los puntos en núcleo, frontera o ruido.
- A diferencia de k-means, no requiere fijar el número de clústeres y admite formas no convexas y ruido abundante.
- Un gráfico de distancias ordenadas ayuda a ajustar `eps`; estandarizar variables hace que las distancias sean comparables.
{{% /summary %}}

## Intuición
DBSCAN forma clústeres conectando puntos que tienen suficientes vecinos dentro de `eps`.

- **Núcleo**: al menos `min_samples` vecinos en el radio `eps`.
- **Frontera**: vecino de un punto núcleo, pero sin suficientes vecinos propios.
- **Ruido**: no es núcleo ni frontera.

Los puntos núcleo que se alcanzan mutuamente por cadenas de densidad forman un clúster. Los puntos frontera se adhieren al clúster más cercano y los restantes se etiquetan como ruido.

## Formulación
Para un punto \\(x_i\\) definimos la vecindad \\(\varepsilon\\)-radius como

$$
\mathcal{N}_\varepsilon(x_i) = \{\, x_j \in \mathcal{X} \mid \lVert x_i - x_j \rVert \le \varepsilon \,\},
$$

y un punto es núcleo si

$$
|\mathcal{N}_\varepsilon(x_i)| \ge \texttt{min\_samples}.
$$

La conectividad por densidad agrupa los núcleos, los bordes se unen al clúster correspondiente y el resto queda como ruido.

## Ejemplo en Python
Agrupamos datos en forma de lunas y mostramos cuántos clústeres y puntos ruidosos detecta DBSCAN.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def ejecutar_demo_dbscan(
    n_samples: int = 600,
    noise: float = 0.08,
    eps: float = 0.3,
    min_samples: int = 10,
    random_state: int = 0,
) -> dict[str, int]:
    """Aplica DBSCAN sobre datos lunares y devuelve conteos de cluster y ruido."""
    features, _ = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )
    features = StandardScaler().fit_transform(features)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)

    unique_labels = sorted(np.unique(labels))
    cluster_ids = [label for label in unique_labels if label != -1]
    noise_count = int(np.sum(labels == -1))

    core_mask = np.zeros(labels.shape[0], dtype=bool)
    if hasattr(model, "core_sample_indices_"):
        core_mask[model.core_sample_indices_] = True

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    palette = plt.cm.get_cmap("tab10", max(len(cluster_ids), 1))

    for order, cluster_id in enumerate(cluster_ids):
        mask = labels == cluster_id
        colour = palette(order)
        if np.any(mask & core_mask):
            ax.scatter(
                features[mask & core_mask, 0],
                features[mask & core_mask, 1],
                c=[colour],
                s=36,
                edgecolor="white",
                linewidth=0.2,
                label=f"cluster {cluster_id} (nucleo)",
            )
        if np.any(mask & ~core_mask):
            ax.scatter(
                features[mask & ~core_mask, 0],
                features[mask & ~core_mask, 1],
                c=[colour],
                s=24,
                edgecolor="white",
                linewidth=0.2,
                marker="o",
                label=f"cluster {cluster_id} (frontera)",
            )

    if noise_count:
        noise_mask = labels == -1
        ax.scatter(
            features[noise_mask, 0],
            features[noise_mask, 1],
            c="#9ca3af",
            marker="x",
            s=28,
            linewidth=0.8,
            label="ruido",
        )

    ax.set_title("DBSCAN clustering")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    plt.show()

    return {"n_clusters": len(cluster_ids), "n_noise": noise_count}


resultado = ejecutar_demo_dbscan()
print(f"Numero de clusters detectados: {resultado['n_clusters']}")
print(f"Numero de puntos de ruido: {resultado['n_noise']}")
```


![Resultado de DBSCAN](/images/basic/clustering/dbscan_block01_es.png)

## Referencias
{{% references %}}
<li>Ester, M., Kriegel, H.-P., Sander, J., &amp; Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. <i>KDD</i>.</li>
<li>Schubert, E., Sander, J., Ester, M., Kriegel, H.-P., &amp; Xu, X. (2017). DBSCAN Revisited, Revisited. <i>ACM Transactions on Database Systems</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
