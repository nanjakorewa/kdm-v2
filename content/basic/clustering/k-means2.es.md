---
title: "k-means++ | Inicialización inteligente para mejorar el clustering k-means"
linkTitle: "k-means++"
seo_title: "k-means++ | Inicialización inteligente para mejorar el clustering k-means"
weight: 2
pre: "2.5.2 "
title_suffix: "Sembrar centroides con mayor criterio"
searchtitle: "Inicialización k-means++ con Python"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means++ separa los centroides iniciales, lo que reduce la probabilidad de que k-means termine en un óptimo local desfavorable.
- Cada nuevo centro se elige con probabilidad proporcional al cuadrado de su distancia a los centroides ya seleccionados, evitando semillas demasiado juntas.
- En `scikit-learn`, basta con usar `KMeans(init="k-means++")` para comparar con la inicialización aleatoria.
- Variantes a gran escala como mini-batch k-means se basan en k-means++ y son muy usadas en flujos y datasets masivos.
{{% /summary %}}

## Intuición
Si todos los centroides iniciales caen en la misma región densa, k-means convergerá rápido pero con un WCSS elevado. k-means++ atenúa ese riesgo: el primer centro se toma al azar y los siguientes se muestrean según su distancia a los ya elegidos, de manera que la configuración inicial quede bien repartida.

## Formulación matemática
Con los centroides \\(\{\mu_1, \dots, \mu_m\}\\) ya escogidos, la probabilidad de que \\(x\\) sea el siguiente centro viene dada por

$$
P(x) = \frac{D(x)^2}{\sum_{x' \in \mathcal{X}} D(x')^2}, \qquad
D(x) = \min_{1 \le j \le m} \lVert x - \mu_j \rVert.
$$

Los puntos alejados (grande \\(D(x)\\)) reciben más peso y, en consecuencia, la inicialización tiende a cubrir mejor el espacio (Arthur & Vassilvitskii, 2007).

## Demostración en Python
El siguiente ejemplo compara tres reinicios con inicialización aleatoria y k-means++ usando un único paso de actualización para resaltar las diferencias.

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def crear_datos_sinteticos(
    n_samples: int = 3000,
    n_centers: int = 8,
    cluster_std: float = 1.5,
    random_state: int = 11711,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera datos artificiales para comparar inicializaciones."""
    return make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )


def comparar_inicializaciones(
    datos: np.ndarray,
    n_clusters: int = 5,
    subset_size: int = 1000,
    n_trials: int = 3,
    random_state: int = 11711,
) -> dict[str, list[float]]:
    """Comparar inicialización aleatoria frente a k-means++ y graficar resultados.

    Args:
        datos: Matriz de características a agrupar.
        n_clusters: Número de clústeres buscados.
        subset_size: Tamaño de la muestra usada en cada ensayo.
        n_trials: Número de ensayos comparativos.
        random_state: Semilla para reproducibilidad.

    Returns:
        Diccionario con las WCSS obtenidas por cada modo de inicialización.
    """
    rng = np.random.default_rng(random_state)
    inercia_aleatoria: list[float] = []
    inercia_kpp: list[float] = []

    fig, axes = plt.subplots(
        n_trials,
        2,
        figsize=(10, 3.2 * n_trials),
        sharex=True,
        sharey=True,
    )

    for ensayo in range(n_trials):
        indices = rng.choice(len(datos), size=subset_size, replace=False)
        subconjunto = datos[indices]

        modelo_aleatorio = KMeans(
            n_clusters=n_clusters,
            init="random",
            n_init=1,
            max_iter=1,
            random_state=random_state + ensayo,
        ).fit(subconjunto)
        modelo_kpp = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=1,
            random_state=random_state + ensayo,
        ).fit(subconjunto)

        inercia_aleatoria.append(float(modelo_aleatorio.inertia_))
        inercia_kpp.append(float(modelo_kpp.inertia_))

        ax_aleatorio = axes[ensayo, 0] if n_trials > 1 else axes[0]
        ax_kpp = axes[ensayo, 1] if n_trials > 1 else axes[1]

        ax_aleatorio.scatter(subconjunto[:, 0], subconjunto[:, 1], c=modelo_aleatorio.labels_, s=10)
        ax_aleatorio.set_title(f"Inicialización aleatoria (ensayo {ensayo + 1})")
        ax_aleatorio.grid(alpha=0.2)

        ax_kpp.scatter(subconjunto[:, 0], subconjunto[:, 1], c=modelo_kpp.labels_, s=10)
        ax_kpp.set_title(f"k-means++ (ensayo {ensayo + 1})")
        ax_kpp.grid(alpha=0.2)

    fig.suptitle("Asignaciones tras una iteración (aleatorio vs k-means++)")
    fig.tight_layout()
    plt.show()

    return {"aleatorio": inercia_aleatoria, "k-means++": inercia_kpp}


CARACTERISTICAS, _ = crear_datos_sinteticos()
metricas = comparar_inicializaciones(
    datos=CARACTERISTICAS,
    n_clusters=5,
    subset_size=1000,
    n_trials=3,
    random_state=2024,
)
for metodo, valores in metricas.items():
    print(f"{metodo} media WCSS: {np.mean(valores):.1f}")
```


![Comparación de inicializaciones](/images/basic/clustering/k-means2_block01_es.png)

## Referencias
{{% references %}}
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>Bahmani, B., Moseley, B., Vattani, A., Kumar, R., &amp; Vassilvitskii, S. (2012). Scalable k-means++. <i>VLDB</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
