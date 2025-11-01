---
title: "k-means"
weight: 1
pre: "2.5.1 "
title_suffix: "Refinar centroides de forma iterativa"
searchtitle: "k-means paso a paso con Python"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means parte de una idea sencilla —agrupar puntos cercanos— y actualiza repetidamente los centroides (representantes) hasta que las asignaciones dejan de cambiar.
- El objetivo que minimiza es la suma de cuadrados intraclúster (WCSS), es decir la distancia cuadrática entre cada muestra y el centroide de su clúster.
- Con `KMeans` de `scikit-learn` es sencillo visualizar la convergencia, experimentar con inicializaciones y estudiar cómo cambian las asignaciones.
- Para decidir \\(k\\) suelen combinarse criterios como el método del codo o la puntuación de silueta junto con el conocimiento del dominio.
{{% /summary %}}

## Intuición
k-means alterna dos pasos muy simples después de fijar el número de clústeres \\(k\\):

1. Asignar cada muestra al centroide más cercano.
2. Recalcular cada centroide como el promedio de las muestras asignadas.

Cuando estas dos operaciones dejan de modificar las asignaciones (o lo hacen muy poco), consideramos que el algoritmo convergió. La sensibilidad a los valores iniciales y a los atípicos aconseja repetir k-means con distintas semillas o usar inicializaciones más cuidadosas como k-means++.

## Objetivo matemático
Para un conjunto de datos \\(\mathcal{X} = \{x_1, \dots, x_n\}\\) y clústeres \\(\{C_1, \dots, C_k\}\\), k-means minimiza

$$
\min_{C_1, \dots, C_k} \sum_{j=1}^k \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2,
$$

donde \\(\mu_j = |C_j|^{-1} \sum_{x_i \in C_j} x_i\\) es el centroide del clúster \\(j\\). Intuitivamente, intenta que la distancia cuadrática entre cada punto y el “centro de masa” de su clúster sea lo más pequeña posible.

## Demostración en Python
Las siguientes secciones replican la guía en japonés con descripciones en español.

### 1. Generar datos y revisar la colocación inicial
```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def generar_datos(
    n_samples: int = 1000,
    random_state: int = 117_117,
    cluster_std: float = 1.5,
    n_centers: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Crear un conjunto sintético adecuado para ejemplos de k-means."""
    return make_blobs(
        n_samples=n_samples,
        random_state=random_state,
        cluster_std=cluster_std,
        centers=n_centers,
    )


def elegir_centroides_iniciales(
    datos: np.ndarray,
    n_clusters: int,
    threshold: float = -8.0,
) -> np.ndarray:
    """Seleccionar centroides iniciales deterministas por debajo de un umbral."""
    candidatos = datos[datos[:, 1] < threshold]
    if len(candidatos) < n_clusters:
        raise ValueError("No hay suficientes candidatos para los centroides solicitados.")
    return candidatos[:n_clusters]


def visualizar_configuracion_inicial(
    datos: np.ndarray,
    centroides: np.ndarray,
    figsize: tuple[float, float] = (7.5, 7.5),
) -> None:
    """Mostrar el conjunto de datos y resaltar los centroides iniciales."""
    japanize_matplotlib.japanize()
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(datos[:, 0], datos[:, 1], c="#4b5563", marker="x", label="muestras")
    ax.scatter(
        centroides[:, 0],
        centroides[:, 1],
        c="#ef4444",
        marker="o",
        s=80,
        label="centroides iniciales",
    )
    ax.set_title("Datos iniciales y semillas de centroides")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    plt.show()


DATOS_X, DATOS_Y = generar_datos()
CENTROIDES_INICIALES = elegir_centroides_iniciales(DATOS_X, n_clusters=4)
visualizar_configuracion_inicial(DATOS_X, CENTROIDES_INICIALES)
```


![Configuración inicial](/images/basic/clustering/k-means1_block01_es.png)

### 2. Observar la convergencia de los centroides
```python
from typing import Sequence

from sklearn.cluster import KMeans


def visualizar_convergencia(
    datos: np.ndarray,
    centroides_iniciales: np.ndarray,
    max_iters: Sequence[int] = (1, 2, 3, 10),
    random_state: int = 1,
) -> dict[int, float]:
    """Ejecutar k-means con distintos límites de iteración y graficar el resultado."""
    japanize_matplotlib.japanize()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    inercia_por_iteracion: dict[int, float] = {}

    for ax, max_iter in zip(axes.ravel(), max_iters, strict=False):
        modelo = KMeans(
            n_clusters=len(centroides_iniciales),
            init=centroides_iniciales,
            max_iter=max_iter,
            n_init=1,
            random_state=random_state,
        )
        modelo.fit(datos)
        etiquetas = modelo.predict(datos)

        ax.scatter(datos[:, 0], datos[:, 1], c=etiquetas, cmap="tab10", s=10)
        ax.scatter(
            modelo.cluster_centers_[:, 0],
            modelo.cluster_centers_[:, 1],
            c="#dc2626",
            marker="o",
            s=80,
            label="centroides",
        )
        ax.set_title(f"max_iter = {max_iter}")
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
        inercia_por_iteracion[max_iter] = float(modelo.inertia_)

    fig.suptitle("Comportamiento de convergencia según max_iter")
    fig.tight_layout()
    plt.show()
    return inercia_por_iteracion


ESTADISTICAS_CONVERGENCIA = visualizar_convergencia(DATOS_X, CENTROIDES_INICIALES)
for iteracion, inercia in ESTADISTICAS_CONVERGENCIA.items():
    print(f"max_iter={iteracion}: inercia={inercia:,.1f}")
```


![Comparación de convergencia](/images/basic/clustering/k-means1_block02_es.png)

### 3. Aumentar la superposición y revisar las asignaciones
```python
def estudiar_superposicion(
    random_state_base: int = 117_117,
    desviaciones: Sequence[float] = (1.0, 2.0, 3.0, 4.5),
) -> None:
    """Mostrar cómo la superposición complica las asignaciones de k-means."""
    japanize_matplotlib.japanize()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    for ax, std in zip(axes.ravel(), desviaciones, strict=False):
        caracteristicas, _ = make_blobs(
            n_samples=1_000,
            random_state=random_state_base,
            cluster_std=std,
        )
        asignaciones = KMeans(
            n_clusters=2,
            random_state=random_state_base,
        ).fit_predict(caracteristicas)
        ax.scatter(
            caracteristicas[:, 0],
            caracteristicas[:, 1],
            c=asignaciones,
            cmap="tab10",
            s=10,
        )
        ax.set_title(f"cluster_std = {std}")
        ax.grid(alpha=0.2)

    fig.suptitle("Impacto de la superposición en las asignaciones")
    fig.tight_layout()
    plt.show()


estudiar_superposicion()
```


![Comparación de superposición](/images/basic/clustering/k-means1_block03_es.png)

### 4. Comparar diagnósticos para elegir \\(k\\)
```python
from sklearn.metrics import silhouette_score


def evaluar_numero_de_clusters(
    datos: np.ndarray,
    rango_k: Sequence[int] = range(2, 11),
) -> dict[str, list[float]]:
    """Calcular WCSS y la puntuación de silueta para diferentes valores de k."""
    inercias: list[float] = []
    siluetas: list[float] = []

    for k in rango_k:
        modelo = KMeans(n_clusters=k, random_state=117_117).fit(datos)
        inercias.append(float(modelo.inertia_))
        siluetas.append(float(silhouette_score(datos, modelo.labels_)))

    return {"inertia": inercias, "silhouette": siluetas}


def graficar_metricas_de_k(
    metricas: dict[str, list[float]],
    rango_k: Sequence[int],
) -> None:
    """Representar el método del codo y la puntuación de silueta."""
    japanize_matplotlib.japanize()
    ks = list(rango_k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, metricas["inertia"], marker="o")
    axes[0].set_title("Método del codo (WCSS)")
    axes[0].set_xlabel("Número de clústeres k")
    axes[0].set_ylabel("WCSS")
    axes[0].grid(alpha=0.2)

    axes[1].plot(ks, metricas["silhouette"], marker="o", color="#ea580c")
    axes[1].set_title("Puntuación de silueta")
    axes[1].set_xlabel("Número de clústeres k")
    axes[1].set_ylabel("Puntuación")
    axes[1].grid(alpha=0.2)

    fig.tight_layout()
    plt.show()


METRICAS_CODO = evaluar_numero_de_clusters(DATOS_X, range(2, 11))
graficar_metricas_de_k(METRICAS_CODO, range(2, 11))

mejor_k = int(
    range(2, 11)[
        max(
            range(len(METRICAS_CODO["silhouette"])),
            key=METRICAS_CODO["silhouette"].__getitem__,
        )
    ]
)
print(f"La silueta alcanza su máximo en k={mejor_k}")
```


![Diagnósticos de k](/images/basic/clustering/k-means1_block04_es.png)

## Referencias
{{% references %}}
<li>MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. <i>Proceedings of the Fifth Berkeley Symposium</i>.</li>
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
