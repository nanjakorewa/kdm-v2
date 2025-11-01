---
title: "X-means | Estimando automáticamente el número de clústeres"
linkTitle: "X-means"
seo_title: "X-means | Estimando automáticamente el número de clústeres"
weight: 3
pre: "2.5.3 "
title_suffix: "Estimando automáticamente el número de clústeres"
searchtitle: "Clustering X-means con Python"
---

{{< katex />}}
{{% youtube "2hkyJcWctUA" %}}

{{% summary %}}
- X-means toma a k-means como base, reparte mejor los centroides iniciales y divide los clústeres solo cuando merece la pena según un criterio de selección de modelos.
- Cada división candidata compara la BIC (Bayesian Information Criterion); solo se acepta cuando dicha puntuación aumenta.
- Con un pequeño utilitario de BIC y `KMeans` de scikit-learn es posible construir una versión ligera del algoritmo.
- El método resulta útil cuando no queremos (o no sabemos) fijar \\(k\\) a priori: las divisiones se realizan únicamente cuando los datos lo justifican.
{{% /summary %}}

## Intuición
En k-means conviene acertar con \\(k\\); de lo contrario, los clústeres se fusionan o se fragmentan. X-means arranca con un \\(k\\) moderado, ejecuta k-means y, a continuación, prueba a dividir cada clúster. Si la BIC del modelo dividido supera a la del original, se adopta la división; de lo contrario, se mantiene el clúster tal cual. Repetir este proceso conduce de forma natural a un \\(k\\) que equilibra ajuste y complejidad.

## Recordatorio matemático
Para un conjunto de clústeres \\(\{C_1, \dots, C_k\}\\) con centros \\(\{\mu_1, \dots, \mu_k\}\\), la BIC se define como

$$
\mathrm{BIC} = \ln L - \tfrac{p}{2} \ln n,
$$

donde \\(L\\) es la verosimilitud del modelo, \\(p\\) el número de parámetros libres y \\(n\\) la cantidad de muestras. Si probamos a dividir un clúster en dos, comparamos las BIC de ambos modelos y conservamos el que obtenga mayor puntuación.

## Demostración en Python
A continuación implementamos un divisor estilo X-means y lo comparamos con un k-means fijo de \\(k=4\\).

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def generar_datos(
    n_samples: int = 1200,
    n_centros: int = 9,
    cluster_std: float = 1.1,
    random_state: int = 1707,
) -> NDArray[np.float64]:
    """Genera un conjunto sintético para experimentar con clustering."""
    rasgos, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centros,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return rasgos


def calcular_bic(
    datos: NDArray[np.float64],
    etiquetas: NDArray[np.int64],
    centros: NDArray[np.float64],
) -> float:
    """Calcula una puntuación BIC adaptada a clústeres tipo k-means."""
    n_muestras, n_caracteristicas = datos.shape
    n_clust = centros.shape[0]
    if n_clust <= 0:
        raise ValueError("El número de clústeres debe ser positivo.")

    tamaños = np.bincount(etiquetas, minlength=n_clust)
    sse = 0.0
    for idx, centro in enumerate(centros):
        if tamaños[idx] == 0:
            continue
        dif = datos[etiquetas == idx] - centro
        sse += float((dif**2).sum())

    varianza = sse / max(n_muestras - n_clust, 1)
    if varianza <= 0:
        varianza = 1e-6

    log_verosimilitud = 0.0
    constante = -0.5 * n_caracteristicas * np.log(2 * np.pi * varianza)
    for tam in tamaños:
        if tam > 0:
            log_verosimilitud += tam * constante - 0.5 * (tam - 1)

    n_param = n_clust * (n_caracteristicas + 1)
    bic = log_verosimilitud - 0.5 * n_param * np.log(n_muestras)
    return float(bic)


def dividir_xmeans(
    datos: NDArray[np.float64],
    k_max: int = 20,
    random_state: int = 42,
) -> NDArray[np.int64]:
    """Aproxima X-means dividiendo clústeres cuando la BIC mejora."""
    rng = np.random.default_rng(random_state)
    pendientes = [np.arange(len(datos))]
    aceptados: list[NDArray[np.int64]] = []

    while pendientes:
        indices = pendientes.pop()
        subconjunto = datos[indices]

        if len(indices) <= 2:
            aceptados.append(indices)
            continue

        etiquetas_padre = np.zeros(len(indices), dtype=int)
        centro_padre = subconjunto.mean(axis=0, keepdims=True)
        bic_padre = calcular_bic(subconjunto, etiquetas_padre, centro_padre)

        modelo_split = KMeans(
            n_clusters=2,
            n_init=10,
            max_iter=200,
            random_state=rng.integers(0, 1_000_000),
        ).fit(subconjunto)
        bic_hijos = calcular_bic(
            subconjunto,
            modelo_split.labels_,
            modelo_split.cluster_centers_,
        )

        if bic_hijos > bic_padre and (len(aceptados) + len(pendientes) + 1) < k_max:
            pendientes.append(indices[modelo_split.labels_ == 0])
            pendientes.append(indices[modelo_split.labels_ == 1])
        else:
            aceptados.append(indices)

    etiquetas_finales = np.empty(len(datos), dtype=int)
    for idx_clust, idx_muestras in enumerate(aceptados):
        etiquetas_finales[idx_muestras] = idx_clust
    return etiquetas_finales


def ejecutar_demo_xmeans(
    random_state: int = 2024,
    k_inicial: int = 4,
    k_max: int = 18,
) -> dict[str, object]:
    """Comparar k-means con k fijo frente a la versión X-means ligera."""
    datos = generar_datos(random_state=random_state)

    etiquetas_kmeans = KMeans(
        n_clusters=k_inicial,
        n_init=10,
        random_state=random_state,
    ).fit_predict(datos)
    etiquetas_xmeans = dividir_xmeans(datos, k_max=k_max, random_state=random_state + 99)

    clust_unicos = np.unique(etiquetas_xmeans)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(datos[:, 0], datos[:, 1], c=etiquetas_kmeans, cmap="tab20", s=10)
    axes[0].set_title(f"k-means (k={k_inicial})")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(datos[:, 0], datos[:, 1], c=etiquetas_xmeans, cmap="tab20", s=10)
    axes[1].set_title(f"X-means (k={len(clust_unicos)})")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Comparación entre k-means y X-means")
    fig.tight_layout()
    plt.show()

    return {
        "kmeans_clusters": int(k_inicial),
        "xmeans_clusters": int(len(clust_unicos)),
        "cluster_sizes": np.bincount(etiquetas_xmeans).tolist(),
    }


resultado = ejecutar_demo_xmeans()
print(f"Clusters pedidos a k-means: {resultado['kmeans_clusters']}")
print(f"Clusters descubiertos por X-means: {resultado['xmeans_clusters']}")
print(f"Tamanos de los clusters X-means: {resultado['cluster_sizes']}")
```


![Comparación k-means vs X-means](/images/basic/clustering/x-means_block01_es.png)

## Referencias
{{% references %}}
<li>Pelleg, D., &amp; Moore, A. W. (2000). X-means: Extending k-means with Efficient Estimation of the Number of Clusters. <i>ICML</i>.</li>
<li>Bahmani, B., Moseley, B., Vattani, A., Kumar, R., &amp; Vassilvitskii, S. (2012). Scalable k-means++. <i>VLDB</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
