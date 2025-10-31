---
title: "Regresión Partial Least Squares (PLS)"
pre: "2.1.9 "
weight: 9
title_suffix: "Factores latentes alineados con la variable objetivo"
---

{{% summary %}}
- PLS extrae factores latentes que maximizan la covarianza entre predictores y objetivo antes de realizar la regresión.
- A diferencia de PCA, los ejes aprendidos incorporan información de la variable objetivo, manteniendo el desempeño predictivo mientras se reduce la dimensionalidad.
- Ajustar el número de factores latentes estabiliza el modelo cuando existe fuerte multicolinealidad.
- Analizar los loadings permite identificar qué combinaciones de características se relacionan más con la variable objetivo.
{{% /summary %}}

## Intuición
La regresión por componentes principales solo considera la varianza de las características, por lo que puede descartar direcciones relevantes para el objetivo. PLS construye factores latentes considerando simultáneamente los predictores y la respuesta, conservando la información más útil para predecir. Así es posible trabajar con una representación compacta sin sacrificar precisión.

## Formulación matemática
Dado un matriz de predictores \\(\mathbf{X}\\) y un vector objetivo \\(\mathbf{y}\\), PLS alterna las actualizaciones de las puntuaciones latentes \\(\mathbf{t} = \mathbf{X} \mathbf{w}\\) y \\(\mathbf{u} = \mathbf{y} c\\) de forma que la covarianza \\(\mathbf{t}^\top \mathbf{u}\\) sea máxima. Repetir el proceso produce un conjunto de factores latentes sobre el que se ajusta un modelo lineal

$$
\hat{y} = \mathbf{t} \boldsymbol{b} + b_0.
$$

El número de factores \\(k\\) suele elegirse mediante validación cruzada.

## Experimentos con Python
Comparamos el desempeño de PLS para distintos números de factores latentes con el conjunto de datos de acondicionamiento físico de Linnerud.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import load_linnerud
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_pls_latent_factors(
    cv_splits: int = 5,
    xlabel: str = "Number of latent factors",
    ylabel: str = "CV MSE (lower is better)",
    label_best: str = "best={k}",
    title: str | None = None,
) -> dict[str, object]:
    """Cross-validate PLS regression for different latent factor counts.

    Args:
        cv_splits: Number of folds for cross-validation.
        xlabel: Label for the number-of-factors axis.
        ylabel: Label for the cross-validation error axis.
        label_best: Format string for the best-factor annotation.
        title: Optional plot title.

    Returns:
        Dictionary with the selected factor count, CV score, and loadings.
    """
    japanize_matplotlib.japanize()
    data = load_linnerud()
    X = data["data"]
    y = data["target"][:, 0]

    max_components = min(X.shape[1], 6)
    components = np.arange(1, max_components + 1)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=0)

    scores = []
    pipelines = []
    for k in components:
        model = Pipeline([
            ("scale", StandardScaler()),
            ("pls", PLSRegression(n_components=int(k))),
        ])
        cv_score = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
        ).mean()
        scores.append(cv_score)
        pipelines.append(model)

    scores_arr = np.array(scores)
    best_idx = int(np.argmax(scores_arr))
    best_k = int(components[best_idx])
    best_mse = float(-scores_arr[best_idx])

    best_model = pipelines[best_idx].fit(X, y)
    x_loadings = best_model["pls"].x_loadings_
    y_loadings = best_model["pls"].y_loadings_

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(components, -scores_arr, marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=label_best.format(k=best_k))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "best_k": best_k,
        "best_mse": best_mse,
        "x_loadings": x_loadings,
        "y_loadings": y_loadings,
    }



metrics = evaluate_pls_latent_factors(
    xlabel="Número de factores latentes",
    ylabel="MSE de CV (más bajo es mejor)",
    label_best="mejor k={k}",
    title="Selección de factores en PLS",
)
print(f"Mejor número de factores: {metrics['best_k']}")
print(f"Mejor MSE de CV: {metrics['best_mse']:.3f}")
print("Cargas de X:
", metrics['x_loadings'])
print("Cargas de Y:
", metrics['y_loadings'])

```

![partial-least-squares block 1](/images/basic/regression/partial-least-squares_block01_es.png)

### Interpretación de los resultados
- El MSE de validación cruzada desciende al añadir factores, alcanza un mínimo y luego empeora si seguimos agregando más.
- Inspeccionar `x_loadings_` y `y_loadings_` muestra qué características contribuyen más a cada factor latente.
- Estandarizar las entradas garantiza que características con diferentes escalas aporten de manera equilibrada.

## Referencias
{{% references %}}
<li>Wold, H. (1975). Soft Modelling by Latent Variables: The Non-Linear Iterative Partial Least Squares (NIPALS) Approach. En <i>Perspectives in Probability and Statistics</i>. Academic Press.</li>
<li>Geladi, P., &amp; Kowalski, B. R. (1986). Partial Least-Squares Regression: A Tutorial. <i>Analytica Chimica Acta</i>, 185, 1–17.</li>
{{% /references %}}
