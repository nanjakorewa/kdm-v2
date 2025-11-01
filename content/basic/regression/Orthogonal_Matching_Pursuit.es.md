---
title: "Orthogonal Matching Pursuit (OMP) | Selección codiciosa de coeficientes dispersos"
linkTitle: "Orthogonal Matching Pursuit (OMP)"
seo_title: "Orthogonal Matching Pursuit (OMP) | Selección codiciosa de coeficientes dispersos"
pre: "2.1.12 "
weight: 12
title_suffix: "Selección codiciosa de coeficientes dispersos"
---

{{% summary %}}
- Orthogonal Matching Pursuit (OMP) selecciona en cada paso la característica más correlacionada con el residuo para construir un modelo lineal disperso.
- Resuelve un problema de mínimos cuadrados restringido a las características ya elegidas, lo que mantiene los coeficientes interpretables.
- En lugar de ajustar una fuerza de regularización, la dispersión se controla indicando cuántas características deben conservarse.
- Estandarizar las características y revisar la multicolinealidad ayuda a que el método permanezca estable al recuperar señales dispersas.
{{% /summary %}}

## Intuición
Cuando solo unas pocas características son realmente útiles, OMP añade en cada iteración la que más reduce el error residual. Es un algoritmo básico de aprendizaje de diccionarios y codificación dispersa, y produce vectores de coeficientes compactos que resaltan los predictores más relevantes.

## Formulación matemática
Inicializamos el residuo como \\(\mathbf{r}^{(0)} = \mathbf{y}\\). Para cada iteración \\(t\\):

1. Calculamos el producto interno entre cada característica \\(\mathbf{x}_j\\) y el residuo \\(\mathbf{r}^{(t-1)}\\), y elegimos la características \\(j\\) con mayor valor absoluto.
2. Añadimos \\(j\\) al conjunto activo \\(\mathcal{A}_t\\).
3. Resolvemos mínimos cuadrados restringidos a \\(\mathcal{A}_t\\) para obtener \\(\hat{\boldsymbol\beta}_{\mathcal{A}_t}\\).
4. Actualizamos el residuo \\(\mathbf{r}^{(t)} = \mathbf{y} - \mathbf{X}_{\mathcal{A}_t} \hat{\boldsymbol\beta}_{\mathcal{A}_t}\\).

El proceso se detiene al alcanzar la dispersión deseada o cuando el residuo es suficientemente pequeño.

## Experimentos con Python
El siguiente código compara OMP y lasso sobre datos cuyo vector de coeficientes verdadero es disperso.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.metrics import mean_squared_error


def run_omp_vs_lasso(
    n_samples: int = 200,
    n_features: int = 40,
    sparsity: int = 4,
    noise_scale: float = 0.5,
    xlabel: str = "feature index",
    ylabel: str = "coefficient",
    label_true: str = "true",
    label_omp: str = "OMP",
    label_lasso: str = "Lasso",
    title: str | None = None,
) -> dict[str, object]:
    """Compare OMP and lasso on synthetic sparse regression data.

    Args:
        n_samples: Number of training samples to generate.
        n_features: Total number of features in the dictionary.
        sparsity: Count of non-zero coefficients in the ground truth.
        noise_scale: Standard deviation of Gaussian noise added to targets.
        xlabel: Label for the coefficient plot x-axis.
        ylabel: Label for the coefficient plot y-axis.
        label_true: Legend label for the ground-truth bars.
        label_omp: Legend label for the OMP bars.
        label_lasso: Legend label for the lasso bars.
        title: Optional title for the bar chart.

    Returns:
        Dictionary containing recovered supports and MSE values.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(0)

    X = rng.normal(size=(n_samples, n_features))
    true_coef = np.zeros(n_features)
    true_support = rng.choice(n_features, size=sparsity, replace=False)
    true_coef[true_support] = rng.normal(loc=0.0, scale=3.0, size=sparsity)
    y = X @ true_coef + rng.normal(scale=noise_scale, size=n_samples)

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
    omp.fit(X, y)
    lasso = Lasso(alpha=0.05)
    lasso.fit(X, y)

    omp_pred = omp.predict(X)
    lasso_pred = lasso.predict(X)

    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.arange(n_features)
    ax.bar(indices - 0.3, true_coef, width=0.2, label=label_true, color="#2ca02c")
    ax.bar(indices, omp.coef_, width=0.2, label=label_omp, color="#1f77b4")
    ax.bar(indices + 0.3, lasso.coef_, width=0.2, label=label_lasso, color="#d62728")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "true_support": np.flatnonzero(true_coef),
        "omp_support": np.flatnonzero(omp.coef_),
        "lasso_support": np.flatnonzero(np.abs(lasso.coef_) > 1e-6),
        "omp_mse": float(mean_squared_error(y, omp_pred)),
        "lasso_mse": float(mean_squared_error(y, lasso_pred)),
    }



metrics = run_omp_vs_lasso(
    xlabel="índice de característica",
    ylabel="coeficiente",
    label_true="verdadero",
    label_omp="OMP",
    label_lasso="Lasso",
    title="Comparación de OMP y Lasso",
)
print("Soporte verdadero:", metrics['true_support'])
print("Soporte OMP:", metrics['omp_support'])
print("Soporte Lasso:", metrics['lasso_support'])
print(f"MSE de OMP: {metrics['omp_mse']:.4f}")
print(f"MSE de Lasso: {metrics['lasso_mse']:.4f}")

```

### Interpretación de los resultados
- Si `n_nonzero_coefs` coincide con el número real de coeficientes distintos de cero, OMP recupera las características relevantes con alta probabilidad.
- Comparado con lasso, OMP produce coeficientes exactamente nulos para las características descartadas.
- Cuando las características están muy correlacionadas, el orden de selección puede variar; conviene realizar preprocesados y diseño de características adecuados.

## Referencias
{{% references %}}
<li>Pati, Y. C., Rezaiifar, R., &amp; Krishnaprasad, P. S. (1993). Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition. En <i>Conference Record of the Twenty-Seventh Asilomar Conference on Signals, Systems and Computers</i>.</li>
<li>Tropp, J. A. (2004). Greed is Good: Algorithmic Results for Sparse Approximation. <i>IEEE Transactions on Information Theory</i>, 50(10), 2231–2242.</li>
{{% /references %}}
