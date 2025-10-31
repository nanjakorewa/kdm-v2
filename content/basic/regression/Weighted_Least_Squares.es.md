---
title: "Mínimos cuadrados ponderados (WLS)"
pre: "2.1.11 "
weight: 11
title_suffix: "Tratar observaciones con varianza desigual"
---

{{% summary %}}
- WLS asigna pesos específicos a cada observación para que las mediciones confiables tengan mayor influencia en la recta ajustada.
- Al multiplicar el error cuadrado por los pesos se atenúan las observaciones de alta varianza y se mantiene el ajuste cercano a los datos fiables.
- Puede ejecutarse con `LinearRegression` de scikit-learn simplemente proporcionando `sample_weight`.
- Los pesos pueden provenir de varianzas conocidas, diagnósticos de residuos o conocimiento del dominio; diseñarlos con cuidado es fundamental.
{{% /summary %}}

## Intuición
Los mínimos cuadrados ordinarios suponen que todas las observaciones son igual de fiables, pero en la práctica la precisión de los sensores varía. WLS “escucha” más a los puntos de datos en los que confiamos al darles un peso mayor durante el ajuste, lo que permite manejar datos heterocedásticos dentro del marco lineal habitual.

## Formulación matemática
Con pesos positivos \(w_i\) minimizamos

$$
L(\boldsymbol\beta, b) = \sum_{i=1}^{n} w_i \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2.
$$

La elección ideal es \(w_i \propto 1/\sigma_i^2\) (el inverso de la varianza), de modo que las observaciones más precisas aporten más.

## Experimentos con Python
Comparamos OLS y WLS en datos cuyo nivel de ruido cambia por regiones.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


def run_weighted_vs_ols(
    n_samples: int = 200,
    threshold: float = 5.0,
    low_noise: float = 0.5,
    high_noise: float = 2.5,
    xlabel: str = "input $",
    ylabel: str = "output $",
    label_scatter: str = "observations (color=noise)",
    label_truth: str = "true line",
    label_ols: str = "OLS",
    label_wls: str = "WLS",
    title: str | None = None,
) -> dict[str, float]:
    """Compare OLS and weighted least squares on heteroscedastic data.

    Args:
        n_samples: Number of evenly spaced samples to generate.
        threshold: Breakpoint separating low- and high-noise regions.
        low_noise: Noise scale for the lower region.
        high_noise: Noise scale for the higher region.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        label_scatter: Legend label for the colored scatter plot.
        label_truth: Legend label for the ground-truth line.
        label_ols: Legend label for the OLS fit.
        label_wls: Legend label for the WLS fit.
        title: Optional title for the plot.

    Returns:
        Dictionary with slopes and intercepts of both fits.
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(7)

    X_vals: np.ndarray = np.linspace(0.0, 10.0, n_samples, dtype=float)
    true_y: np.ndarray = 1.2 * X_vals + 3.0

    noise_scale = np.where(X_vals < threshold, low_noise, high_noise)
    y_noisy = true_y + rng.normal(scale=noise_scale)

    weights = 1.0 / (noise_scale**2)
    X = X_vals[:, np.newaxis]

    ols = LinearRegression()
    ols.fit(X, y_noisy)

    wls = LinearRegression()
    wls.fit(X, y_noisy, sample_weight=weights)

    grid = np.linspace(0.0, 10.0, 200, dtype=float)[:, np.newaxis]
    ols_pred = ols.predict(grid)
    wls_pred = wls.predict(grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        X,
        y_noisy,
        c=noise_scale,
        cmap="coolwarm",
        s=25,
        label=label_scatter,
    )
    ax.plot(grid, 1.2 * grid.ravel() + 3.0, color="#2ca02c", label=label_truth)
    ax.plot(grid, ols_pred, color="#1f77b4", linestyle="--", linewidth=2, label=label_ols)
    ax.plot(grid, wls_pred, color="#d62728", linewidth=2, label=label_wls)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    return {
        "ols_slope": float(ols.coef_[0]),
        "ols_intercept": float(ols.intercept_),
        "wls_slope": float(wls.coef_[0]),
        "wls_intercept": float(wls.intercept_),
    }



metrics = run_weighted_vs_ols(
    xlabel="entrada $",
    ylabel="salida $",
    label_scatter="observaciones (color=ruido)",
    label_truth="línea real",
    label_ols="OLS",
    label_wls="WLS",
    title="Comparación entre OLS y WLS",
)
print(f"Pendiente OLS: {metrics['ols_slope']:.3f}, intercepto: {metrics['ols_intercept']:.3f}")
print(f"Pendiente WLS: {metrics['wls_slope']:.3f}, intercepto: {metrics['wls_intercept']:.3f}")

```

![weighted-least-squares block 1](/images/basic/regression/weighted-least-squares_block01_es.png)

### Interpretación de los resultados
- El uso de pesos inclina el ajuste hacia la región de bajo ruido y produce estimaciones cercanas a la recta verdadera.
- OLS queda sesgado por la zona ruidosa y subestima la pendiente.
- El rendimiento depende de elegir pesos adecuados; los diagnósticos y el conocimiento del dominio son claves.

## Referencias
{{% references %}}
<li>Carroll, R. J., &amp; Ruppert, D. (1988). <i>Transformation and Weighting in Regression</i>. Chapman &amp; Hall.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
