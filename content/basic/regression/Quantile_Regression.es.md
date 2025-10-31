---
title: "Regresión por cuantiles"
pre: "2.1.7 "
weight: 7
title_suffix: "Estimar los contornos de la distribución condicional"
---

{{% summary %}}
- La regresión por cuantiles estima directamente cuantiles arbitrarios —mediana, percentil 10, etc.— en lugar de limitarse a la media.
- Minimizar la pérdida pinball aporta robustez frente a valores atípicos y permite tratar ruido asimétrico.
- Se pueden ajustar modelos independientes por cuantil y combinarlos para construir intervalos de predicción.
- La estandarización y la regularización ayudan a estabilizar la convergencia y a mantener el poder de generalización.
{{% /summary %}}

## Intuición
Mientras que mínimos cuadrados captura el comportamiento medio, la regresión por cuantiles responde a la pregunta “¿con qué frecuencia la respuesta cae por debajo de este valor?”. Resulta ideal para planificar escenarios pesimistas, centrales y optimistas en la demanda, o para estimar el Value at Risk en finanzas, donde la media no basta para decidir.

## Formulación matemática
Con residuo \(r = y - \hat{y}\) y nivel de cuantil \(\tau \in (0, 1)\), la pérdida pinball se define como

$$
L_\tau(r) =
\begin{cases}
\tau r & (r \ge 0) \\
(\tau - 1) r & (r < 0)
\end{cases}
$$

Minimizar esta pérdida produce el predictor lineal del cuantil \(\tau\). Para \(\tau = 0.5\) se recupera la mediana, equivalente a la regresión por desviaciones absolutas.

## Experimentos con Python
Usamos `QuantileRegressor` para estimar los cuantiles 0.1, 0.5 y 0.9, y los comparamos con mínimos cuadrados ordinarios.

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def run_quantile_regression_demo(
    taus: tuple[float, ...] = (0.1, 0.5, 0.9),
    n_samples: int = 400,
    xlabel: str = "input x",
    ylabel: str = "output y",
    label_observations: str = "observations",
    label_mean: str = "mean (OLS)",
    label_template: str = "quantile τ={tau}",
    title: str | None = None,
) -> dict[float, tuple[float, float]]:
    """Fit quantile regressors alongside OLS and plot the conditional bands.

    Args:
        taus: Quantile levels to fit (each in (0, 1)).
        n_samples: Number of synthetic observations to generate.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        label_observations: Legend label for the scatter plot.
        label_mean: Legend label for the OLS line.
        label_template: Format string for quantile labels.
        title: Optional title for the plot.

    Returns:
        Mapping of quantile level to (min prediction, max prediction).
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(123)

    x_values: np.ndarray = np.linspace(0.0, 10.0, n_samples, dtype=float)
    noise: np.ndarray = rng.gamma(shape=2.0, scale=1.0, size=n_samples) - 2.0
    y_values: np.ndarray = 1.5 * x_values + 5.0 + noise
    X: np.ndarray = x_values[:, np.newaxis]

    quantile_models: dict[float, make_pipeline] = {}
    for tau in taus:
        model = make_pipeline(
            StandardScaler(with_mean=True),
            QuantileRegressor(alpha=0.001, quantile=float(tau), solver="highs"),
        )
        model.fit(X, y_values)
        quantile_models[tau] = model

    ols = LinearRegression()
    ols.fit(X, y_values)

    grid: np.ndarray = np.linspace(0.0, 10.0, 200, dtype=float)[:, np.newaxis]
    preds = {tau: model.predict(grid) for tau, model in quantile_models.items()}
    ols_pred: np.ndarray = ols.predict(grid)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(X, y_values, s=15, alpha=0.4, label=label_observations)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4", "#ff7f0e", "#2ca02c"])
    for idx, tau in enumerate(taus):
        color = color_cycle[idx % len(color_cycle)]
        ax.plot(
            grid,
            preds[tau],
            color=color,
            linewidth=2,
            label=label_template.format(tau=tau),
        )

    ax.plot(grid, ols_pred, color="#9467bd", linestyle="--", label=label_mean)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    plt.show()

    summary: dict[float, tuple[float, float]] = {
        tau: (float(pred.min()), float(pred.max())) for tau, pred in preds.items()
    }
    return summary



summary = run_quantile_regression_demo(
    xlabel="entrada x",
    ylabel="salida y",
    label_observations="observaciones",
    label_mean="media (OLS)",
    label_template="cuantil τ={tau}",
    title="Regresión cuantil",
)
for tau, (ymin, ymax) in summary.items():
    print(f"τ={tau:.1f}: predicción mínima {ymin:.2f}, predicción máxima {ymax:.2f}")

```

![quantile-regression block 1](/images/basic/regression/quantile-regression_block01_es.png)

### Interpretación de los resultados
- Cada cuantil produce una línea distinta, capturando la dispersión vertical de los datos.
- Frente al modelo enfocado en la media (OLS), la regresión por cuantiles se adapta al ruido sesgado.
- Combinar varios cuantiles genera intervalos de predicción que comunican la incertidumbre relevante para decidir.

## Referencias
{{% references %}}
<li>Koenker, R., &amp; Bassett, G. (1978). Regression Quantiles. <i>Econometrica</i>, 46(1), 33–50.</li>
<li>Koenker, R. (2005). <i>Quantile Regression</i>. Cambridge University Press.</li>
{{% /references %}}
