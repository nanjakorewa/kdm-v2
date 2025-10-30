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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import QuantileRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(123)
n_samples = 400
X = np.linspace(0, 10, n_samples)
# Introducir ruido asimétrico
noise = rng.gamma(shape=2.0, scale=1.0, size=n_samples) - 2.0
y = 1.5 * X + 5 + noise
X = X[:, None]

taus = [0.1, 0.5, 0.9]
models = {}
for tau in taus:
    model = make_pipeline(
        StandardScaler(with_mean=True),
        QuantileRegressor(alpha=0.001, quantile=tau, solver="highs"),
    )
    model.fit(X, y)
    models[tau] = model

ols = LinearRegression().fit(X, y)

grid = np.linspace(0, 10, 200)[:, None]
preds = {tau: m.predict(grid) for tau, m in models.items()}
ols_pred = ols.predict(grid)

for tau, pred in preds.items():
    print(f"tau={tau:.1f}, predicción mínima {pred.min():.2f}, máxima {pred.max():.2f}")

plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=15, alpha=0.4, label="observaciones")
colors = {0.1: "#1f77b4", 0.5: "#2ca02c", 0.9: "#d62728"}
for tau, pred in preds.items():
    plt.plot(grid, pred, color=colors[tau], linewidth=2, label=f"cuantil τ={tau}")
plt.plot(grid, ols_pred, color="#9467bd", linestyle="--", label="media (OLS)")
plt.xlabel("entrada X")
plt.ylabel("salida y")
plt.legend()
plt.tight_layout()
plt.show()
```

![quantile-regression block 1](/images/basic/regression/quantile-regression_block01.svg)

### Interpretación de los resultados
- Cada cuantil produce una línea distinta, capturando la dispersión vertical de los datos.
- Frente al modelo enfocado en la media (OLS), la regresión por cuantiles se adapta al ruido sesgado.
- Combinar varios cuantiles genera intervalos de predicción que comunican la incertidumbre relevante para decidir.

## Referencias
{{% references %}}
<li>Koenker, R., &amp; Bassett, G. (1978). Regression Quantiles. <i>Econometrica</i>, 46(1), 33–50.</li>
<li>Koenker, R. (2005). <i>Quantile Regression</i>. Cambridge University Press.</li>
{{% /references %}}
