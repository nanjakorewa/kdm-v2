---
title: "Regresión robusta | Controlar outliers con la pérdida de Huber"
linkTitle: "Regresión robusta"
seo_title: "Regresión robusta | Controlar outliers con la pérdida de Huber"
pre: "2.1.3 "
weight: 3
title_suffix: "Controlar outliers con la pérdida de Huber"
---

{{% summary %}}
- Los mínimos cuadrados ordinarios (OLS) reaccionan con fuerza a los outliers porque los residuos al cuadrado crecen rápidamente y distorsionan el ajuste.
- La pérdida de Huber mantiene la pérdida cuadrática para residuos pequeños y adopta una penalización lineal para residuos grandes, lo que reduce la influencia de valores extremos.
- Ajustar el umbral \\(\delta\\) (epsilon en scikit-learn) y la penalización L2 opcional \\(\alpha\\) permite equilibrar robustez y varianza.
- Combinar escalado con validación cruzada proporciona modelos estables incluso cuando los datos reales mezclan observaciones normales y anomalías.
{{% /summary %}}

## Intuición
Los outliers pueden surgir por fallos de sensores, errores de captura o cambios de régimen. En OLS el residuo de un outlier, al elevarse al cuadrado, domina el ajuste y la recta se desplaza hacia ese punto. La regresión robusta trata los residuos grandes con menos severidad para que el modelo siga la tendencia dominante mientras descuenta observaciones dudosas. La pérdida de Huber es un ejemplo clásico: se comporta como la pérdida cuadrática cerca de cero y como la absoluta en las colas.

## Formulación matemática
Sea el residuo \\(r = y - \hat{y}\\). Para un umbral \\(\delta > 0\\), la pérdida de Huber se define como

$$
\ell_\delta(r) =
\begin{cases}
\dfrac{1}{2} r^2, & |r| \le \delta, \\
\delta \bigl(|r| - \dfrac{1}{2}\delta\bigr), & |r| > \delta.
\end{cases}
$$

Los residuos pequeños se tratan igual que en OLS, pero los grandes crecen de manera lineal. La función de influencia se satura:

$$
\psi_\delta(r) =
\begin{cases}
r, & |r| \le \delta, \\
\delta\,\mathrm{sign}(r), & |r| > \delta.
\end{cases}
$$

En scikit-learn el parámetro `epsilon` corresponde a \\(\delta\\). También se puede añadir una penalización L2 \\(\alpha \lVert \boldsymbol\beta \rVert_2^2\\) para estabilizar los coeficientes cuando las variables están correlacionadas.

## Experimentos con Python
Visualizamos las formas de las pérdidas y comparamos OLS, Ridge y Huber en un conjunto de datos sintético que incluye un outlier extremo.

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

### Pérdida de Huber frente a pérdidas cuadrática y absoluta

```python
def huber_loss(r: np.ndarray, delta: float = 1.5):
    half_sq = 0.5 * np.square(r)
    lin = delta * (np.abs(r) - 0.5 * delta)
    return np.where(np.abs(r) <= delta, half_sq, lin)

delta = 1.5
r_vals = np.arange(-2, 2, 0.01)
h_vals = huber_loss(r_vals, delta=delta)

plt.figure(figsize=(8, 6))
plt.plot(r_vals, np.square(r_vals), "red",   label=r"cuadrática $r^2$")
plt.plot(r_vals, np.abs(r_vals),    "orange",label=r"absoluta $|r|$")
plt.plot(r_vals, h_vals,            "green", label=fr"Huber ($\delta={delta}$)")
plt.axhline(0, color="k", linewidth=0.8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlabel("residuo $r$")
plt.ylabel("pérdida")
plt.title("Comparación de las pérdidas cuadrática, absoluta y Huber")
plt.show()
```

### Conjunto de datos de juguete con un outlier

```python
np.random.seed(42)

N = 30
x1 = np.arange(N)
x2 = np.arange(N)
X = np.c_[x1, x2]
epsilon = np.random.rand(N)
y = 5 * x1 + 10 * x2 + epsilon * 10

y[5] = 500  # introducimos un outlier extremo

plt.figure(figsize=(8, 6))
plt.plot(x1, y, "ko", label="datos")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Datos con un outlier")
plt.show()
```

### Comparación entre OLS, Ridge y Huber

```python
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression

plt.figure(figsize=(8, 6))

huber = HuberRegressor(alpha=0.0, epsilon=3.0)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="Huber")

ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="Ridge (α=1.0)")

ols = LinearRegression()
ols.fit(X, y)
plt.plot(x1, ols.predict(X), "r-", label="OLS")

plt.plot(x1, y, "kx", alpha=0.7)
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Efecto del outlier en distintos regresores")
plt.grid(alpha=0.3)
plt.show()
```

### Interpretación de los resultados
- OLS (rojo) es fuertemente arrastrado por el outlier.
- Ridge (naranja) se estabiliza un poco gracias a la penalización L2, pero sigue viéndose afectado.
- Huber (verde) limita la influencia del outlier y sigue mejor la tendencia principal.

## Referencias
{{% references %}}
<li>Huber, P. J. (1964). Robust Estimation of a Location Parameter. <i>The Annals of Mathematical Statistics</i>, 35(1), 73–101.</li>
<li>Hampel, F. R. et al. (1986). <i>Robust Statistics: The Approach Based on Influence Functions</i>. Wiley.</li>
<li>Huber, P. J., &amp; Ronchetti, E. M. (2009). <i>Robust Statistics</i> (2nd ed.). Wiley.</li>
{{% /references %}}
