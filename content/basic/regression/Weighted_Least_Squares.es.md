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
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(7)
n_samples = 200
X = np.linspace(0, 10, n_samples)
true_y = 1.2 * X + 3

# Diferentes niveles de ruido según la región (errores heterocedásticos)
noise_scale = np.where(X < 5, 0.5, 2.5)
y = true_y + rng.normal(scale=noise_scale)

weights = 1.0 / (noise_scale ** 2)  # Usamos el inverso de la varianza como peso
X = X[:, None]

ols = LinearRegression().fit(X, y)
wls = LinearRegression().fit(X, y, sample_weight=weights)

grid = np.linspace(0, 10, 200)[:, None]
ols_pred = ols.predict(grid)
wls_pred = wls.predict(grid)

print("Pendiente OLS:", ols.coef_[0], " Intercepto:", ols.intercept_)
print("Pendiente WLS:", wls.coef_[0], " Intercepto:", wls.intercept_)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, c=noise_scale, cmap="coolwarm", s=25, label="observaciones (color=ruido)")
plt.plot(grid, 1.2 * grid.ravel() + 3, color="#2ca02c", label="recta real")
plt.plot(grid, ols_pred, color="#1f77b4", linestyle="--", linewidth=2, label="OLS")
plt.plot(grid, wls_pred, color="#d62728", linewidth=2, label="WLS")
plt.xlabel("entrada $x$")
plt.ylabel("salida $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![weighted-least-squares block 1](/images/basic/regression/weighted-least-squares_block01.svg)

### Interpretación de los resultados
- El uso de pesos inclina el ajuste hacia la región de bajo ruido y produce estimaciones cercanas a la recta verdadera.
- OLS queda sesgado por la zona ruidosa y subestima la pendiente.
- El rendimiento depende de elegir pesos adecuados; los diagnósticos y el conocimiento del dominio son claves.

## Referencias
{{% references %}}
<li>Carroll, R. J., &amp; Ruppert, D. (1988). <i>Transformation and Weighting in Regression</i>. Chapman &amp; Hall.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
