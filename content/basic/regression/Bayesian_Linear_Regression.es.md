---
title: "Regresión lineal bayesiana"
pre: "2.1.6 "
weight: 6
title_suffix: "Cuantificar la incertidumbre de predicción"
---

{{% summary %}}
- La regresión lineal bayesiana trata los coeficientes como variables aleatorias y estima simultáneamente las predicciones y su incertidumbre.
- La distribución posterior se obtiene de forma analítica a partir de la distribución previa y la verosimilitud, lo que la hace robusta con pocos datos o mucho ruido.
- La distribución predictiva es gaussiana, de modo que su media y varianza se pueden visualizar para apoyar la toma de decisiones.
- `BayesianRidge` en scikit-learn ajusta automáticamente la varianza del ruido y facilita el uso práctico del método.
{{% /summary %}}

## Intuición
Los mínimos cuadrados ordinarios buscan un único conjunto “óptimo” de coeficientes, pero en los datos reales esa estimación sigue siendo incierta. La regresión lineal bayesiana modela dicha incertidumbre tratando los coeficientes de forma probabilística y combinando el conocimiento previo con las observaciones. Incluso con pocas muestras obtenemos tanto la predicción esperada como la dispersión alrededor de ella.

## Formulación matemática
Suponemos una distribución previa gaussiana multivariante de media 0 y varianza \(\tau^{-1}\) para el vector de coeficientes \(\boldsymbol\beta\), y ruido gaussiano \(\epsilon_i \sim \mathcal{N}(0, \alpha^{-1})\) en las observaciones. La posterior resulta ser

$$
p(\boldsymbol\beta \mid \mathbf{X}, \mathbf{y}) = \mathcal{N}(\boldsymbol\beta \mid \boldsymbol\mu, \mathbf{\Sigma})
$$

con

$$
\mathbf{\Sigma} = (\alpha \mathbf{X}^\top \mathbf{X} + \tau \mathbf{I})^{-1}, \qquad
\boldsymbol\mu = \alpha \mathbf{\Sigma} \mathbf{X}^\top \mathbf{y}.
$$

La distribución predictiva para una nueva entrada \(\mathbf{x}_*\) también es gaussiana, \(\mathcal{N}(\hat{y}_*, \sigma_*^2)\). `BayesianRidge` estima \(\alpha\) y \(\tau\) a partir de los datos, evitando un ajuste manual.

## Experimentos con Python
El ejemplo siguiente compara mínimos cuadrados ordinarios con la regresión lineal bayesiana en presencia de valores atípicos.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_squared_error

# Crear una tendencia lineal ruidosa e inyectar algunos atípicos
rng = np.random.default_rng(0)
X = np.linspace(-4, 4, 120)
y = 1.8 * X - 0.5 + rng.normal(scale=1.0, size=X.shape)
outlier_idx = rng.choice(len(X), size=6, replace=False)
y[outlier_idx] += rng.normal(scale=8.0, size=outlier_idx.shape)
X = X[:, None]

# Ajustar ambos modelos
ols = LinearRegression().fit(X, y)
bayes = BayesianRidge(compute_score=True).fit(X, y)

grid = np.linspace(-6, 6, 200)[:, None]
ols_mean = ols.predict(grid)
bayes_mean, bayes_std = bayes.predict(grid, return_std=True)

print("MSE de OLS:", mean_squared_error(y, ols.predict(X)))
print("MSE bayesiano:", mean_squared_error(y, bayes.predict(X)))
print("Media posterior de los coeficientes:", bayes.coef_)
print("Diagonal de la varianza posterior:", np.diag(bayes.sigma_))

upper = bayes_mean + 1.96 * bayes_std
lower = bayes_mean - 1.96 * bayes_std

plt.figure(figsize=(10, 5))
plt.scatter(X, y, color="#ff7f0e", alpha=0.6, label="observaciones")
plt.plot(grid, ols_mean, color="#1f77b4", linestyle="--", label="OLS")
plt.plot(grid, bayes_mean, color="#2ca02c", linewidth=2, label="media bayesiana")
plt.fill_between(grid.ravel(), lower, upper, color="#2ca02c", alpha=0.2, label="IC 95%")
plt.xlabel("entrada $x$")
plt.ylabel("salida $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![bayesian-linear-regression block 1](/images/basic/regression/bayesian-linear-regression_block01_es.png)

### Interpretación de los resultados
- OLS se ve arrastrado por los outliers, mientras que la regresión bayesiana mantiene más estable la predicción media.
- `return_std=True` devuelve la desviación estándar predictiva y permite trazar intervalos creíbles con facilidad.
- La varianza posterior revela qué coeficientes conservan mayor incertidumbre.

## Referencias
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
