---
title: "Regresión polinómica"
pre: "2.1.4 "
weight: 4
title_suffix: "Capturar patrones no lineales con un modelo lineal"
---

{{% summary %}}
- La regresión polinómica amplía las características con potencias para que un modelo lineal pueda ajustar relaciones no lineales.
- El modelo sigue siendo una combinación lineal de coeficientes, por lo que mantiene soluciones cerradas e interpretabilidad.
- A mayor grado mayor expresividad, pero también riesgo de sobreajuste; por ello la regularización y la validación cruzada son esenciales.
- Estandarizar las características y ajustar el grado junto con la penalización produce predicciones estables.
{{% /summary %}}

## Intuición
Una recta no puede describir curvas suaves o patrones en forma de colina. Al expandir la entrada con términos polinómicos—\(x, x^2, x^3, \dots\) en el caso univariado o potencias e interacciones en el multivariado—expresamos comportamientos no lineales sin abandonar el marco lineal.

## Formulación matemática
Para \(\mathbf{x} = (x_1, \dots, x_m)\) generamos un vector de características polinómicas \(\phi(\mathbf{x})\) hasta el grado \(d\). Por ejemplo, si \(m = 2\) y \(d = 2\),

$$
\phi(\mathbf{x}) = (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2),
$$

y el modelo queda

$$
y = \mathbf{w}^\top \phi(\mathbf{x}).
$$

Como el número de términos crece rápidamente con el grado, en la práctica se comienza con grados bajos (2 o 3) y se combina con regularización (p. ej., Ridge) cuando hace falta.

## Experimentos con Python
Añadimos características de grado tres y ajustamos una curva a datos generados a partir de una función cúbica con ruido.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 200)
y_true = 0.5 * X**3 - 1.2 * X**2 + 2.0 * X + 1.5
y = y_true + rng.normal(scale=2.0, size=X.shape)

X = X[:, None]

linear_model = LinearRegression().fit(X, y)
poly_model = make_pipeline(
    PolynomialFeatures(degree=3, include_bias=False),
    LinearRegression()
).fit(X, y)

grid = np.linspace(-3.5, 3.5, 300)[:, None]
linear_pred = linear_model.predict(grid)
poly_pred = poly_model.predict(grid)
true_curve = 0.5 * grid.ravel()**3 - 1.2 * grid.ravel()**2 + 2.0 * grid.ravel() + 1.5

print("MSE regresión lineal:", mean_squared_error(y, linear_model.predict(X)))
print("MSE regresión polinómica grado 3:", mean_squared_error(y, poly_model.predict(X)))

plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=20, color="#ff7f0e", alpha=0.6, label="observaciones")
plt.plot(grid, true_curve, color="#2ca02c", linewidth=2, label="curva real")
plt.plot(grid, linear_pred, color="#1f77b4", linestyle="--", linewidth=2, label="regresión lineal")
plt.plot(grid, poly_pred, color="#d62728", linewidth=2, label="polinómica grado 3")
plt.xlabel("entrada $x$")
plt.ylabel("salida $y$")
plt.legend()
plt.tight_layout()
plt.show()
```

![polynomial-regression block 1](/images/basic/regression/polynomial-regression_block01_es.png)

### Interpretación de los resultados
- La regresión lineal simple falla en reproducir la curvatura, especialmente cerca del centro, mientras que el modelo cúbico sigue la curva verdadera.
- Subir el grado mejora el ajuste en entrenamiento pero puede volver inestables las extrapolaciones.
- Combinar características polinómicas con una regresión regularizada (p. ej., Ridge) en una tubería ayuda a controlar el sobreajuste.

## Referencias
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
