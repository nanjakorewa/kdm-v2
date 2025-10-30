---
title: "Regresión lineal y mínimos cuadrados ordinarios"
pre: "2.1.1 "
weight: 1
title_suffix: "Comprender desde los primeros principios"
---

{{% summary %}}
- La regresión lineal modela la relación lineal entre entrada y salida y sirve como base tanto para la predicción como para la interpretación.
- El método de mínimos cuadrados ordinarios estima los coeficientes minimizando la suma de los residuos al cuadrado y ofrece una solución en forma cerrada.
- La pendiente indica cuánto cambia la salida cuando la entrada aumenta una unidad, mientras que la ordenada al origen representa el valor esperado cuando la entrada es cero.
- Cuando el ruido o los valores atípicos son grandes conviene combinar estandarización y variantes robustas para mantener fiable el preprocesamiento y la evaluación.
{{% /summary %}}

## Intuición
Cuando la nube de puntos \((x_i, y_i)\) forma aproximadamente una línea recta, prolongarla es una forma natural de interpolar nuevas entradas. Mínimos cuadrados ordinarios dibuja una recta que pasa lo más cerca posible de todos los puntos haciendo que la desviación total entre las observaciones y la recta sea mínima.

## Formulación matemática
Un modelo lineal univariado se expresa como

$$
y = w x + b.
$$

Al minimizar la suma de cuadrados de los residuos \(\epsilon_i = y_i - (w x_i + b)\)

$$
L(w, b) = \sum_{i=1}^{n} \big(y_i - (w x_i + b)\big)^2,
$$

obtenemos la solución analítica

$$
w = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}, \qquad b = \bar{y} - w \bar{x},
$$

donde \(\bar{x}\) y \(\bar{y}\) son las medias de \(x\) y \(y\). La misma idea se extiende a la regresión multivariante usando vectores y matrices.

## Experimentos con Python
El siguiente ejemplo ajusta una recta con `scikit-learn` y dibuja el resultado. El código es el mismo que en la versión japonesa para mantener las figuras sincronizadas.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # opcional; conserva etiquetas en japonés si se ejecuta el notebook
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Generar datos
n_samples = 100
X = np.linspace(-5, 5, n_samples)[:, np.newaxis]
epsilon = np.random.normal(scale=2, size=n_samples)
y = 2 * X.ravel() + 1 + epsilon  # relación verdadera y = 2x + 1 + ruido

# Ajustar el modelo con estandarización opcional (útil para varias características)
lin_reg = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
y_pred = lin_reg.predict(X)

# Gráfica
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="observaciones", c="orange")
plt.plot(X, y_pred, label="regresión lineal (OLS)")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![linear-regression block 1](/images/basic/regression/linear-regression_block01.svg)

### Interpretación de los resultados
- **Pendiente \(w\)**: muestra cuánto aumenta o disminuye la salida cuando la entrada crece una unidad; la estimación debería acercarse al valor real.
- **Ordenada \(b\)**: representa el valor esperado cuando la entrada es 0 y ajusta la posición vertical de la recta.
- Estandarizar las características con `StandardScaler` estabiliza el aprendizaje cuando las escalas de entrada difieren.

## Referencias
{{% references %}}
<li>Draper, N. R., &amp; Smith, H. (1998). <i>Applied Regression Analysis</i> (3rd ed.). John Wiley &amp; Sons.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
