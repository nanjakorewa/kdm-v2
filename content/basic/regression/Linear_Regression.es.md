---
title: "Método de mínimos cuadrados"
pre: "2.1.1 "
weight: 1
title_suffix: "Conceptos e implementación"
---

{{% youtube "KKuAxQbuJpk" %}}




<div class="pagetop-box">
  <p>El <b>método de mínimos cuadrados</b> busca los coeficientes de una función que mejor ajusta pares de observaciones <code>(x_i, y_i)</code>, minimizando la suma de los residuos al cuadrado. Nos centramos en el caso más simple, una recta <code>y = wx + b</code>, y revisamos la intuición y una implementación práctica.</p>
  </div>

{{% notice tip %}}
Las fórmulas se muestran con KaTeX. <code>$\hat y$</code> denota la predicción del modelo y <code>$\epsilon$</code> denota el ruido.
{{% /notice %}}

## Objetivo
- Aprender la recta <code>$\hat y = wx + b$</code> que mejor se ajusta a los datos.
- “Mejor” significa minimizar la suma de errores cuadráticos (SSE):
  <code>$\displaystyle L(w,b) = \sum_{i=1}^n (y_i - (w x_i + b))^2$</code>

## Crear un conjunto de datos sencillo
Generamos una recta con ruido y fijamos la semilla para reproducibilidad.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # opcional para etiquetas en japonés

rng = np.random.RandomState(42)
n_samples = 200

# Recta verdadera (pendiente 0.8, intercepto 0.5) con ruido
X = np.linspace(-10, 10, n_samples)
epsilon = rng.normal(loc=0.0, scale=1.0, size=n_samples)
y = 0.8 * X + 0.5 + epsilon

# Convertir a 2D para scikit-learn: (n_samples, 1)
X_2d = X.reshape(-1, 1)

plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="observaciones", c="orange")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_6_0.png)

{{% notice info %}}
En scikit-learn, las características son siempre un arreglo 2D: filas = muestras, columnas = variables. Use <code>X.reshape(-1, 1)</code> para una sola variable.
{{% /notice %}}

## Inspeccionar el ruido
Veamos la distribución de <code>epsilon</code>.

```python
plt.figure(figsize=(10, 5))
plt.hist(epsilon, bins=30)
plt.xlabel("$\\epsilon$")
plt.ylabel("frecuencia")
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_8_0.png)

## Regresión lineal (mínimos cuadrados) con scikit-learn
Usamos <a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html" target="_blank" rel="noopener">sklearn.linear_model.LinearRegression</a>.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()  # fit_intercept=True por defecto
model.fit(X_2d, y)

print("pendiente w:", model.coef_[0])
print("intercepto b:", model.intercept_)

y_pred = model.predict(X_2d)

# Métricas
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("MSE:", mse)
print("R^2:", r2)

# Gráfico
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="observaciones", c="orange")
plt.plot(X, y_pred, label="recta ajustada", c="C0")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_10_0.png)

{{% notice tip %}}
No es obligatorio escalar para mínimos cuadrados, pero ayuda con problemas multivariados y regularización.
{{% /notice %}}

## Solución en forma cerrada (referencia)
Para <code>$\hat y = wx + b$</code>:

- <code>$\displaystyle w = \frac{\operatorname{Cov}(x,y)}{\operatorname{Var}(x)}$</code>
- <code>$\displaystyle b = \bar y - w\,\bar x$</code>

Verificar con NumPy:

```python
x_mean, y_mean = X.mean(), y.mean()
w_hat = ((X - x_mean) * (y - y_mean)).sum() / ((X - x_mean) ** 2).sum()
b_hat = y_mean - w_hat * x_mean
print(w_hat, b_hat)
```

## Errores comunes
- Formas de arreglos: <code>X</code> debe ser <code>(n_samples, n_features)</code>. Con una variable, use <code>reshape(-1, 1)</code>.
- Forma del objetivo: <code>y</code> puede ser <code>(n_samples,)</code>. <code>(n,1)</code> también funciona; cuide el broadcasting.
- Intercepto: <code>fit_intercept=True</code> por defecto. Si centró <code>X</code> e <code>y</code>, <code>False</code> está bien.
- Reproducibilidad: fije la semilla con <code>np.random.RandomState</code> o <code>np.random.default_rng</code>.

## Más allá (multivariado)
Con varias características, mantenga <code>X</code> como <code>(n_samples, n_features)</code>. Un pipeline combina preprocesamiento y estimador.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_multi = rng.normal(size=(n_samples, 2))
y_multi = 1.0 * X_multi[:, 0] - 2.0 * X_multi[:, 1] + 0.3 + rng.normal(size=n_samples)

pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_multi, y_multi)
```

{{% notice note %}}
El código es para aprendizaje; las figuras están prerenderizadas para el sitio.
{{% /notice %}}

