---
title: "Valores atípicos y robustez"
pre: "2.1.3 "
weight: 3
title_suffix: "Tratamiento con regresión Huber"
---

{{% youtube "CrN5Si0379g" %}}


<div class="pagetop-box">
  <p>Los <b>valores atípicos</b> (outliers) son observaciones que se desvían fuertemente del resto. Qué es un outlier depende del problema, la distribución y la escala del objetivo. Aquí comparamos mínimos cuadrados (pérdida cuadrática) con la <b>pérdida de Huber</b> usando datos con un punto extremo.</p>
  </div>

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

---

## 1. Por qué OLS es sensible a outliers

OLS minimiza la suma de cuadrados de los residuos
$$
\text{RSS} = \sum_{i=1}^n (y_i - \hat y_i)^2.
$$
Al <b>elevar al cuadrado</b> los residuos, un solo punto extremo puede dominar la pérdida y <b>arrastrar</b> la recta ajustada hacia el outlier.

---

## 2. Pérdida de Huber: compromiso entre cuadrada y absoluta

La <b>pérdida de Huber</b> usa cuadrática para residuos pequeños y absoluta para grandes. Para <code>r = y - \hat y</code> y umbral <code>\(\delta > 0\)</code>:

$$
\ell_\delta(r) = \begin{cases}
\dfrac{1}{2}r^2, & |r| \le \delta \\
\delta\left(|r| - \dfrac{1}{2}\delta\right), & |r| > \delta.
\end{cases}
$$

La derivada (influencia) es
$$
\psi_\delta(r) = \frac{d}{dr}\ell_\delta(r) = \begin{cases}
r, & |r| \le \delta \\
\delta\,\mathrm{sign}(r), & |r| > \delta,
\end{cases}
$$
por lo que el gradiente se <b>satura</b> ante residuos grandes (outliers).

> Nota: en <code>HuberRegressor</code> de scikit-learn, el umbral es <code>epsilon</code> (corresponde a <code>\(\delta\)</code>).

---

## 3. Visualizar las formas de las pérdidas

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
plt.title("Cuadrática vs absoluta vs Huber")
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_5_0.png)

---

## 4. ¿Qué pasa con un outlier? (datos)

Creamos un problema lineal con 2 variables y <b>inyectamos un outlier extremo</b> en <code>y</code>.

```python
np.random.seed(42)

N = 30
x1 = np.arange(N)
x2 = np.arange(N)
X = np.c_[x1, x2]                      # (N, 2)
epsilon = np.random.rand(N)            # ruido en [0, 1)
y = 5 * x1 + 10 * x2 + epsilon * 10

y[5] = 500  # un outlier muy grande

plt.figure(figsize=(8, 6))
plt.plot(x1, y, "ko", label="datos")
plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Conjunto con un outlier")
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_7_0.png)

---

## 5. Comparar OLS vs Ridge vs Huber

- <b>OLS</b>: muy sensible a outliers.  
- <b>Ridge</b> (L2): encoge coeficientes; algo más estable, pero aún afectado.  
- <b>Huber</b>: satura la influencia de outliers; la recta se arrastra menos.

{{% notice document %}}
- [HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)
{{% /notice %}}

{{% notice seealso %}}
[Enfoque de preprocesamiento: etiquetar anomalías (JP)](https://k-dm.work/ja/prep/numerical/add_label_to_anomaly/)
{{% /notice %}}

```python
from sklearn.linear_model import HuberRegressor, Ridge, LinearRegression

plt.figure(figsize=(8, 6))

# Huber: epsilon=3 para reducir la influencia de outliers
huber = HuberRegressor(alpha=0.0, epsilon=3.0)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="Huber")

# Ridge (L2). Con alpha≈0, se parece a OLS
ridge = Ridge(alpha=1.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="Ridge (α=1.0)")

# OLS
lr = LinearRegression()
lr.fit(X, y)
plt.plot(x1, lr.predict(X), "r-", label="OLS")

# datos brutos
plt.plot(x1, y, "kx", alpha=0.7)

plt.xlabel("$x_1$")
plt.ylabel("$y$")
plt.legend()
plt.title("Efecto de un outlier en las rectas ajustadas")
plt.grid(alpha=0.3)
plt.show()
```

![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_9_0.png)

Interpretación:
- OLS (rojo) es fuertemente arrastrado por el outlier.
- Ridge (naranja) mitiga un poco pero sigue afectado.
- Huber (verde) reduce la influencia del outlier y sigue mejor la tendencia global.

---

## 6. Parámetros: epsilon y alpha

- <code>epsilon</code> (umbral <code>\(\delta\)</code>):
  - Mayor → más cercano a OLS; menor → más cercano a pérdida absoluta.
  - Depende de la escala de residuos; estandarice o use escalado robusto.
- <code>alpha</code> (penalización L2):
  - Estabiliza coeficientes; útil con colinealidad.

Sensibilidad a <code>epsilon</code>:

```python
from sklearn.metrics import mean_squared_error

for eps in [1.2, 1.5, 2.0, 3.0]:
    h = HuberRegressor(alpha=0.0, epsilon=eps).fit(X, y)
    mse = mean_squared_error(y, h.predict(X))
    print(f"epsilon={eps:>3}: MSE={mse:.3f}")
```

---

## 7. Notas prácticas

- <b>Escalado</b>: si las escalas de variables/objetivo difieren, cambia el significado de <code>epsilon</code>; estandarice o use escalado robusto.
- <b>Puntos de alto apalancamiento</b>: Huber es robusta a outliers verticales en <code>y</code>, no necesariamente a extremos en <code>X</code>.
- <b>Elegir umbrales</b>: ajuste <code>epsilon</code> y <code>alpha</code> (por ejemplo, con <code>GridSearchCV</code>).
- <b>Evalúe con CV</b>: no se guíe solo por el ajuste en entrenamiento.

---

## 8. Resumen

- OLS es sensible a outliers; el ajuste puede ser arrastrado.
- Huber usa cuadrática para errores pequeños y absoluta para grandes, <b>saturando gradientes</b> ante outliers.
- Ajuste <code>epsilon</code> y <code>alpha</code> para balancear robustez y ajuste.
- Cuidado con puntos de apalancamiento; combine con inspección y preprocesamiento si es necesario.

---

