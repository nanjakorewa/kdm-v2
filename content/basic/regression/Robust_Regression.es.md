---
title: "Valores atípicos y robustez"
pre: "2.1.3 "
weight: 3
title_suffix: "Manejar los valores atípicos y hacer una regresión lineal robusta en python"
---

<div class="pagetop-box">
    <p>Outlier es un término general para designar un valor anómalo (muy grande o, por el contrario, pequeño) en comparación con otros valores. Los valores anómalos dependen del entorno del problema y de la naturaleza de los datos. En esta página, revisaremos la diferencia de resultados entre la "regresión con error cuadrático" y la "regresión con pérdida de Huber" para datos con valores atípicos.</p>
</div>


```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

## Visualización de la pérdida de Huber


```python
def huber_loss(y_pred: float, y: float, delta=1.0):
    """Pérdida de Huber"""
    huber_1 = 0.5 * (y - y_pred) ** 2
    huber_2 = delta * (np.abs(y - y_pred) - 0.5 * delta)
    return np.where(np.abs(y - y_pred) <= delta, huber_1, huber_2)


delta = 1.5
x_vals = np.arange(-2, 2, 0.01)
y_vals = np.where(
    np.abs(x_vals) <= delta,
    0.5 * np.square(x_vals),
    delta * (np.abs(x_vals) - 0.5 * delta),
)

# plot graph
fig = plt.figure(figsize=(8, 8))
plt.plot(x_vals, x_vals ** 2, "red", label=r"$(y-\hat{y})^2$")  # Error al cuadrado
plt.plot(x_vals, np.abs(x_vals), "orange", label=r"$|y-\hat{y}|$")  # Error absoluto
plt.plot(
    x_vals, huber_loss(x_vals * 2, x_vals), "green", label=r"huber-loss"
)  # pérdida de Huber
plt.axhline(y=0, color="k")
plt.grid(True)
plt.legend()
plt.show()
```


    
![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_5_0.png)
    


## Comparación con el método de mínimos cuadrados

### Preparar los datos para el experimento

Para comparar la regresión con pérdida de Huber y la regresión lineal normal, se incluye intencionadamente un valor atípico en los datos.


```python
N = 30
x1 = np.array([i for i in range(N)])
x2 = np.array([i for i in range(N)])
X = np.array([x1, x2]).T
epsilon = np.array([np.random.random() for i in range(N)])
y = 5 * x1 + 10 * x2 + epsilon * 10
y[5] = 500

plt.figure(figsize=(8, 8))
plt.plot(x1, y, "ko", label="data")
plt.legend()
plt.show()
```


    
![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_7_0.png)
    


### Comparar con mínimos cuadrados, regresión ridge y regresión huber

{{% notice document %}}
[sklearn.linear_model.HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor)
{{% /notice %}}

{{% notice seealso %}}
[Pre-processing method: labeling outliers①](https://k-dm.work/en/prep/numerical/add_label_to_anomaly/)
{{% /notice %}}


```python
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.linear_model import LinearRegression


plt.figure(figsize=(8, 8))
huber = HuberRegressor(alpha=0.0, epsilon=3)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="huber regresión")

ridge = Ridge(alpha=0.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="ridge regresión")

lr = LinearRegression()
lr.fit(X, y)
plt.plot(x1, lr.predict(X), "r-", label="least square regresión")
plt.plot(x1, y, "x")

plt.plot(x1, y, "ko", label="data")
plt.legend()
plt.show()
```


    
![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_9_0.png)
    

