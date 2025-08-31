---
title: "Método de mínimos cuadrados"
pre: "2.1.1 "
weight: 1
searchtitle: "Ejecutar la regresión por mínimos cuadrados en Python"
---

<div class="pagetop-box">
    <p>El método de los mínimos cuadrados se refiere a encontrar los coeficientes de una función para minimizar la suma de los cuadrados de los residuos cuando se ajusta una función a una colección de pares de números $(x_i, y_i)$ con el fin de conocer su relación. En esta página, trataremos de realizar el método de mínimos cuadrados sobre datos de muestra utilizando scikit-learn.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

{{% notice tip %}}
Se importa `japanize_matplotlib` para mostrar el japonés en el gráfico.
{{% /notice %}}

## Generar datos de regresión para los experimentos

Utilice `np.linspace` para crear datos. Crea una lista de valores igualmente espaciados entre los valores que usted especifica. El siguiente código crea 500 datos de muestra para la regresión lineal.

```python
# Datos en un conjunto de entrenamiento
n_samples = 500
X = np.linspace(-10, 10, n_samples)[:, np.newaxis]
epsolon = np.random.normal(size=n_samples)
y = np.linspace(-2, 2, n_samples) + epsolon

# Visualizar las líneas
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="Target", c="orange")
plt.xlabel("$x_1$")
plt.xlabel("$y$")
plt.legend()
plt.show()
```


![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_6_0.png)


## Comprobar el ruido en y

Sobre `y = np.linspace(-2, 2, n_samples) + epsolon`, trazo un histograma para `epsolon`.
Confirma que el ruido con una distribución cercana a la distribución normal está en la variable objetivo.

```python
plt.figure(figsize=(10, 5))
plt.hist(epsolon)
plt.xlabel("$\epsilon$")
plt.ylabel("#data")
plt.show()
```


    
![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_8_0.png)
    


## Ajustar una línea recta por el método de los mínimos cuadrados

{{% notice document %}}
[sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
{{% /notice %}}


```python
# Entrenar su modelo
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
y_pred = lin_r.predict(X)

# Comprueba la línea ajustada por regresión lineal.
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="target", c="orange")
plt.plot(X, y_pred, label="Línea recta ajustada por regresión lineal")
plt.xlabel("$x_1$")
plt.xlabel("$y$")
plt.legend()
plt.show()
```


    
![png](/images/basic/regression/01_Linear_Regression_files/01_Linear_Regression_10_0.png)
    

