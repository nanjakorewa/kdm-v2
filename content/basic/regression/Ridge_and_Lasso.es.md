---
title: "Regresión Ridge y Lasso"
pre: "2.1.2 "
weight: 2
searchtitle: "Ejecutar la regresión Ridge y Lasso en Python"
---

<div class="pagetop-box">
    <p>La regresión Ridge y Lasso son modelos de regresión lineal diseñados para evitar el sobreaprendizaje de los modelos mediante la penalización de los coeficientes del modelo. Estas técnicas que intentan evitar el sobreaprendizaje y aumentar la generalizabilidad se denominan regularización. En esta página, utilizaremos python para ejecutar la regresión por mínimos cuadrados, por crestas y por lazos y comparar sus coeficientes.</p>
</div>

## Importar los módulos necesarios

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

## Generar datos de regresión para los experimentos

{{% notice document %}}
[sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)
{{% /notice %}}

Generar artificialmente datos de regresión utilizando [sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html). Especifique dos características (`n_informativa` = 2) como necesarias para la predicción. De lo contrario, son características redundantes que no son útiles para la predicción.


```python
n_features = 5
n_informative = 2
X, y = make_regression(
    n_samples=500,
    n_features=n_features,
    n_informative=n_informative,
    noise=0.5,
    random_state=777,
)
X = np.concatenate([X, np.log(X + 100)], 1)  # Añadir features redundantes
y_mean = y.mean(keepdims=True)
y_std = np.std(y, keepdims=True)
y = (y - y_mean) / y_std
```

## Comparar los modelos de mínimos cuadrados, regresión ridge y regresión lasso

Entrenemos cada modelo con los mismos datos y veamos las diferencias.
Normalmente, el coeficiente alfa del término de regularización debería cambiarse, pero en este caso, es fijo.

{{% notice document %}}
- [sklearn.pipeline.make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)
- [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)
{{% /notice %}}


```python
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
rid_r = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2)).fit(X, y)
las_r = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.1)).fit(X, y)
```

## Compara los valores de los coeficientes de cada modelo

Traza el valor absoluto de cada coeficiente. Se puede ver en el gráfico que para la regresión lineal, los coeficientes casi nunca son cero. También podemos ver que la regresión Lasso tiene coeficientes = 0 para todas las características excepto las necesarias para la predicción.

Dado que había dos características útiles (`n_informativa = 2`), podemos ver que la regresión lineal tiene coeficientes incondicionales incluso en las características que no son útiles, mientras que la regresión Lasso-Ridge no pudo reducirse a dos características. Aun así, dio lugar a que muchas características no deseadas tuvieran coeficientes de cero.

```python
feat_index = np.array([i for i in range(X.shape[1])])

plt.figure(figsize=(12, 4))
plt.bar(
    feat_index - 0.2,
    np.abs(lin_r.steps[1][1].coef_),
    width=0.2,
    label="Linear regresión",
)
plt.bar(feat_index, np.abs(rid_r.steps[1][1].coef_), width=0.2, label="Ridge regresión")
plt.bar(
    feat_index + 0.2,
    np.abs(las_r.steps[1][1].coef_),
    width=0.2,
    label="Lasso regresión",
)

plt.xlabel(r"$\beta$")
plt.ylabel(r"$|\beta|$")
plt.grid()
plt.legend()
```

    
![png](/images/basic/regression/02_Ridge_and_Lasso_files/02_Ridge_and_Lasso_10_1.png)
    

