---
title: "Adaboost (Regresión)"
pre: "2.4.4 "
weight: 4
title_suffix: "Explicación del funcionamiento"
---

{{% youtube "1K-h4YzrnsY" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
```

```python
# NOTA: Modelo creado para observar sample_weight en Adaboost
class DummyRegressor:
    def __init__(self):
        self.model = DecisionTreeRegressor(max_depth=5)
        self.error_vector = None
        self.X_for_plot = None
        self.y_for_plot = None

    def fit(self, X, y):
        self.model.fit(X, y)
        y_pred = self.model.predict(X)

        # El peso se calcula en función del error de regresión
        # Referencia: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_weight_boosting.py#L1130
        self.error_vector = np.abs(y_pred - y)
        self.X_for_plot = X.copy()
        self.y_for_plot = y.copy()
        return self.model

    def predict(self, X, check_input=True):
        return self.model.predict(X)

    def get_params(self, deep=False):
        return {}

    def set_params(self, deep=False):
        return {}

```

## Ajustar un modelo de regresión a los datos de entrenamiento

```python
# Datos de entrenamiento
X = np.linspace(-10, 10, 500)[:, np.newaxis]
y = (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10 + 10 + np.linspace(-2, 2, 500)

# Crear el modelo de regresión
reg = AdaBoostRegressor(
    DummyRegressor(),
    n_estimators=100,
    random_state=100,
    loss="linear",
    learning_rate=0.8,
)
reg.fit(X, y)
y_pred = reg.predict(X)

# Evaluar el ajuste del modelo a los datos de entrenamiento
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", label="Datos de entrenamiento")
plt.plot(X, y_pred, c="r", label="Predicción del modelo final", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Evaluación del ajuste a los datos de entrenamiento")
plt.legend()
plt.show()
```


    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_6_0.png)
    

## Visualización de los pesos de las muestras (cuando `loss='linear'`)
En Adaboost, los pesos de las muestras se determinan en función del error de regresión. Visualizaremos la magnitud de los pesos cuando el parámetro `loss` está configurado como `'linear'`. Observaremos cómo las muestras con mayor peso tienen una probabilidad más alta de ser seleccionadas durante el entrenamiento.


> loss{‘linear’, ‘square’, ‘exponential’}, default=’linear’
> The loss function to use when updating the weights after each boosting iteration.

{{% notice document %}}
[sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor)
{{% /notice %}}


```python
def visualize_weight(reg, X, y, y_pred):
    """Función para visualizar los valores equivalentes a los pesos de las muestras (frecuencia de muestreo)

    Parameters
    ----------
    reg : sklearn.ensemble._weight_boosting
        Modelo de boosting
    X : numpy.ndarray
        Datos de entrenamiento
    y : numpy.ndarray
        Datos objetivo
    y_pred:
        Predicciones del modelo
    """
    assert reg.estimators_ is not None, "len(reg.estimators_) > 0"

    for i, estimators_i in enumerate(reg.estimators_):
        if i % 100 == 0:
            # Contar la cantidad de veces que aparece cada dato en la creación del modelo número i
            weight_dict = {xi: 0 for xi in X.ravel()}
            for xi in estimators_i.X_for_plot.ravel():
                weight_dict[xi] += 1

            # Graficar la frecuencia de aparición con círculos naranjas (más frecuencia, círculos más grandes)
            weight_x_sorted = sorted(weight_dict.items(), key=lambda x: x[0])
            weight_vec = np.array([s * 100 for xi, s in weight_x_sorted])

            # Graficar
            plt.figure(figsize=(10, 5))
            plt.title(f"Visualización de las muestras ponderadas tras el modelo número {i}, loss={reg.loss}")
            plt.scatter(X, y, c="k", marker="x", label="Datos de entrenamiento")
            plt.scatter(
                estimators_i.X_for_plot,
                estimators_i.y_for_plot,
                marker="o",
                alpha=0.2,
                c="orange",
                s=weight_vec,
            )
            plt.plot(X, y_pred, c="r", label="Predicción del modelo final", linewidth=2)
            plt.legend(loc="upper right")
            plt.show()


## Crear el modelo de regresión con loss="linear"
reg = AdaBoostRegressor(
    DummyRegressor(),
    n_estimators=101,
    random_state=100,
    loss="linear",
    learning_rate=1,
)
reg.fit(X, y)
y_pred = reg.predict(X)
visualize_weight(reg, X, y, y_pred)
```

    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_8_0.png)
    



    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_8_1.png)
    



```python
## Crear un modelo de regresión con `loss="square"`
reg = AdaBoostRegressor(
    DummyRegressor(),
    n_estimators=101,
    random_state=100,
    loss="square",
    learning_rate=1,
)
reg.fit(X, y)
y_pred = reg.predict(X)
visualize_weight(reg, X, y, y_pred)
```


    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_9_0.png)
    

    
![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_9_1.png)
    

