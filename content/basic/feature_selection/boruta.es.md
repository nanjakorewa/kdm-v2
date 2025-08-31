---
title: "Boruta"
pre: "2.7.1 "
weight: 1
searchtitle: "Ejecutar selección de características con Boruta"
---

## Boruta
Vamos a seleccionar características utilizando Boruta. El código en este bloque es un ejemplo directo del uso de Boruta.

Referencia:  
`Kursa, Miron B., and Witold R. Rudnicki. "Feature selection with the Boruta package." Journal of statistical software 36 (2010): 1-13.`

{{% youtube "xOkKnsqhUgw" %}}


```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


# FIXME
np.random.seed(777)
np.int = int
np.float = float
np.bool = bool
```

```python
# Cargar los datos X e y
X = pd.read_csv("examples/test_X.csv", index_col=0).values
y = pd.read_csv("examples/test_y.csv", header=None, index_col=0).values
y = y.ravel()

# Definir un clasificador Random Forest utilizando todos los núcleos disponibles
# y muestreo proporcional a las etiquetas de y
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)

# Definir el método de selección de características Boruta
feat_selector = BorutaPy(rf, n_estimators="auto", verbose=2, random_state=1)

# Encontrar todas las características relevantes - se deberían seleccionar 5 características
feat_selector.fit(X, y)

# Comprobar las características seleccionadas - las primeras 5 características deberían ser seleccionadas
feat_selector.support_

# Comprobar el ranking de las características
feat_selector.ranking_

# Aplicar transform() a X para reducirlo a las características seleccionadas
X_filtered = feat_selector.transform(X)
```

    Iteration: 	1 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	2 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	3 / 100
    Confirmed: 	0
    Tentative: 	10
    
    BorutaPy finished running.
    
    Iteration: 	9 / 100
    Confirmed: 	5
    Tentative: 	0
    Rejected: 	5


## Experimentos con Datos Artificiales

```python
from sklearn.datasets import make_classification
from xgboost import XGBClassifier


def fs_by_boruta(model, X, y):
    feat_selector = BorutaPy(model, n_estimators="auto", verbose=2, random_state=1)
    feat_selector.fit(X, y)
    X_filtered = feat_selector.transform(X)

    if X.shape[1] == X_filtered.shape[1]:
        print("No se encontraron características innecesarias")
    else:
        print("Se eliminaron características innecesarias")
        print(f"{X.shape[1]} --> {X_filtered.shape[1]}")

    return X_filtered
```

### No se elimina ninguna característica si todas son necesarias

```python
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=10,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
model = XGBClassifier(max_depth=4)
fs_by_boruta(model, X, y)
```

    Iteration: 	1 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	2 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	3 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	4 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	5 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	6 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	7 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	8 / 100
    Confirmed: 	10
    Tentative: 	0
    Rejected: 	0
    
    
    BorutaPy finished running.
    
    Iteration: 	9 / 100
    Confirmed: 	10
    Tentative: 	0
    Rejected: 	0
    不用な特徴は見つかりませんでした
    




    array([[ 0.38760058, -0.4398061 ,  1.0103586 , ..., -2.11674403,
            -3.59368321, -0.43265007],
           [-2.18745511, -2.45701675,  1.99758878, ...,  1.16128752,
            -1.92766999,  3.18705784],
           [ 3.98304273,  0.06250274, -1.31136045, ...,  1.45498409,
            -4.17178063, -2.21695578],
           ...,
           [-0.44293666,  3.25707522, -0.50633794, ..., -0.72410483,
            -1.5420989 ,  0.75991518],
           [-1.12641706, -0.48636924,  0.92918889, ..., -1.01001779,
            -2.69280573, -3.47050681],
           [-2.3936814 ,  1.44048113,  1.95832126, ..., -5.15104933,
            -1.02766442,  1.4853396 ]])



### Se eliminan las características innecesarias
Crearemos un conjunto de datos con 100 características, de las cuales solo 10 son útiles, para observar cuántas características innecesarias pueden ser eliminadas.

La documentación de [sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html) especifica:

> Sin mezclar, las características en `X` se apilan horizontalmente en el siguiente orden: las `n_informative` características principales, seguidas por `n_redundant` combinaciones lineales de las características informativas, seguidas por `n_repeated` duplicados, seleccionados al azar con reemplazo de las características informativas y redundantes. Las características restantes se rellenan con ruido aleatorio. Por lo tanto, sin mezclar, todas las características útiles están contenidas en las columnas `X[:, :n_informative + n_redundant + n_repeated]`.

Por lo tanto, verificaremos que las primeras 10 columnas, que corresponden a las características útiles, no sean eliminadas durante el proceso.

```python
X, y = make_classification(
    n_samples=2000,
    n_features=100,
    n_informative=10,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
model = XGBClassifier(max_depth=5)

X_b = fs_by_boruta(model, X, y)
```

    Iteration: 	1 / 100
    Confirmed: 	0
    Tentative: 	100
    Rejected: 	0
    Iteration: 	2 / 100
...
    
    BorutaPy finished running.
    
    Iteration: 	100 / 100
    Confirmed: 	10
    Tentative: 	1
    Rejected: 	88
    不用な特徴を削除しました
    100 --> 10
    

#### Verificar que las características útiles permanecen en los datos después del filtrado
Si el filtrado funciona como se espera, las primeras 10 columnas, que corresponden a características útiles, deberían permanecer intactas en los datos procesados.

```python
X[:, :10] == X_b[:, :10]
```




    array([[ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True],
           ...,
           [ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True]])


