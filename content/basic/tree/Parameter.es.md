---
title: "Parámetros de los Árboles de Decisión"
pre: "2.3.3 "
weight: 3
title_suffix: "Comprender cómo funcionan"
---

{{% youtube "AOEtom_l3Wk" %}}

<div class="pagetop-box">
    <p>Los árboles de decisión tienen varios parámetros, y los resultados pueden variar según cómo se especifiquen. En esta página, exploraremos visualmente cómo funciona cada uno de estos parámetros.</p>
</div>

- `max_depth` especifica la profundidad máxima del árbol.
- `min_samples_split` indica el número mínimo de datos necesarios para crear una bifurcación.
- `min_samples_leaf` define el número mínimo de datos necesarios para formar una hoja.
- `max_leaf_nodes` establece un límite en el número de hojas del árbol.
- `ccp_alpha` es un parámetro para podar el árbol de decisión considerando su complejidad.
- `class_weight` permite asignar pesos a las clases en problemas de clasificación.

{{% notice document %}}
- [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)
- [parrt/dtreeviz](https://github.com/parrt/dtreeviz)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
from dtreeviz.trees import dtreeviz, rtreeviz_bivar_3D
```

## Ajustando un árbol de decisión a datos simples

```python
# Datos de ejemplo
X, y = make_regression(n_samples=100, n_features=2, random_state=11)

# Entrenamiento del árbol de decisión
dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X, y)

# Visualización
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="MPG",
    elev=40,
    azim=120,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_5_0.png)
    


## Experimentando con diferentes parámetros del árbol de decisión
Vamos a observar cómo se comporta un árbol de decisión al modificar sus parámetros, utilizando un conjunto de datos con una estructura más compleja. Primero, examinaremos un árbol de decisión donde el único parámetro modificado es `max_depth=3`, mientras que los demás permanecen con sus valores predeterminados.

```python
# Datos de ejemplo
X, y = make_regression(
    n_samples=500, n_features=2, effective_rank=4, noise=0.1, random_state=1
)
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Entrenamiento del árbol de decisión
dt = DecisionTreeRegressor(max_depth=3, random_state=117117)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_7_0.png)
    



    
![png](/images/basic/tree/Parameter_files/Parameter_7_1.png)
    


### max_depth = 10
Cuando el valor de `max_depth` es alto, se genera un árbol más profundo y complejo. Esto permite representar reglas más intrincadas, pero también puede llevar a un sobreajuste si los datos son escasos.


```python
dt = DecisionTreeRegressor(max_depth=10, random_state=117117)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_9_0.png)
    


### max-depth=5


```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_11_0.png)
    


### min_samples_split=60

El parámetro `min_samples_split` especifica el número mínimo de datos necesarios para crear una bifurcación. 

- Si se reduce el valor de `min_samples_split`, el árbol puede generar reglas más detalladas y complejas. 
- Si se incrementa, es más probable evitar el sobreajuste.


```python
dt = DecisionTreeRegressor(max_depth=5, min_samples_split=60, random_state=117117)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_13_0.png)
    


### ccp_alpha=0.4
El parámetro `ccp_alpha` penaliza la complejidad del árbol. Al establecer un valor para `ccp_alpha`, cuanto mayor sea este valor, más simple será el árbol resultante.

```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117, ccp_alpha=0.4)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_15_0.png)
    


### max_leaf_nodes=5
El parámetro `max_leaf_nodes` especifica el número máximo de hojas que puede tener el árbol final. Se puede observar que el número de hojas coincide con el valor especificado en `max_leaf_nodes`.

```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117, max_leaf_nodes=5)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_17_0.png)
    

## En presencia de valores atípicos

El parámetro `criterion` especifica el criterio que se utiliza para dividir los nodos. Vamos a observar cómo cambia el árbol al utilizar `criterion="squared_error"` en un conjunto de datos con valores atípicos. 

- `squared_error` penaliza fuertemente los valores atípicos y es probable que esto afecte las divisiones del árbol.
- En comparación, `absolute_error` es menos sensible a los valores atípicos.

En este ejemplo, se modifican algunos datos para simular valores atípicos multiplicándolos por 5.

```python
# Introducción de valores atípicos: multiplicar algunos datos por 5
X, y = make_regression(n_samples=100, n_features=2, random_state=11)
y[1:20] = y[1:20] * 5
```


```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117, criterion="absolute_error")
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_20_0.png)
    



```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117, criterion="squared_error")

dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_21_0.png)
    

