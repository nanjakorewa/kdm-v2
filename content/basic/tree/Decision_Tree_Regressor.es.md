---
title: "Árbol de decisión (regresión)"
pre: "2.3.2 "
weight: 2
searchtitle: "Visualización de árboles de decisión (regresión) en python"
---

<div class="pagetop-box">
    <p>Un árbol de decisión (regresión) es un tipo de modelo que utiliza una combinación de reglas. La colección de reglas se representa mediante un gráfico en forma de árbol (estructura de árbol), que es fácil de interpretar. Esta página ejecuta una regresión de un árbol de decisión y además visualiza el árbol resultante.</p>
</div>

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from dtreeviz.trees import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dtreeviz.trees import dtreeviz
```

{{% notice document %}}
[dtreeviz : Decision Tree Visualization](https://github.com/parrt/dtreeviz)
{{% /notice %}}

## Generar datos de muestra para los árboles de decisión

```python
X, y = make_regression(n_samples=100, n_features=2, random_state=777)
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```


    
![png](/images/basic/tree/Decision_Tree_Regressor_files/Decision_Tree_Regressor_6_0.png)
    


## Compruebe cómo se ramifica el árbol de regresión


```python
tree = DecisionTreeRegressor(max_depth=3, random_state=117117)
model = tree.fit(X, y)
viz = dtreeviz(tree, X, y, target_name="y")
viz.save("./regression_tree.svg")
```

### Visualizar la ramificación de un árbol de regresión


```python
from IPython.display import SVG

SVG(filename="regression_tree.svg")
```




    
![svg](/images/basic/tree/Decision_Tree_Regressor_files/Decision_Tree_Regressor_10_0.svg)
    


