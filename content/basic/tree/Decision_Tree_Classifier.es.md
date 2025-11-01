---
title: "Clasificador con árboles de decisión | Dividir datos según la ganancia de información"
linkTitle: "Árbol de decisión"
seo_title: "Clasificador con árboles de decisión | Dividir datos según la ganancia de información"
pre: "2.3.1 "
weight: 1
searchtitle: "Ejecución de árboles de decisión (clasificación) en python"
---

<div class="pagetop-box">
    <p>Un árbol de decisión (clasificación) es un tipo de modelo que utiliza una combinación de reglas para clasificar. La colección de reglas se representa mediante un gráfico en forma de árbol (estructura de árbol), que es fácil de interpretar. Esta página realiza la clasificación del árbol de decisión y además visualiza el árbol.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

## Generar datos de muestra

Generar datos de muestra para la clasificación de 2 clases.


```python
n_classes = 2
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_classes=n_classes,
    n_clusters_per_class=1,
)
```

## Crear un árbol de decisión

Entrene el modelo con `DecisionTreeClassifier(criterion="gini").fit(X, y)` para visualizar los límites de decisión del árbol creado.
El `criterio="gini"` es una opción para especificar un indicador para determinar la ramificación.

{{% notice document %}}
[sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
{{% /notice %}}


```python
# Clasificador de árbol de decisión
clf = DecisionTreeClassifier(criterion="gini").fit(X, y)

# Conjunto de datos para el mapa de colores del límite de decisión
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualizar los límites de la decisión
plt.figure(figsize=(8, 8))
plt.tight_layout()
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)
plt.xlabel("x1")
plt.ylabel("x2")

for i, color, label_name in zip(range(n_classes), ["r", "b"], ["A", "B"]):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=label_name, cmap=plt.cm.Pastel1)

plt.legend()
plt.show()
```


    
![png](/images/basic/tree/Decision_Tree_Classifier_files/Decision_Tree_Classifier_7_0.png)
    


## La estructura del árbol de decisión se presenta como una imagen

{{% notice document %}}
[sklearn.tree.plot_tree — scikit-learn 1.0.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
{{% /notice %}}


```python
plt.figure()
clf = DecisionTreeClassifier(criterion="gini").fit(X, y)
plt.figure(figsize=(12, 12))
plot_tree(clf, filled=True)
plt.show()
```

    
![png](/images/basic/tree/Decision_Tree_Classifier_files/Decision_Tree_Classifier_9_1.png)
    

