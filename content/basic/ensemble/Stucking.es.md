---
title: "Stacking | Combinar modelos para un ensamble más potente"
linkTitle: "Stacking"
seo_title: "Stacking | Combinar modelos para un ensamble más potente"
pre: "2.4.2"
weight: 2
title_suffix: "Explicación del Funcionamiento"
---

{{% youtube "U5F1PYw_P3E" %}}

<div class="pagetop-box">
    <p><b>Stacking</b> se refiere a un modelo que repite el proceso de "crear múltiples modelos de predicción y usar sus salidas como entrada para otro modelo de predicción". En esta página, implementaremos el stacking y analizaremos qué modelos de la primera capa resultaron más efectivos.</p>
</div>

```python
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_graphviz
from subprocess import call
```

## Creación de Datos para Experimentos
En esta sección, se generarán datos con 20 características para realizar los experimentos.


```python
# Crear datos con 20 características
n_features = 20
X, y = make_classification(
    n_samples=2500,
    n_features=n_features,
    n_informative=10,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=4,
    random_state=777,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=777
)
```

### Representación Gráfica de los Datos Según Varias Características
El objetivo de esta sección es visualizar los datos generados utilizando múltiples características para verificar que no pueden ser clasificados fácilmente mediante reglas simples.


```python
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.scatter(X[:, 2], X[:, 7], c=y)
plt.xlabel("x2")
plt.ylabel("x7")
plt.subplot(2, 2, 2)
plt.scatter(X[:, 4], X[:, 9], c=y)
plt.xlabel("x4")
plt.ylabel("x9")
plt.subplot(2, 2, 3)
plt.scatter(X[:, 5], X[:, 1], c=y)
plt.xlabel("x5")
plt.ylabel("x1")
plt.subplot(2, 2, 4)
plt.scatter(X[:, 1], X[:, 3], c=y)
plt.xlabel("x1")
plt.ylabel("x3")
plt.show()
```


    
![png](/images/basic/ensemble/Stucking_files/Stucking_6_0.png)
    


## Comparación entre Stacking y Bosque Aleatorio

### Clasificación Utilizando un Bosque Aleatorio


```python
model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=777)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rf_score = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC = {rf_score}")
```

    ROC-AUC = 0.855797033310609


### Stacking Utilizando Múltiples Árboles

En esta sección, implementaremos el método de Stacking utilizando únicamente `DecisionTreeClassifier`. Esto nos permitirá observar que la precisión no mejora significativamente cuando se utiliza un solo tipo de modelo.


{{% notice document %}}
[sklearn.ensemble.StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
{{% /notice %}}


### Comparación de ROC-AUC: Stacking vs. Bosque Aleatorio

El siguiente código implementa un modelo de Stacking compuesto únicamente por `DecisionTreeClassifier` con diferentes profundidades como modelos base, seguido de otro `DecisionTreeClassifier` como modelo final. Luego, compara su rendimiento (ROC-AUC) con el de un modelo de Bosque Aleatorio previamente entrenado.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score

# Modelos base para la primera capa del Stacking
estimators = [
    ("dt1", DecisionTreeClassifier(max_depth=3, random_state=777)),
    ("dt2", DecisionTreeClassifier(max_depth=4, random_state=777)),
    ("dt3", DecisionTreeClassifier(max_depth=5, random_state=777)),
    ("dt4", DecisionTreeClassifier(max_depth=6, random_state=777)),
]

# Número de modelos base en el Stacking
n_estimators = len(estimators)

# Modelo final del Stacking
final_estimator = DecisionTreeClassifier(max_depth=3, random_state=777)

# Crear y entrenar el modelo de Stacking
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train, y_train)

# Evaluación en los datos de prueba
y_pred = clf.predict(X_test)
clf_score = roc_auc_score(y_test, y_pred)

print("ROC-AUC")
print(f"Stacking con Árboles de Decisión = {clf_score:.4f}, Bosque Aleatorio = {rf_score:.4f}")
```

    ROC-AUC
    Stacking con Árboles de Decisión＝0.7359716965608031, Bosque Aleatorio＝0.855797033310609


### Visualización de los Árboles Utilizados en el Stacking


```python
export_graphviz(
    clf.final_estimator_,
    out_file="tree_final_estimator.dot",
    class_names=["A", "B"],
    feature_names=[e[0] for e in estimators],
    proportion=True,
    filled=True,
)

call(
    [
        "dot",
        "-Tpng",
        "tree_final_estimator.dot",
        "-o",
        f"tree_final_estimator.png",
        "-Gdpi=200",
    ]
)
display(Image(filename="tree_final_estimator.png"))
```


    
![png](/images/basic/ensemble/Stucking_files/Stucking_13_0.png)
    


### Importancia de las Características en los Árboles Utilizados en el Stacking

En esta sección, evaluaremos la importancia de las características en los árboles de decisión utilizados dentro del modelo de Stacking. Esto nos permitirá observar si algún árbol domina el proceso de predicción.



```python
plt.figure(figsize=(6, 3))
plot_index = [i for i in range(n_estimators)]
plt.bar(plot_index, clf.final_estimator_.feature_importances_)
plt.xticks(plot_index, [e[0] for e in estimators])
plt.xlabel("model name")
plt.ylabel("feature-importance")
plt.show()
```


    
![png](/images/basic/ensemble/Stucking_files/Stucking_15_0.png)
    


### Evaluación del Rendimiento de Cada Árbol en la Primera Capa del Stacking

```python
scores = []
for clf_estim in clf.estimators_:
    print("====")
    y_pred = clf_estim.predict(X_test)
    scr = roc_auc_score(y_test, y_pred)
    scores.append(scr)
    print(clf_estim)
    print(scr)

n_estimators = len(estimators)
plot_index = [i for i in range(n_estimators)]

plt.figure(figsize=(8, 4))
plt.bar(plot_index, scores)
plt.xticks(plot_index, [e[0] for e in estimators])
plt.xlabel("model name")
plt.ylabel("roc-auc")
plt.show()
```

    ====
    DecisionTreeClassifier(max_depth=3, random_state=777)
    0.7660117774277722
    ====
    DecisionTreeClassifier(max_depth=4, random_state=777)
    0.7744128916993818
    ====
    DecisionTreeClassifier(max_depth=5, random_state=777)
    0.8000158677919086
    ====
    DecisionTreeClassifier(max_depth=6, random_state=777)
    0.8084639977432473



    
![png](/images/basic/ensemble/Stucking_files/Stucking_17_1.png)
    


{{% notice tip %}}
・[Stacked Generalization (Stacking)](http://machine-learning.martinsewell.com/ensembles/stacking/)<br/>
・[MLWave/Kaggle-Ensemble-Guide](https://github.com/MLWave/Kaggle-Ensemble-Guide)
{{% /notice %}}
