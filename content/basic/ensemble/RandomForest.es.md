---
title: "Bosque Aleatorio"
pre: "2.4.1"
weight: 1
title_suffix: "Explicación del Funcionamiento"
---

{{% youtube "ewvjQMj8nA8" %}}

<div class="pagetop-box">
    <p><b>Bosque Aleatorio</b> es un algoritmo de aprendizaje en conjunto que mejora la capacidad de generalización y la precisión de las predicciones al combinar árboles de decisión creados utilizando características seleccionadas al azar.</p>
    <p>En esta página, ejecutaremos un Bosque Aleatorio y exploraremos el rendimiento y los detalles de los árboles de decisión individuales dentro del modelo.</p>
</div>

{{% notice document %}}
[sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn-ensemble-randomforestclassifier)
{{% /notice %}}



```python
import numpy as np
import matplotlib.pyplot as plt
```

## Entrenamiento de Bosque Aleatorio
{{% notice seealso %}}
Sobre el ROC-AUC, puedes encontrar una explicación sobre cómo trazarlo en [ROC-AUC](http://localhost:1313/ja/eval/classification/roc-auc/).
{{% /notice %}}


{{% notice document %}}
- [sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score)
- [sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split)
{{% /notice %}}


```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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

model = RandomForestClassifier(
    n_estimators=50, max_depth=3, random_state=777, bootstrap=True, oob_score=True
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rf_score = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC en los datos de prueba = {rf_score}")
```

    ROC-AUC en los datos de prueba = 0.814573097628059


## Verificar el rendimiento de cada árbol incluido en el Bosque Aleatorio

```python
import japanize_matplotlib

estimator_scores = []
for i in range(10):
    estimator = model.estimators_[i]
    estimator_pred = estimator.predict(X_test)
    estimator_scores.append(roc_auc_score(y_test, estimator_pred))

plt.figure(figsize=(10, 4))
bar_index = [i for i in range(len(estimator_scores))]
plt.bar(bar_index, estimator_scores)
plt.bar([10], rf_score)
plt.xticks(bar_index + [10], bar_index + ["RF"])
plt.xlabel("Índice del Árbol")
plt.ylabel("ROC-AUC")
plt.show()
```


    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_6_0.png)
    


### Importancia de las Características
#### Importancia basada en la Impureza

El siguiente gráfico muestra la importancia de las características calculada en función de la reducción de impureza en el modelo de Bosque Aleatorio. Esto nos permite identificar cuáles son las características más relevantes para las predicciones.

```python
plt.figure(figsize=(10, 4))
feature_index = [i for i in range(n_features)]
plt.bar(feature_index, model.feature_importances_)
plt.xlabel("Índice de la Característica")
plt.ylabel("Importancia de la Característica")
plt.show()
```


    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_8_0.png)
    


### permutation importance
{{% notice document %}}
[permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance)
{{% /notice %}}


```python
from sklearn.inspection import permutation_importance

p_imp = permutation_importance(
    model, X_train, y_train, n_repeats=10, random_state=77
).importances_mean

plt.figure(figsize=(10, 4))
plt.bar(feature_index, p_imp)
plt.xlabel("Índice de la Característica")
plt.ylabel("Importancia de la Característica")
plt.show()
```


    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_10_0.png)
    


## Exportar cada árbol incluido en el Bosque Aleatorio


```python
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from IPython.display import display

for i in range(10):
    try:
        estimator = model.estimators_[i]
        export_graphviz(
            estimator,
            out_file=f"tree{i}.dot",
            feature_names=[f"x{i}" for i in range(n_features)],
            class_names=["A", "B"],
            proportion=True,
            filled=True,
        )

        call(["dot", "-Tpng", f"tree{i}.dot", "-o", f"tree{i}.png", "-Gdpi=500"])
        display(Image(filename=f"tree{i}.png"))
    except KeyboardInterrupt:
        pass
```


    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_0.png)
    



    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_1.png)
    



    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_2.png)
    



    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_3.png)
    



    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_4.png)
    



    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_5.png)
    



    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_6.png)
    



    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_7.png)
    



    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_8.png)
    



    
![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_9.png)
    


### Puntuación OOB (Out-of-Bag)
La validación mediante OOB permite verificar que los resultados obtenidos son consistentes con los obtenidos en los datos de prueba. Esto asegura que el modelo generaliza bien sin sobreajustarse.

En esta sección, compararemos la **precisión obtenida con OOB** y la **precisión en los datos de prueba** al variar los parámetros como la semilla aleatoria y la profundidad de los árboles.


```python
from sklearn.metrics import accuracy_score

for i in range(10):
    model_i = RandomForestClassifier(
        n_estimators=50,
        max_depth=3 + i % 2,
        random_state=i,
        bootstrap=True,
        oob_score=True,
    )
    model_i.fit(X_train, y_train)
    y_pred = model_i.predict(X_test)
    oob_score = model_i.oob_score_
    test_score = accuracy_score(y_test, y_pred)
    print(f"Validación OOB={oob_score} Validación en Datos de Prueba＝{test_score}")
```

| Validación OOB | Validación en Datos de Prueba |
|-----------------|-------------------------------|
| 0.7869          | 0.8121                        |
| 0.8101          | 0.8364                        |
| 0.7887          | 0.8024                        |
| 0.8161          | 0.8315                        |
| 0.7910          | 0.8073                        |
| 0.8101          | 0.8339                        |
| 0.7815          | 0.8133                        |
| 0.8060          | 0.8339                        |
| 0.7833          | 0.7952                        |
| 0.8084          | 0.8388                        |

