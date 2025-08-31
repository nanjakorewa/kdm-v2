---
title: "ROC-AUC"
pre: "4.3.1 "
weight: 1
searchtitle: "trazar el gráfico ROC-AUC en python"
---

El área bajo la curva ROC se denomina AUC (Area Under the Curve) y se utiliza como índice de evaluación de los modelos de clasificación; el mejor es cuando el AUC es 1, y 0,5 para los modelos aleatorios y totalmente inválidos.

- El ROC-AUC es un ejemplo típico de índice de evaluación de clasificación binaria
- 1 es el mejor, 0,5 se acerca a una predicción totalmente aleatoria
- Por debajo de 0,5 puede ser cuando la predicción es lo contrario de la respuesta correcta
- El trazado de la curva ROC puede ayudar a determinar cuál debe ser el umbral de clasificación


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
```

## Plot ROC Curve
{{% notice document %}}
[sklearn.metrics.roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
{{% /notice %}}

### Function to plot ROC Curve

```python
def plot_roc_curve(test_y, pred_y):
    """Trazar la curva ROC a partir de las respuestas correctas y las predicciones

    Args:
        test_y (ndarray of shape (n_samples,)): y
        pred_y (ndarray of shape (n_samples,)): Valor previsto para y
    """
    # Tasa de falsos positivos, tasa de verdaderos positivos
    fprs, tprs, thresholds = roc_curve(test_y, pred_y)

    # gráfico ROC-AUC
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle="-", c="k", alpha=0.2, label="ROC-AUC=0.5")
    plt.plot(fprs, tprs, color="orange", label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Rellene el área correspondiente a la puntuación ROC-AUC
    y_zeros = [0 for _ in tprs]
    plt.fill_between(fprs, y_zeros, tprs, color="orange", alpha=0.3, label="ROC-AUC")
    plt.legend()
    plt.show()
```

### Create a model and plot ROC Curve against sample data


```python
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    n_informative=4,
    n_clusters_per_class=3,
    random_state=RND,
)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.33, random_state=RND
)

model = RandomForestClassifier(max_depth=5)
model.fit(train_X, train_y)
pred_y = model.predict_proba(test_X)[:, 1]
plot_roc_curve(test_y, pred_y)
```


    
![png](/images/eval/classification/ROC-AUC_files/ROC-AUC_6_0.png)
    


### Calculate ROC-AUC
{{% notice document %}}
[sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
{{% /notice %}}


```python
from sklearn.metrics import roc_auc_score

roc_auc_score(test_y, pred_y)
```




    0.89069793083171


