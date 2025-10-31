---
title: "Regresión por componentes principales (PCR)"
pre: "2.1.8 "
weight: 8
title_suffix: "Mitigar la multicolinealidad mediante reducción de dimensionalidad"
---

{{% summary %}}
- PCR aplica PCA para comprimir las características antes de ajustar la regresión lineal, reduciendo la inestabilidad causada por la multicolinealidad.
- Los componentes principales priorizan las direcciones con mayor varianza, filtrando el ruido y preservando la estructura informativa.
- Elegir cuántos componentes conservar permite equilibrar el riesgo de sobreajuste y el coste computacional.
- Un buen preprocesamiento —estandarización y tratamiento de valores perdidos— sienta las bases para un modelo preciso e interpretable.
{{% /summary %}}

## Intuición
Cuando los predictores están fuertemente correlacionados, los coeficientes de mínimos cuadrados pueden variar en exceso. PCR realiza primero PCA para combinar las direcciones correlacionadas y luego realiza la regresión sobre los componentes principales ordenados por la varianza explicada. Al centrarse en la variación dominante, los coeficientes resultantes son más estables.

## Formulación matemática
Aplicamos PCA a la matriz de diseño estandarizada \(\mathbf{X}\) y conservamos los \(k\) autovectores principales. Con las puntuaciones \(\mathbf{Z} = \mathbf{X} \mathbf{W}_k\), el modelo de regresión

$$
y = \boldsymbol{\gamma}^\top \mathbf{Z} + b
$$

se ajusta a los datos. Los coeficientes en el espacio original se recuperan mediante \(\boldsymbol{\beta} = \mathbf{W}_k \boldsymbol{\gamma}\). El número de componentes \(k\) se elige usando la varianza explicada acumulada o validación cruzada.

## Experimentos con Python
Evaluamos la regresión PCR en el conjunto de datos de diabetes variando el número de componentes y midiendo el desempeño con validación cruzada.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import japanize_matplotlib

X, y = load_diabetes(return_X_y=True)

def build_pcr(n_components):
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=0)),
        ("reg", LinearRegression()),
    ])

components = range(1, X.shape[1] + 1)
cv_scores = []
for k in components:
    model = build_pcr(k)
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    cv_scores.append(score.mean())

best_k = components[int(np.argmax(cv_scores))]
print("Número óptimo de componentes:", best_k)

best_model = build_pcr(best_k).fit(X, y)
print("Proporción de varianza explicada:", best_model["pca"].explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(components, [-s for s in cv_scores], marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"mejor={best_k}")
plt.xlabel("Número de componentes k")
plt.ylabel("CV MSE (más bajo es mejor)")
plt.legend()
plt.tight_layout()
plt.show()
```

![principal-component-regression block 1](/images/basic/regression/principal-component-regression_block01_es.png)

### Interpretación de los resultados
- Al aumentar los componentes, el ajuste sobre el entrenamiento mejora, pero el MSE de validación cruzada alcanza un mínimo intermedio.
- La proporción de varianza explicada indica cuánta variabilidad captura cada componente.
- Analizar los loadings muestra qué características originales contribuyen más a cada dirección principal.

## Referencias
{{% references %}}
<li>Jolliffe, I. T. (2002). <i>Principal Component Analysis</i> (2nd ed.). Springer.</li>
<li>Massy, W. F. (1965). Principal Components Regression in Exploratory Statistical Research. <i>Journal of the American Statistical Association</i>, 60(309), 234–256.</li>
{{% /references %}}
