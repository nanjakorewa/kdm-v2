---
title: "Regresión Partial Least Squares (PLS)"
pre: "2.1.9 "
weight: 9
title_suffix: "Factores latentes alineados con la variable objetivo"
---

{{% summary %}}
- PLS extrae factores latentes que maximizan la covarianza entre predictores y objetivo antes de realizar la regresión.
- A diferencia de PCA, los ejes aprendidos incorporan información de la variable objetivo, manteniendo el desempeño predictivo mientras se reduce la dimensionalidad.
- Ajustar el número de factores latentes estabiliza el modelo cuando existe fuerte multicolinealidad.
- Analizar los loadings permite identificar qué combinaciones de características se relacionan más con la variable objetivo.
{{% /summary %}}

## Intuición
La regresión por componentes principales solo considera la varianza de las características, por lo que puede descartar direcciones relevantes para el objetivo. PLS construye factores latentes considerando simultáneamente los predictores y la respuesta, conservando la información más útil para predecir. Así es posible trabajar con una representación compacta sin sacrificar precisión.

## Formulación matemática
Dado un matriz de predictores \(\mathbf{X}\) y un vector objetivo \(\mathbf{y}\), PLS alterna las actualizaciones de las puntuaciones latentes \(\mathbf{t} = \mathbf{X} \mathbf{w}\) y \(\mathbf{u} = \mathbf{y} c\) de forma que la covarianza \(\mathbf{t}^\top \mathbf{u}\) sea máxima. Repetir el proceso produce un conjunto de factores latentes sobre el que se ajusta un modelo lineal

$$
\hat{y} = \mathbf{t} \boldsymbol{b} + b_0.
$$

El número de factores \(k\) suele elegirse mediante validación cruzada.

## Experimentos con Python
Comparamos el desempeño de PLS para distintos números de factores latentes con el conjunto de datos de acondicionamiento físico de Linnerud.

```python
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_linnerud
import matplotlib.pyplot as plt
import japanize_matplotlib

data = load_linnerud()
X = data["data"]          # Rutinas de entrenamiento y medidas corporales
y = data["target"][:, 0]  # Aquí predecimos únicamente el número de dominadas

components = range(1, min(X.shape[1], 6) + 1)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

scores = []
for k in components:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("pls", PLSRegression(n_components=k)),
    ])
    score = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error").mean()
    scores.append(score)

best_k = components[int(np.argmax(scores))]
print("Número óptimo de factores latentes:", best_k)

best_model = Pipeline([
    ("scale", StandardScaler()),
    ("pls", PLSRegression(n_components=best_k)),
]).fit(X, y)

print("Loadings de X:\n", best_model["pls"].x_loadings_)
print("Loadings de Y:\n", best_model["pls"].y_loadings_)

plt.figure(figsize=(8, 4))
plt.plot(components, [-s for s in scores], marker="o")
plt.axvline(best_k, color="red", linestyle="--")
plt.xlabel("Número de factores latentes")
plt.ylabel("CV MSE (más bajo es mejor)")
plt.tight_layout()
plt.show()
```

![partial-least-squares block 1](/images/basic/regression/partial-least-squares_block01.svg)

### Interpretación de los resultados
- El MSE de validación cruzada desciende al añadir factores, alcanza un mínimo y luego empeora si seguimos agregando más.
- Inspeccionar `x_loadings_` y `y_loadings_` muestra qué características contribuyen más a cada factor latente.
- Estandarizar las entradas garantiza que características con diferentes escalas aporten de manera equilibrada.

## Referencias
{{% references %}}
<li>Wold, H. (1975). Soft Modelling by Latent Variables: The Non-Linear Iterative Partial Least Squares (NIPALS) Approach. En <i>Perspectives in Probability and Statistics</i>. Academic Press.</li>
<li>Geladi, P., &amp; Kowalski, B. R. (1986). Partial Least-Squares Regression: A Tutorial. <i>Analytica Chimica Acta</i>, 185, 1–17.</li>
{{% /references %}}
