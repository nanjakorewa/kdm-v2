---
title: "Regresión Elastic Net"
pre: "2.1.5 "
weight: 5
title_suffix: "Fusionar las ventajas de L1 y L2"
---

{{% summary %}}
- Elastic Net combina las penalizaciones L1 (lasso) y L2 (ridge) para equilibrar esparsidad y estabilidad.
- Permite que grupos de características fuertemente correlacionadas se conserven mientras ajusta su importancia de forma conjunta.
- Ajustar \(\alpha\) y `l1_ratio` con validación cruzada facilita encontrar el equilibrio entre sesgo y varianza.
- Estandarizar los datos y permitir suficientes iteraciones mejora la estabilidad numérica del optimizador.
{{% /summary %}}

## Intuición
Lasso puede llevar coeficientes exactamente a cero y realiza selección de variables, pero cuando las características están muy correlacionadas puede quedarse con una sola y descartar las demás. Ridge reduce los coeficientes de manera suave y es estable, aunque nunca los anula. Elastic Net combina ambos efectos para que las variables correlacionadas permanezcan juntas mientras que los coeficientes poco útiles se acercan a cero.

## Formulación matemática
Elastic Net minimiza

$$
\min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left( y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b) \right)^2 + \alpha \left( \rho \lVert \boldsymbol\beta \rVert_1 + (1 - \rho) \lVert \boldsymbol\beta \rVert_2^2 \right),
$$

donde \(\alpha > 0\) controla la fuerza de regularización y \(\rho \in [0,1]\) (`l1_ratio`) define la mezcla entre L1 y L2. Al variar \(\rho\) podemos transitar entre el comportamiento de ridge y lasso.

## Experimentos con Python
Utilizamos `ElasticNetCV` para ajustar simultáneamente \(\alpha\) y `l1_ratio`, y luego inspeccionamos los coeficientes y el rendimiento.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Datos con fuerte correlación entre características
X, y, coef = make_regression(
    n_samples=500,
    n_features=30,
    n_informative=10,
    noise=15.0,
    coef=True,
    random_state=123,
)

# Duplicamos algunas columnas para reforzar la correlación
X = np.hstack([X, X[:, :5] + np.random.normal(scale=0.1, size=(X.shape[0], 5))])
feature_names = [f"x{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Ajuste automático de alpha y l1_ratio
enet_cv = ElasticNetCV(
    l1_ratio=[0.2, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=np.logspace(-3, 1, 30),
    cv=5,
    random_state=42,
    max_iter=5000,
)
enet_cv.fit(X_train, y_train)

print("Mejor alpha:", enet_cv.alpha_)
print("Mejor l1_ratio:", enet_cv.l1_ratio_)

# Entrenamos con los hiperparámetros óptimos y evaluamos
enet = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_, max_iter=5000)
enet.fit(X_train, y_train)

train_pred = enet.predict(X_train)
test_pred = enet.predict(X_test)

print("Train R^2:", r2_score(y_train, train_pred))
print("Test R^2:", r2_score(y_test, test_pred))
print("Test RMSE:", mean_squared_error(y_test, test_pred, squared=False))

# Coeficientes más relevantes
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": enet.coef_,
}).sort_values("coef", key=lambda s: s.abs(), ascending=False)
print(coef_df.head(10))
```

### Interpretación de los resultados
- `ElasticNetCV` evalúa automáticamente múltiples combinaciones L1/L2 y selecciona un buen punto medio.
- Cuando varias características correlacionadas permanecen, sus coeficientes tienden a alinearse en magnitud, lo que facilita la interpretación.
- Si la convergencia es lenta, estandarice las entradas o aumente `max_iter`.

## Referencias
{{% references %}}
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
<li>Friedman, J., Hastie, T., &amp; Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. <i>Journal of Statistical Software</i>, 33(1), 1–22.</li>
{{% /references %}}
