---
title: "Regresiones Ridge y Lasso"
pre: "2.1.2 "
weight: 2
title_suffix: "Mejorar la generalización con regularización"
---

{{% summary %}}
- Ridge contrae suavemente los coeficientes mediante una penalización L2 y se mantiene estable incluso con multicolinealidad.
- Lasso aplica una penalización L1 que puede llevar algunos coeficientes exactamente a cero, lo que incorpora selección de variables e interpretabilidad.
- Ajustar la fuerza de regularización \(\alpha\) permite equilibrar el ajuste del conjunto de entrenamiento con la capacidad de generalizar.
- Combinar estandarización y validación cruzada ayuda a elegir hiperparámetros que evitan el sobreajuste sin perder rendimiento.
{{% /summary %}}

## Intuición
Los mínimos cuadrados ordinarios pueden producir coeficientes extremos cuando hay valores atípicos o las variables explicativas están muy correlacionadas. Ridge y Lasso añaden penalizaciones para evitar que los coeficientes crezcan sin control y así equilibran precisión e interpretabilidad. Ridge “encoge todo un poco E, mientras que Lasso “apaga Ealgunos coeficientes al volverlos cero.

## Formulación matemática
Ambos métodos minimizan el error cuadrático habitual más un término de regularización:

- **Ridge**
  $$
  \min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2 + \alpha \lVert \boldsymbol\beta \rVert_2^2
  $$
- **Lasso**
  $$
  \min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\right)^2 + \alpha \lVert \boldsymbol\beta \rVert_1
  $$

Cuanto mayor sea \(\alpha\), más fuerte será la contracción. En el caso de Lasso, cuando \(\alpha\) supera cierto umbral algunos coeficientes se vuelven exactamente cero, generando modelos dispersos.

## Experimentos con Python
El siguiente ejemplo aplica ridge, lasso y regresión lineal ordinaria al mismo problema sintético y compara la magnitud de los coeficientes y el desempeño de generalización.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Parámetros
n_samples = 200
n_features = 10
n_informative = 3
rng = np.random.default_rng(42)

# Coeficientes verdaderos (solo los tres primeros informativos)
coef = np.zeros(n_features)
coef[:n_informative] = rng.normal(loc=3.0, scale=1.0, size=n_informative)

X = rng.normal(size=(n_samples, n_features))
y = X @ coef + rng.normal(scale=5.0, size=n_samples)

linear = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
ridge = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
).fit(X, y)
lasso = make_pipeline(
    StandardScaler(with_mean=False),
    LassoCV(alphas=np.logspace(-3, 1, 9), cv=5, max_iter=50_000)
).fit(X, y)

models = {
    "Linear": linear,
    "Ridge": ridge,
    "Lasso": lasso,
}

# Magnitud de los coeficientes
indices = np.arange(n_features)
width = 0.25
plt.figure(figsize=(10, 4))
plt.bar(indices - width, np.abs(linear[-1].coef_), width=width, label="Linear")
plt.bar(indices, np.abs(ridge[-1].coef_), width=width, label="Ridge")
plt.bar(indices + width, np.abs(lasso[-1].coef_), width=width, label="Lasso")
plt.xlabel("índice de la característica")
plt.ylabel("|coeficiente|")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Puntajes R^2 con validación cruzada
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name:>6}: R^2 = {scores.mean():.3f} ± {scores.std():.3f}")
```

### Interpretación de los resultados
- Ridge reduce ligeramente todos los coeficientes y se mantiene estable incluso con multicolinealidad.
- Lasso lleva algunos coeficientes a cero, conservando únicamente las características más informativas.
- Seleccione \(\alpha\) mediante validación cruzada para equilibrar sesgo y varianza, y estandarice las características para compararlas en la misma escala.

## Referencias
{{% references %}}
<li>Hoerl, A. E., &amp; Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. <i>Technometrics</i>, 12(1), 55–67.</li>
<li>Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. <i>Journal of the Royal Statistical Society: Series B</i>, 58(1), 267–288.</li>
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
{{% /references %}}
