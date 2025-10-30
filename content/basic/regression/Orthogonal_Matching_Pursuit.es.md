---
title: "Orthogonal Matching Pursuit (OMP)"
pre: "2.1.12 "
weight: 12
title_suffix: "Selección codiciosa de coeficientes dispersos"
---

{{% summary %}}
- Orthogonal Matching Pursuit (OMP) selecciona en cada paso la característica más correlacionada con el residuo para construir un modelo lineal disperso.
- Resuelve un problema de mínimos cuadrados restringido a las características ya elegidas, lo que mantiene los coeficientes interpretables.
- En lugar de ajustar una fuerza de regularización, la dispersión se controla indicando cuántas características deben conservarse.
- Estandarizar las características y revisar la multicolinealidad ayuda a que el método permanezca estable al recuperar señales dispersas.
{{% /summary %}}

## Intuición
Cuando solo unas pocas características son realmente útiles, OMP añade en cada iteración la que más reduce el error residual. Es un algoritmo básico de aprendizaje de diccionarios y codificación dispersa, y produce vectores de coeficientes compactos que resaltan los predictores más relevantes.

## Formulación matemática
Inicializamos el residuo como \(\mathbf{r}^{(0)} = \mathbf{y}\). Para cada iteración \(t\):

1. Calculamos el producto interno entre cada característica \(\mathbf{x}_j\) y el residuo \(\mathbf{r}^{(t-1)}\), y elegimos la características \(j\) con mayor valor absoluto.
2. Añadimos \(j\) al conjunto activo \(\mathcal{A}_t\).
3. Resolvemos mínimos cuadrados restringidos a \(\mathcal{A}_t\) para obtener \(\hat{\boldsymbol\beta}_{\mathcal{A}_t}\).
4. Actualizamos el residuo \(\mathbf{r}^{(t)} = \mathbf{y} - \mathbf{X}_{\mathcal{A}_t} \hat{\boldsymbol\beta}_{\mathcal{A}_t}\).

El proceso se detiene al alcanzar la dispersión deseada o cuando el residuo es suficientemente pequeño.

## Experimentos con Python
El siguiente código compara OMP y lasso sobre datos cuyo vector de coeficientes verdadero es disperso.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
n_samples, n_features = 200, 40
X = rng.normal(size=(n_samples, n_features))
true_coef = np.zeros(n_features)
true_support = [1, 5, 12, 33]
true_coef[true_support] = [2.5, -1.5, 3.0, 1.0]
y = X @ true_coef + rng.normal(scale=0.5, size=n_samples)

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=4).fit(X, y)
lasso = Lasso(alpha=0.05).fit(X, y)

print("Índices no nulos de OMP:", np.flatnonzero(omp.coef_))
print("Índices no nulos de Lasso:", np.flatnonzero(np.abs(lasso.coef_) > 1e-6))

print("MSE de OMP:", mean_squared_error(y, omp.predict(X)))
print("MSE de Lasso:", mean_squared_error(y, lasso.predict(X)))

coef_df = pd.DataFrame({
    "true": true_coef,
    "omp": omp.coef_,
    "lasso": lasso.coef_,
})
print(coef_df.head(10))
```

### Interpretación de los resultados
- Si `n_nonzero_coefs` coincide con el número real de coeficientes distintos de cero, OMP recupera las características relevantes con alta probabilidad.
- Comparado con lasso, OMP produce coeficientes exactamente nulos para las características descartadas.
- Cuando las características están muy correlacionadas, el orden de selección puede variar; conviene realizar preprocesados y diseño de características adecuados.

## Referencias
{{% references %}}
<li>Pati, Y. C., Rezaiifar, R., &amp; Krishnaprasad, P. S. (1993). Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition. En <i>Conference Record of the Twenty-Seventh Asilomar Conference on Signals, Systems and Computers</i>.</li>
<li>Tropp, J. A. (2004). Greed is Good: Algorithmic Results for Sparse Approximation. <i>IEEE Transactions on Information Theory</i>, 50(10), 2231–2242.</li>
{{% /references %}}
