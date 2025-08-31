---
title: "Regresión Ridge y Lasso"
pre: "2.1.2 "
weight: 2
title_suffix: "Intuición y práctica"
---

{{% youtube "rhGYOBrxPXA" %}}


<div class="pagetop-box">
  <p><b>Ridge</b> (L2) y <b>Lasso</b> (L1) agregan penalizaciones al tamaño de los coeficientes para <b>reducir el sobreajuste</b> y mejorar la <b>generalización</b>. A esto lo llamamos <b>regularización</b>. Ajustaremos OLS, Ridge y Lasso bajo las mismas condiciones y compararemos su comportamiento.</p>
  </div>

---

## 1. Intuición y fórmulas

OLS minimiza la suma de cuadrados de los residuos:
$$
\text{RSS}(\boldsymbol\beta, b) = \sum_{i=1}^n \big(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\big)^2.
$$
Con muchas variables, colinealidad fuerte o ruido, los coeficientes pueden crecer demasiado y sobreajustar. La regularización agrega una penalización para <b>encoger</b> los coeficientes.

- <b>Ridge (L2)</b>
  $$
  \min_{\boldsymbol\beta, b}\; \text{RSS}(\boldsymbol\beta,b) + \alpha \lVert \boldsymbol\beta \rVert_2^2
  $$
  Encoge todos los coeficientes de forma suave (rara vez exacto cero).

- <b>Lasso (L1)</b>
  $$
  \min_{\boldsymbol\beta, b}\; \text{RSS}(\boldsymbol\beta,b) + \alpha \lVert \boldsymbol\beta \rVert_1
  $$
  Lleva algunos coeficientes exactamente a cero (selección de variables).

> Heurísticas:
> - Ridge “encoge todo un poco” → estable, no esparso.
> - Lasso “lleva algunos a cero” → modelos esparsos, puede ser inestable con colinealidad fuerte.

---

## 2. Preparación

{{% notice tip %}}
Con regularización, <b>las escalas de las variables importan</b>. Use <b>StandardScaler</b> para que la penalización actúe de forma equitativa.
{{% /notice %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

---

## 3. Datos: solo 2 variables informativas

{{% notice document %}}
- [sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)
{{% /notice %}}

Creamos datos con exactamente dos variables informativas (<code>n_informative=2</code>) y luego añadimos transformaciones redundantes para simular muchas variables no útiles.

```python
n_features = 5
n_informative = 2

X, y = make_regression(
    n_samples=500,
    n_features=n_features,
    n_informative=n_informative,
    noise=0.5,
    random_state=777,
)

# Agregar transformaciones no lineales redundantes
X = np.concatenate([X, np.log(X + 100)], axis=1)

# Estandarizar y para escalar de forma justa
y = (y - y.mean()) / np.std(y)
```

---

## 4. Entrenar tres modelos con el mismo pipeline

{{% notice document %}}
- [make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)  
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)  
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
{{% /notice %}}

```python
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
rid_r = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2)).fit(X, y)
las_r = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.1, max_iter=10_000)).fit(X, y)
```

> Lasso puede converger lentamente; en la práctica, aumente <code>max_iter</code> si hace falta.

---

## 5. Comparar magnitudes de coeficientes (gráfico)

- OLS: los coeficientes rara vez cercanos a cero.  
- Ridge: encoge los coeficientes de forma general.  
- Lasso: algunos coeficientes quedan exactamente en cero (selección de variables).

```python
feat_index = np.arange(X.shape[1])

plt.figure(figsize=(12, 4))
plt.bar(feat_index - 0.25, np.abs(lin_r.steps[1][1].coef_), width=0.25, label="Linear")
plt.bar(feat_index,         np.abs(rid_r.steps[1][1].coef_), width=0.25, label="Ridge")
plt.bar(feat_index + 0.25,  np.abs(las_r.steps[1][1].coef_), width=0.25, label="Lasso")

plt.xlabel(r"índice del coeficiente $\beta$")
plt.ylabel(r"$|\beta|$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

![png](/images/basic/regression/02_Ridge_and_Lasso_files/02_Ridge_and_Lasso_10_1.png)

{{% notice tip %}}
En este ejemplo solo dos variables son informativas (<code>n_informative=2</code>). Lasso tiende a anular muchas variables inútiles (modelo esparso), mientras Ridge encoge los coeficientes para estabilizar la solución.
{{% /notice %}}

---

## 6. Comprobación de generalización (CV)

Un <code>alpha</code> fijo puede engañar; compare con validación cruzada.

```python
from sklearn.metrics import make_scorer, mean_squared_error

scorer = make_scorer(mean_squared_error, greater_is_better=False)

models = {
    "Linear": lin_r,
    "Ridge (α=2)": rid_r,
    "Lasso (α=0.1)": las_r,
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name:12s}  R^2 (CV media±std): {scores.mean():.3f} ± {scores.std():.3f}")
```

{{% notice tip %}}
No hay una única respuesta. Con colinealidad fuerte o pocas muestras, Ridge/Lasso suelen ser más estables que OLS. <b>Ajuste α</b> con CV.
{{% /notice %}}

---

## 7. Elegir α automáticamente

En la práctica, use <code>RidgeCV</code>/<code>LassoCV</code> (o <code>GridSearchCV</code>).

```python
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
).fit(X, y)

lasso_cv = make_pipeline(
    StandardScaler(with_mean=False),
    LassoCV(alphas=np.logspace(-3, 1, 9), cv=5, max_iter=50_000)
).fit(X, y)

print("Mejor α (RidgeCV):", ridge_cv.steps[1][1].alpha_)
print("Mejor α (LassoCV):", lasso_cv.steps[1][1].alpha_)
```

---

## 8. ¿Cuál usar?

- Muchas variables / colinealidad fuerte → pruebe <b>Ridge</b> (estabiliza).
- Quiere selección de variables / interpretabilidad → <b>Lasso</b>.
- Colinealidad fuerte y Lasso inestable → <b>Elastic Net</b> (L1+L2).

---

## 9. Resumen

- La regularización agrega <b>penalizaciones al tamaño de los coeficientes</b> para reducir el sobreajuste.
- <b>Ridge (L2)</b> encoge de forma suave; rara vez cero exacto.
- <b>Lasso (L1)</b> lleva algunos coeficientes a cero; selección de variables.
- <b>Estandarice</b> y <b>ajuste α con CV</b>.

---

