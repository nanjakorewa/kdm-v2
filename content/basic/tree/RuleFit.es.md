---
title: "RuleFit | Reglas + Lineal con L1"
linkTitle: "RuleFit"
seo_title: "RuleFit | Reglas + Lineal con L1"
pre: "2.3.4 "
weight: 4
title_suffix: "Reglas + Lineal con L1"
---

{{% notice ref %}}
Friedman, Jerome H., y Bogdan E. Popescu. “Predictive learning via rule ensembles.” The Annals of Applied Statistics (2008). ([pdf](https://jerryfriedman.su.domains/ftp/RuleFit.pdf))
{{% /notice %}}

<div class="pagetop-box">
  <p><b>RuleFit</b> combina <b>reglas</b> extraídas de ensamblados de árboles con las características originales en un modelo <b>lineal</b>, ajustado con <b>L1</b> para esparsidad e interpretabilidad.</p>
</div>

---

## 1. Idea (con fórmulas)

1) <b>Extraer reglas</b> de árboles; cada camino a una hoja define $r_j(x)\in\{0,1\}).  
2) <b>Añadir términos lineales</b> escalados \\(z_k(x)$.  
3) <b>Ajuste lineal con L1</b>:

$$
\hat y(x) = \beta_0 + \sum_j \beta_j r_j(x) + \sum_k \gamma_k z_k(x)
$$

L1 deja solo reglas/términos importantes.

---

## 2. Datos (OpenML: house_sales)

{{% notice document %}}
- [fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset = fetch_openml(data_id=42092, as_frame=True)
X = dataset.data.select_dtypes("number").copy()
y = dataset.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 3. Ejecutar RuleFit

{{% notice document %}}
- Implementación Python: <a href="https://github.com/christophM/rulefit" target="_blank" rel="noopener">christophM/rulefit</a>
{{% /notice %}}

```python
from rulefit import RuleFit
import warnings
warnings.simplefilter("ignore")

rf = RuleFit(max_rules=200, tree_random_state=27)
rf.fit(X_train.values, y_train.values, feature_names=list(X_train.columns))

pred_tr = rf.predict(X_train.values)
pred_te = rf.predict(X_test.values)

def report(y_true, y_pred, name):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:8s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R2={r2:.3f}")

report(y_train, pred_tr, "Train")
report(y_test,  pred_te, "Test")
```

---

## 4. Ver reglas principales

```python
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values(by="importance", ascending=False)
rules.head(10)
```

- <b>rule</b>: condición if–then (o `type=linear` para términos lineales)  
- <b>coef</b>: coeficiente (misma unidad que el objetivo)  
- <b>support</b>: fracción de muestras que cumplen la regla  
- <b>importance</b>: importancia

Lectura: `type=linear` muestra efectos marginales; `type=rule` captura interacciones; busque coeficientes grandes con soporte razonable.

---

## 5. Validar reglas con visualización

Compruebe efectos lineales simples y reglas concretas con diagramas (dispersión, cajas, etc.).

---

## 6. Consejos prácticos

- Escalado y tratamiento de atípicos  
- Variables categóricas: agrupar niveles raros antes de one-hot  
- Transformar el objetivo si está sesgado (p. ej., `log(y)`)  
- Controlar nº y profundidad de reglas; ajustar con CV  
- Evitar leakage; preprocesar con `fit` en train

---

## 7. Resumen

- RuleFit = <b>reglas</b> de árboles + <b>términos lineales</b> + <b>L1</b>.  
- Equilibrio entre no linealidad e interpretabilidad.  
- Ajuste la granularidad y valide con gráficos.

---

