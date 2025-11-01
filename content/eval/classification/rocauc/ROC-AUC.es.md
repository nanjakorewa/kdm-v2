---
title: "ROC-AUC | Guía para ajustar umbrales y comparar modelos"
linkTitle: "ROC-AUC"
seo_title: "ROC-AUC | Guía para ajustar umbrales y comparar modelos"
pre: "4.3.3 "
weight: 1
---

{{< lead >}}
La curva ROC muestra cómo evolucionan la tasa de verdaderos positivos y la tasa de falsos positivos cuando movemos el umbral de decisión, y el AUC (área bajo la curva) resume esa capacidad de discriminación. Con código reproducible en Python 3.13 podemos visualizarla y usarla para calibrar el modelo.
{{< /lead >}}

---

## 1. Qué representan la curva ROC y el AUC

La curva ROC traza la **tasa de falsos positivos (FPR)** en el eje x y la **tasa de verdaderos positivos (TPR)** en el eje y mientras variamos el umbral entre 0 y 1. El AUC toma valores entre 0.5 (azar) y 1.0 (separación perfecta).

- AUC ≈ 1.0 → el modelo distingue muy bien ambas clases.
- AUC ≈ 0.5 → comportamiento aleatorio.
- AUC < 0.5 → la predicción puede estar “invertida”, por lo que cambiar el signo o el umbral podría mejorarla.

---

## 2. Implementación y visualización con Python 3.13

Comprueba el intérprete e instala las dependencias:

`ash
python --version        # p. ej. Python 3.13.0
pip install scikit-learn matplotlib
`

El código siguiente entrena una regresión logística sobre el conjunto Breast Cancer, genera la curva ROC y guarda la figura en static/images/eval/classification/rocauc. Es compatible con generate_eval_assets.py para regenerar los activos cuando sea necesario.

`python
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, solver="lbfgs"),
)
pipeline.fit(X_train, y_train)
proba = pipeline.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, proba)
print(f"ROC-AUC: {auc:.3f}")

fig, ax = plt.subplots(figsize=(5, 5))
RocCurveDisplay.from_predictions(
    y_test,
    proba,
    name="Logistic Regression",
    ax=ax,
)
ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5, label="Aleatorio")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Curva ROC (Breast Cancer Dataset)")
ax.legend(loc="lower right")
fig.tight_layout()
output_dir = Path("static/images/eval/classification/rocauc")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "roc_curve.png", dpi=150)
plt.close(fig)
`

{{< figure src="/images/eval/classification/rocauc/roc_curve.png" alt="Ejemplo de curva ROC" caption="El área bajo la curva (AUC) resume la capacidad del modelo para ordenar correctamente las clases." >}}

---

## 3. Uso práctico para ajustar umbrales

- **Dominios sensibles al recall** (salud, fraude): selecciona un punto de la curva que maximice TPR con un FPR aceptable.
- **Balance precisión–recall**: los modelos con AUC alto suelen mantener buen rendimiento en un rango amplio de umbrales.
- **Comparación de modelos**: el AUC ofrece un escalar independiente del umbral para evaluar alternativas antes de elegir el punto de operación.

Combina ROC-AUC con el análisis de precisión–recall para comprender el costo de mover el umbral.

---

## 4. Lista de comprobación operativa

1. **Revisa el desbalanceo** – incluso con AUC ≈ 0.5, otro umbral puede rescatar casos valiosos.
2. **Prueba pesos de clase** – observa si ajustar los pesos mejora el AUC.
3. **Comparte la visualización** – incluir la curva ROC en dashboards facilita discutir trade-offs con el equipo.
4. **Notebook reproducible en Python 3.13** – mantener el flujo documentado agiliza las reevaluaciones tras cada retraining.

---

## Resumen

- ROC-AUC mide la capacidad de un modelo para ordenar los positivos por encima de los negativos a lo largo de todos los umbrales.
- En Python 3.13, RocCurveDisplay y oc_auc_score simplifican el cálculo y la visualización.
- Úsalo junto a métricas de precision–recall para seleccionar umbrales acordes a los objetivos de negocio.
---
