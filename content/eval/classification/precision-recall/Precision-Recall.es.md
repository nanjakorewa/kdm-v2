---
title: "Precisión, exhaustividad y F1 | Ajuste de umbrales en Python 3.13"
linkTitle: "Precision-Recall"
seo_title: "Precisión, exhaustividad y F1 | Ajuste de umbrales en Python 3.13"
pre: "4.3.2 "
weight: 2
---

{{< lead >}}
La precisión indica cuántas predicciones positivas son correctas, la exhaustividad revela cuántos positivos reales recuperamos y el F1 combina ambas métricas. Con código reproducible en Python 3.13 podemos explorar la curva precision–recall y decidir qué umbral ofrece el mejor equilibrio.
{{< /lead >}}

---

## 1. Definiciones esenciales

Dados los recuentos de la matriz de confusión — verdaderos positivos (TP), falsos positivos (FP), falsos negativos (FN) — las métricas se definen así:


\text{Precision} = \frac{TP}{TP + FP}, \qquad
\text{Recall} = \frac{TP}{TP + FN}


- **Precisión**: proporción de aciertos entre las predicciones positivas. Útil cuando los falsos positivos son costosos.
- **Exhaustividad (Recall)**: proporción de positivos reales detectados. Es clave cuando los falsos negativos deben evitarse.
- **F1**: media armónica de precisión y exhaustividad; aporta un resumen único.


F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}


---

## 2. Implementación y curva PR en Python 3.13

Comprueba el intérprete e instala las dependencias necesarias:

`ash
python --version        # p. ej. Python 3.13.0
pip install scikit-learn matplotlib
`

El siguiente script genera un conjunto de datos desbalanceado (clase positiva 5 %), entrena una regresión logística con pesos equilibrados y dibuja la curva precision–recall junto con el average precision (AP). La figura se guarda en static/images/eval/classification/precision-recall/pr_curve.png y puede regenerarse mediante generate_eval_assets.py.

`python
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=40_000,
    n_features=20,
    n_informative=6,
    weights=[0.95, 0.05],
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)
print(classification_report(y_test, y_pred, digits=3))

precision, recall, thresholds = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)

fig, ax = plt.subplots(figsize=(5, 4))
ax.step(recall, precision, where="post", label=f"Curva PR (AP={ap:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precisión")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
output_dir = Path("static/images/eval/classification/precision-recall")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "pr_curve.png", dpi=150)
plt.close(fig)
`

{{< figure src="/images/eval/classification/precision-recall/pr_curve.png" alt="Curva precision–recall" caption="El average precision (AP) corresponde al área bajo la curva PR. Cambiar el umbral desplaza el punto sobre la curva." >}}

---

## 3. Cómo elegir el umbral

- El umbral por defecto (0.5) puede producir una exhaustividad demasiado baja si la clase positiva es minoritaria.
- Cada punto de la curva PR está asociado a un umbral del array 	hresholds que devuelve precision_recall_curve.
- Reducir el umbral aumenta el recall a costa de disminuir la precisión; la decisión final depende del coste de cada tipo de error.

`python
threshold = 0.3
custom_pred = (proba >= threshold).astype(int)
print(
    "threshold=0.30",
    "Precision=", precision_score(y_test, custom_pred),
    "Recall=", recall_score(y_test, custom_pred),
    "F1=", f1_score(y_test, custom_pred),
)
`

---

## 4. Estrategias de promedio para multiclase

El parámetro verage permite seleccionar cómo combinar métricas por clase:

- macro — media simple; todas las clases pesan lo mismo.
- weighted — media ponderada por el soporte de cada clase; mantiene el equilibrio global.
- micro — recálculo sobre todas las observaciones; útil pero puede ocultar el comportamiento de clases minoritarias.

`python
precision_score(y_test, y_pred, average="macro")
recall_score(y_test, y_pred, average="weighted")
f1_score(y_test, y_pred, average="micro")
`

---

## Resumen

- Precisión reduce falsos positivos, recall reduce falsos negativos; el F1 resume ambos.
- La curva precision–recall muestra cómo cambia el equilibrio al desplazar el umbral, y el AP condensa esa información en un único número.
- Con scikit-learn en Python 3.13 es sencillo calcular y compartir la curva para apoyar decisiones sobre el umbral y comparar modelos.
---
