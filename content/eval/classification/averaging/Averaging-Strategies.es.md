---
title: "Elegir estrategias de promediado para métricas de clasificación"
linkTitle: "Estrategias de promediado"
seo_title: "Estrategias de promediado | Evaluación multiclase y multietiqueta"
pre: "4.3.14 "
weight: 14
---

{{< lead >}}
Al calcular Precision, Recall, F1 u otras métricas en un escenario multiclase o multietiqueta, el argumento `average` de scikit-learn controla cómo se agregan los resultados. Aquí comparamos las cuatro estrategias más habituales con ejemplos en Python 3.13.
{{< /lead >}}

---

## 1. Principales opciones de promediado
| average    | Cómo se calcula                                              | Cuándo utilizarla                                                  |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------------ |
| `micro`    | Suma TP/FP/FN de todas las muestras y luego calcula la métrica | Enfatiza la corrección global sin importar la distribución de clases |
| `macro`    | Calcula la métrica por clase y promedia sin ponderar          | Da el mismo peso a cada clase; resalta las minoritarias            |
| `weighted` | Calcula la métrica por clase y promedia ponderando por soporte | Mantiene las proporciones reales; se comporta parecido a Accuracy  |
| `samples`  | Solo para multietiqueta. Promedia las métricas por muestra    | Para casos donde cada muestra puede tener varias etiquetas         |

---

## 2. Comparación en Python 3.13
```bash
python --version  # ej.: Python 3.13.0
pip install scikit-learn matplotlib
```

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=30_000,
    n_features=20,
    n_informative=6,
    weights=[0.85, 0.1, 0.05],  # Clases desbalanceadas
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, multi_class="ovr"),
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
for avg in ["micro", "macro", "weighted"]:
    print(f"F1 ({avg}):", f1_score(y_test, y_pred, average=avg))
```

`classification_report` muestra las métricas por clase junto con `macro avg`, `weighted avg` y `micro avg`, lo que permite comparar las estrategias de un vistazo.

---

## 3. Cómo elegir
- **micro**: ideal si te importa la corrección global y cada predicción tiene el mismo peso.
- **macro**: úsalo cuando las clases minoritarias sean críticas; penaliza la baja cobertura en etiquetas raras.
- **weighted**: útil para mantener la distribución real de clases y seguir informando Precision/Recall/F1.
- **samples**: opción por defecto en tareas multietiqueta donde una muestra puede tener varias etiquetas verdaderas.

---

## Conclusiones
- El parámetro `average` cambia radicalmente el significado de la métrica; ajústalo al objetivo del proyecto.
- Recuerda: `macro` trata a las clases por igual, `micro` prioriza la proporción global, `weighted` mantiene el balance y `samples` está pensado para multietiqueta.
- scikit-learn permite calcular varios promedios a la vez, así que vale la pena reportarlos para evitar malinterpretar la calidad del modelo.
