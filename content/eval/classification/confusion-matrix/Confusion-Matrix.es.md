---
title: "Matriz de confusión | Cómo interpretar el rendimiento de un clasificador"
linkTitle: "Matriz de confusión"
seo_title: "Matriz de confusión | Cómo interpretar el rendimiento de un clasificador"
pre: "4.3.0 "
weight: 0
---

{{< lead >}}
La matriz de confusión ofrece una radiografía rápida de cómo un modelo asigna cada clase. Analizarla junto con métricas como precisión, exhaustividad y F1 facilita detectar los sesgos de predicción.
{{< /lead >}}

---

## 1. Anatomía de la matriz de confusión

En un problema binario la matriz tiene forma de tabla 2×2:

|                | Predicción: Negativa | Predicción: Positiva |
| -------------- | -------------------- | -------------------- |
| **Real: Negativa** | Verdadero negativo (TN) | Falso positivo (FP)   |
| **Real: Positiva** | Falso negativo (FN)     | Verdadero positivo (TP) |

- Las filas representan la verdad terreno y las columnas las predicciones del modelo.
- Revisar TP / FP / FN / TN ayuda a ver si el modelo favorece una clase sobre otra.

---

## 2. Ejemplo completo con Python 3.13

Confirma que trabajas con **Python 3.13** e instala las dependencias:

```bash
python --version  # p. ej. Python 3.13.0
pip install scikit-learn matplotlib
```

El siguiente script entrena una regresión logística sobre el conjunto Breast Cancer, imprime la matriz y la muestra como mapa de calor. La `Pipeline` con `StandardScaler` evita problemas de convergencia y estabiliza el entrenamiento.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, solver="lbfgs"),
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", colorbar=False)
plt.tight_layout()
plt.show()
```

{{< figure src="/images/eval/confusion-matrix/binary_matrix.png" alt="Matriz de confusión para el conjunto Breast Cancer" caption="Matriz de confusión generada con scikit-learn en Python 3.13" >}}

---

## 3. Normalizar para comparar proporciones

Si existen clases desbalanceadas, conviene normalizar por filas para observar tasas de error.

```python
cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
print(cm_norm)

disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
disp_norm.plot(cmap="Blues", values_format=".2f", colorbar=False)
plt.tight_layout()
plt.show()
```

- `normalize="true"`: proporción dentro de cada clase real  
- `normalize="pred"`: proporción dentro de cada clase predicha  
- `normalize="all"`: proporción sobre todas las observaciones  

---

## 4. Extensión a múltiples clases

`ConfusionMatrixDisplay.from_predictions` construye y etiqueta automáticamente la matriz en tareas multiclase.

```python
ConfusionMatrixDisplay.from_predictions(
    y_true=etiquetas_reales,
    y_pred=predicciones,
    normalize="true",
    values_format=".2f",
    cmap="Blues",
)
plt.tight_layout()
plt.show()
```

---

## 5. Puntos de control en proyectos reales

- **Falsos negativos vs. falsos positivos**: decide qué error es más costoso (ej. salud, fraude) y vigila esos valores.
- **Apóyate en mapas de calor**: facilitan identificar clases con sesgo y comunicar hallazgos al resto del equipo.
- **Métricas derivadas**: obtén precisión, exhaustividad y F1 a partir de la matriz, y compáralas con ROC-AUC o curvas PR para obtener una visión completa.
- **Notebooks reproducibles**: guardar el flujo en un cuaderno de Python 3.13 acelera los ciclos de ajuste y reentrenamiento.

---

## Resumen

- La matriz de confusión resume TP / FP / FN / TN y visibiliza los sesgos del clasificador.
- La normalización revela tasas de error cuando hay clases desbalanceadas.
- Combínala con métricas derivadas y con los requisitos del negocio para definir criterios de evaluación accionables.
