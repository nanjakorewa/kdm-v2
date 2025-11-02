---

title: "Exactitud (Accuracy) | Fundamentos y riesgos en Python 3.13"

linkTitle: "Accuracy"

seo_title: "Exactitud (Accuracy) | Fundamentos y riesgos en Python 3.13"

pre: "4.3.1 "

weight: 1

---



{{< lead >}}

La exactitud mide el porcentaje de ejemplos que el modelo acierta, pero puede resultar engañosa cuando las clases están desbalanceadas. Aquí revisamos cómo calcularla y visualizarla en Python 3.13 y qué métricas complementarias conviene reportar.

{{< /lead >}}



---



## 1. Definición



A partir de la matriz de confusión (verdaderos positivos TP, falsos positivos FP, falsos negativos FN, verdaderos negativos TN) la exactitud se define como:



$$

\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

$$



Indica qué proporción total se clasifica correctamente, pero no revela el comportamiento frente a clases minoritarias.



---



## 2. Implementación y visualización con Python 3.13



Verifica el intérprete e instala las dependencias:



```bash

python --version        # p. ej. Python 3.13.0

pip install scikit-learn matplotlib

```



El siguiente script entrena un Random Forest sobre el conjunto Breast Cancer, calcula Accuracy y Balanced Accuracy y muestra ambas métricas en un gráfico de barras. El uso de `Pipeline` + `StandardScaler` deja el flujo reproducible. Las imágenes se guardan en `static/images/eval/...` y pueden regenerarse con `generate_eval_assets.py`.



```python

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=42, stratify=y

)



pipeline = make_pipeline(

    StandardScaler(),

    RandomForestClassifier(random_state=42, n_estimators=300),

)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



acc = accuracy_score(y_test, y_pred)

bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}, Balanced Accuracy: {bal_acc:.3f}")



fig, ax = plt.subplots(figsize=(5, 4))

scores = np.array([acc, bal_acc])

labels = ["Accuracy", "Balanced Accuracy"]

colors = ["#2563eb", "#f97316"]

bars = ax.bar(labels, scores, color=colors)

ax.set_ylim(0, 1.05)

for bar, score in zip(bars, scores):

    ax.text(bar.get_x() + bar.get_width() / 2, score + 0.02, f"{score:.3f}", ha="center", va="bottom")

ax.set_ylabel("Puntuación")

ax.set_title("Accuracy vs. Balanced Accuracy (Breast Cancer Dataset)")

ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

output_dir = Path("static/images/eval/classification/accuracy")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "accuracy_vs_balanced.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/accuracy/accuracy_vs_balanced.png" alt="Comparación entre Accuracy y Balanced Accuracy" caption="Balanced Accuracy pone en evidencia los fallos cuando hay clases desbalanceadas." >}}



---



## 3. Qué hacer ante clases desbalanceadas



Accuracy ignora el costo relativo de los falsos negativos y falsos positivos. Para tener una visión completa:



- **Precision / Recall / F1**: cuantifican las falsas alarmas y las omisiones.

- **Balanced Accuracy**: promedia el recall de cada clase, haciendo visibles las minoritarias.

- **Matriz de confusión**: indica en qué clases se concentran los errores.

- **Curvas ROC-AUC / PR**: muestran cómo cambian las métricas al ajustar el umbral de decisión.



Balanced Accuracy equivale al promedio de los recalls por clase y suele adoptarse como métrica principal cuando el costo de perder casos minoritarios es alto.



---



## 4. Lista de comprobación operativa



1. **¿Coincide con el costo de negocio?** Revisa la matriz de confusión: una “exactitud del 99 %” puede ocultar que nunca se detectan los eventos realmente críticos.

2. **¿Hay margen al ajustar el umbral?** Analiza ROC-AUC o PR para ver cómo evolucionaría Accuracy si cambias el umbral.

3. **Reporta múltiples métricas**: comparte Precision, Recall, F1 y Balanced Accuracy junto con Accuracy para exponer el equilibrio de errores.

4. **Cuaderno reproducible**: mantén un notebook en Python 3.13 que permita repetir la evaluación tras cada retraining.



---



## Resumen



- Accuracy es un buen titular, pero engañoso cuando las clases están desbalanceadas.

- Un pipeline con escalamientos en Python 3.13 facilita la reproducción del cálculo.

- Combínala con Balanced Accuracy y métricas por clase para tomar decisiones fundamentadas.

