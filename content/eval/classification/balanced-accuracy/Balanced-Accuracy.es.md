---

title: "Balanced Accuracy | Evaluar modelos con datos desbalanceados"

linkTitle: "Balanced Accuracy"

seo_title: "Balanced Accuracy | Evaluar modelos con datos desbalanceados"

pre: "4.3.6 "

weight: 6

---



{{< lead >}}

Balanced Accuracy promedia el recall de cada clase, por lo que sigue siendo útil cuando el conjunto de datos está desbalanceado. Con el ejemplo en Python 3.13 podemos ver en qué se diferencia de la exactitud habitual y cuándo conviene reportarla.

{{< /lead >}}



---



## 1. Definición



La fórmula se define como el promedio entre la tasa de verdaderos positivos (TPR) y la tasa de verdaderos negativos (TNR):





\mathrm{Balanced\ Accuracy} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)





En entornos multiclase se promedia el recall de cada clase siguiendo el mismo criterio.



---



## 2. Implementación en Python 3.13



```bash

python --version        # p. ej. Python 3.13.0

pip install scikit-learn matplotlib

```



Reutilizamos el clasificador Random Forest del artículo de Accuracy y calculamos ambas métricas. El diagrama de barras se almacena en static/images/eval/classification/accuracy/accuracy_vs_balanced.png, listo para regenerarse con generate_eval_assets.py cuando actualices el cuaderno.



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

```



{{< figure src="/images/eval/classification/accuracy/accuracy_vs_balanced.png" alt="Comparación Accuracy vs Balanced Accuracy" caption="Balanced Accuracy mantiene el recall de cada clase en igualdad de condiciones." >}}



---



## 3. Cuándo preferir Balanced Accuracy



- **Datos muy desbalanceados** – la Accuracy clásica recompensa a la clase mayoritaria, mientras que Balanced Accuracy expone si la clase minoritaria queda sin detectar.

- **Comparación de modelos** – en benchmarks con clases raras, Balanced Accuracy evita conclusiones engañosas.

- **Ajuste de umbral** – combínala con curvas precision–recall para saber si ambas clases se mantienen visibles en el punto de operación.



---



## 4. Métricas complementarias



| Métrica | Qué mide | Advertencia en datos desbalanceados |

| --- | --- | --- |

| Accuracy | Porcentaje total de aciertos | Puede ignorar por completo a la clase minoritaria |

| Recall / Sensitivity | Tasa de detección por clase | Se reporta clase por clase |

| **Balanced Accuracy** | Media del recall por clase | Hace visibles las pérdidas en clases pequeñas |

| Macro F1 | Media armónica de precision y recall (por clase) | Útil cuando también importa la precisión |



---



## Resumen



- Balanced Accuracy promedia el recall de cada clase y es ideal para conjuntos desbalanceados.

- Con alanced_accuracy_score en Python 3.13 puedes obtenerla en una línea y compararla con la Accuracy tradicional.

- Acompáñala de precision, recall y F1 para comunicar claramente la calidad de un modelo a todas las partes interesadas.

---

