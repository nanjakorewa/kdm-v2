---
title: "Sensibilidad y especificidad: equilibrar falsos negativos y falsos positivos"
linkTitle: "Sensibilidad / Especificidad"
seo_title: "Sensibilidad vs. especificidad | Equilibrar falsos negativos y falsos positivos"
title_suffix: "Equilibrar falsos negativos y falsos positivos"
pre: "4.3.7 "
weight: 7
---

{{< lead >}}
La sensibilidad (recall para la clase positiva) mide qué tan bien capturamos los positivos, mientras que la especificidad mide qué tan bien evitamos alarmas falsas en los negativos. Ambas son clave cuando los costos de falsos negativos y falsos positivos difieren notablemente.
{{< /lead >}}

---

## 1. Definición
A partir de la matriz de confusión se obtienen las fórmulas:

$$
\mathrm{Sensitivity} = \frac{TP}{TP + FN}, \qquad
\mathrm{Specificity} = \frac{TN}{TN + FP}
$$

- Mayor sensibilidad implica menos positivos perdidos.  
- Mayor especificidad implica menos negativos marcados por error.

---

## 2. Cálculo en Python 3.13
```bash
python --version  # ej.: Python 3.13.0
pip install numpy scikit-learn
```

```python
import numpy as np
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)  # [[TN, FP], [FN, TP]]
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Sensibilidad:", round(sensitivity, 3))
print("Especificidad:", round(specificity, 3))
```

En scikit-learn, la sensibilidad aparece como `recall`. Para obtener la especificidad se puede usar `recall_score(y_true, y_pred, pos_label=0)` o calcularla manualmente como en el ejemplo.

---

## 3. Umbrales y trade-offs
En modelos probabilísticos, ajustar el umbral de decisión modifica la sensibilidad y la especificidad.

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, probas)
specificities = 1 - fpr
```

- Cada punto de la curva ROC representa un par específico de sensibilidad y especificidad.
- Si conocemos los costos de clasificación errónea, podemos optimizar el umbral usando métricas como el Índice de Youden o funciones de costo.

---

## 4. Índice de Youden y equilibrio
El Índice de Youden ayuda a encontrar un equilibrio entre ambas métricas:

$$
J = \mathrm{Sensitivity} + \mathrm{Specificity} - 1
$$

El umbral que maximiza \\(J\\) ofrece un buen compromiso cuando importan ambos tipos de error.

---

## 5. Recomendaciones prácticas
- **Priorizar la sensibilidad**: En tamizajes de enfermedades graves se busca evitar falsos negativos.
- **Priorizar la especificidad**: En sistemas antifraude se restringe el número de falsos positivos para no bloquear transacciones legítimas.
- **Informes claros**: Presenta sensibilidad y especificidad junto a Accuracy para que los responsables evalúen los riesgos explícitamente.

---

## Conclusiones
- La sensibilidad mide los positivos perdidos; la especificidad mide las falsas alarmas sobre negativos.
- Al mover el umbral se intercambian ambas métricas; ajusta el balance a los costos del negocio.
- Usa el Índice de Youden y métricas complementarias para revelar riesgos que Accuracy por sí solo no muestra.
