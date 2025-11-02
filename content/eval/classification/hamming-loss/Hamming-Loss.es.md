---
title: "Hamming Loss: tasa de error por etiqueta"
linkTitle: "Hamming Loss"
seo_title: "Hamming Loss | Medir el error por etiqueta en tareas multietiqueta"
title_suffix: "Medir el error por etiqueta en tareas multietiqueta"
pre: "4.3.13 "
weight: 13
---

{{< lead >}}
Hamming Loss cuantifica la proporción de etiquetas predichas de forma incorrecta. Es especialmente útil en clasificación multietiqueta, donde importa el número promedio de etiquetas falladas por muestra.
{{< /lead >}}

---

## 1. Definición
Sean \\(Y_i\\) los conjuntos de etiquetas verdaderas, \\(\hat{Y}_i\\) los conjuntos predichos y \\(L\\) el número total de etiquetas:

$$
\mathrm{Hamming\ Loss} = \frac{1}{nL} \sum_{i=1}^n \lvert Y_i \triangle \hat{Y}_i \rvert
$$

Donde \\(Y \triangle \hat{Y}\\) es la diferencia simétrica (etiquetas que solo aparecen en uno de los conjuntos). En escenarios multietiqueta equivale al número promedio de etiquetas incorrectas por muestra.

---

## 2. Cálculo en Python 3.13
```bash
python --version  # ej.: Python 3.13.0
pip install scikit-learn
```

```python
from sklearn.metrics import hamming_loss

print("Hamming Loss:", hamming_loss(y_true, y_pred))
```

`y_true` y `y_pred` deben ser matrices binarias 0/1 que indiquen la presencia de cada etiqueta (el resultado de `MultiLabelBinarizer` funciona muy bien).

---

## 3. Interpretación del puntaje
- Cuanto más cerca de 0, mejor. Un clasificador perfecto da 0.
- Un valor de 0.05 significa que “en promedio el 5 % de las etiquetas se predijeron mal”.
- Si las etiquetas tienen diferente importancia, considera usar una versión ponderada del Hamming Loss.

---

## 4. Relación con otras métricas
| Métrica         | Qué mide                      | Cuándo usarla                                |
| --------------- | ----------------------------- | -------------------------------------------- |
| Exact Match     | Exactitud perfecta por muestra| Requiere coincidencia total del conjunto     |
| **Hamming Loss**| Tasa de error por etiqueta    | Conocer el número promedio de errores        |
| Micro F1        | Equilibrio precision/recall   | Considerar el desbalance de positivos/negativos |
| Índice de Jaccard| Solapamiento de conjuntos    | Evaluar la similitud global de etiquetas     |

Hamming Loss es menos estricta que Exact Match y ofrece una señal más gradual para iterar sobre el modelo.

---

## 5. Consejos prácticos
- **Recomendación de etiquetas**: mide cuántas etiquetas se equivocan en promedio por ítem.
- **Sistemas de alertas**: controla la tasa de falsas alarmas cuando hay múltiples reglas simultáneas.
- **Evaluación ponderada**: aplica pesos según el impacto de cada etiqueta cuando el costo de error varía.

---

## Conclusiones
- Hamming Loss captura la tasa de error por etiqueta y facilita seguir la mejora en tareas multietiqueta.
- `hamming_loss` de scikit-learn es sencillo de usar y complementa a Exact Match y F1 para obtener una visión integral.
- Combínala con diagnósticos por etiqueta para priorizar la corrección donde los errores sean más costosos.
