---
title: "Top-k Accuracy y Recall@k"
linkTitle: "Top-k Accuracy"
seo_title: "Top-k Accuracy y Recall@k | Evaluar listas de candidatos"
title_suffix: "Evaluar si la etiqueta correcta aparece entre los candidatos principales"
pre: "4.3.12 "
weight: 12
---

{{< lead >}}
Top-k Accuracy (o Recall@k) mide con qué frecuencia la etiqueta correcta aparece entre los k principales candidatos que devuelve el modelo. Es la métrica habitual en clasificación multiclase y sistemas de recomendación cuando basta con que la respuesta correcta esté en la lista corta.
{{< /lead >}}

---

## 1. Definición
Si el modelo genera puntajes por clase y \\(S_k(x)\\) denota los k mejores candidatos para la entrada \\(x\\), entonces

$$
\mathrm{Top\text{-}k\ Accuracy} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{ y_i \in S_k(x_i) \}
$$

Equivale a `Recall@k`: contamos un acierto cuando la etiqueta verdadera está entre las k primeras predicciones.

---

## 2. Cálculo en Python 3.13
```bash
python --version  # ej.: Python 3.13.0
pip install scikit-learn
```

```python
import numpy as np
from sklearn.metrics import top_k_accuracy_score

proba = model.predict_proba(X_test)  # shape: (n_samples, n_classes)
top3 = top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)
top5 = top_k_accuracy_score(y_test, proba, k=5, labels=model.classes_)

print("Top-3 Accuracy:", round(top3, 3))
print("Top-5 Accuracy:", round(top5, 3))
```

`top_k_accuracy_score` recibe probabilidades por clase y devuelve la métrica para el `k` especificado. Si pasamos `labels`, la función es robusta ante diferencias en el orden de las clases.

---

## 3. Aspectos de diseño
- **Elegir k**: Ajusta el valor al número de candidatos que tu interfaz o producto puede mostrar.
- **Empates**: Cuando hay puntajes idénticos, el ranking puede volverse inestable; usa ordenamientos estables o desempates numéricos.
- **Múltiples respuestas correctas**: En tareas multietiqueta conviene combinarla con Precision@k y Recall@k.

---

## 4. Casos prácticos
- **Sistemas de recomendación**: Verifica si el ítem que el usuario elige realmente aparece en la lista sugerida.
- **Clasificación de imágenes**: En datasets con muchas clases (por ejemplo, ImageNet) es habitual reportar Top-5 Accuracy.
- **Búsqueda de información**: Evalúa si el documento relevante aparece dentro de los 10 primeros resultados.

---

## 5. Comparación con otras métricas de ranking
| Métrica          | Qué captura                         | Cuándo se usa                                    |
| ---------------- | ----------------------------------- | ------------------------------------------------ |
| Top-k Accuracy   | Si la etiqueta correcta está en k   | Cuando basta con que la respuesta esté en la lista |
| NDCG             | Relevancia ponderada por posición   | Importa la posición exacta dentro del ranking    |
| MAP              | Promedio de precision@k             | Rankings con múltiples elementos relevantes      |
| Hit Rate         | Sinónimo de Top-k Accuracy          | Terminología habitual en recomendadores          |

---

## Conclusiones
- Top-k Accuracy verifica si la respuesta correcta figura entre los candidatos; es sencilla y muy informativa.
- `top_k_accuracy_score` de scikit-learn permite evaluar distintos valores de k con poco esfuerzo.
- Complétala con NDCG o MAP para tener en cuenta la calidad del orden y la presencia de múltiples respuestas relevantes.
