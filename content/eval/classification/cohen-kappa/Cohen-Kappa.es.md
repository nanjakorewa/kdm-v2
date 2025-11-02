---
title: "Kappa de Cohen: acuerdo más allá del azar"
linkTitle: "Kappa de Cohen"
seo_title: "Kappa de Cohen | Medir el acuerdo más allá del azar"
title_suffix: "Medir el acuerdo más allá del azar"
pre: "4.3.11 "
weight: 11
---

{{< lead >}}
El coeficiente Kappa de Cohen descuenta el acuerdo que se obtendría por azar y se usa para evaluar tanto la calidad de las anotaciones como modelos de clasificación con clases desbalanceadas. Permite saber si el modelo entiende realmente la tarea o solo aprovecha la distribución sesgada.
{{< /lead >}}

---

## 1. Definición
Sea \\(p_o\\) el acuerdo observado y \\(p_e\\) el acuerdo esperado por azar. Entonces

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- \\(\kappa = 1\\): acuerdo perfecto  
- \\(\kappa = 0\\): equivalente al azar  
- \\(\kappa < 0\\): peor que el azar

---

## 2. Cálculo en Python 3.13
```bash
python --version  # ej.: Python 3.13.0
pip install scikit-learn
```

```python
from sklearn.metrics import cohen_kappa_score, confusion_matrix

print("Kappa de Cohen:", cohen_kappa_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

La función funciona para clasificación multiclase. Con `weights="quadratic"` obtenemos la versión ponderada para etiquetas ordinales.

---

## 3. Guía de interpretación
Landis y Koch (1977) propusieron la siguiente tabla orientativa. Ajusta los umbrales según lo que requiera tu dominio.

| κ       | Interpretación         |
| ------- | ---------------------- |
| < 0     | Acuerdo muy bajo       |
| 0.0–0.2 | Acuerdo ligero         |
| 0.2–0.4 | Acuerdo aceptable      |
| 0.4–0.6 | Acuerdo moderado       |
| 0.6–0.8 | Acuerdo considerable   |
| 0.8–1.0 | Casi perfecto          |

---

## 4. Ventajas para evaluar modelos
- **Robusto al desbalance**: Un modelo que solo predice la clase mayoritaria obtiene un κ bajo, evitando que Accuracy sea engañoso.
- **Control de calidad de anotaciones**: Compara predicciones del modelo con etiquetas humanas o entre anotadores de forma objetiva.
- **Kappa ponderado**: En resultados ordinales (por ejemplo, puntuaciones de 5 niveles) tiene en cuenta qué tan lejos cae la predicción incorrecta.

---

## 5. Recomendaciones prácticas
- Si Accuracy es alta pero κ es bajo, probablemente el modelo dependa del azar. Analiza la matriz de confusión para encontrar patrones.
- En sectores regulados se puede exigir reportar κ; documenta el proceso de cálculo para auditorías.
- Úsalo al auditar el etiquetado para identificar anotadores o subconjuntos con decisiones inconsistentes.

---

## Conclusiones clave
- El Kappa de Cohen descuenta el acuerdo casual, por lo que es útil en problemas desbalanceados y en la comprobación de anotaciones.
- `cohen_kappa_score` de scikit-learn ofrece versiones estándar y ponderadas con muy poco código.
- Combina κ con Accuracy, F1 y otras métricas para evaluar el desempeño del modelo y la calidad del etiquetado de forma integral.
