---

title: "Average Precision (AP) | Evaluar la curva precision–recall"

linkTitle: "Average Precision"

seo_title: "Average Precision (AP) | Evaluar la curva precision–recall"

pre: "4.3.9 "

weight: 9

---



{{< lead >}}

El average precision (AP) resume la curva precision–recall ponderando la precisión por el incremento en recall. Permite analizar el comportamiento de un modelo a lo largo de todos los umbrales, especialmente cuando las clases están desbalanceadas. Veamos cómo calcularlo en Python 3.13 y cómo se complementa con F1 y ROC-AUC.

{{< /lead >}}



---



## 1. Definición



Si la curva PR está formada por puntos \((R_n, P_n)\), el average precision se define como:





\mathrm{AP} = \sum_{n}(R_n - R_{n-1}) P_n





El incremento en recall actúa como peso, por lo que AP refleja la precisión media cuando desplazamos el umbral de decisión.



---



## 2. Cálculo en Python 3.13



```bash

python --version        # p. ej. Python 3.13.0

pip install scikit-learn matplotlib

```



Reutilizando las probabilidades proba del ejemplo precision–recall, podemos obtener AP con unas pocas llamadas de scikit-learn:



```python

from sklearn.metrics import precision_recall_curve, average_precision_score



precision, recall, thresholds = precision_recall_curve(y_test, proba)

ap = average_precision_score(y_test, proba)

print(f"Average Precision: {ap:.3f}")

```



La curva PR correspondiente es la misma pr_curve.png generada anteriormente.



{{< figure src="/images/eval/classification/precision-recall/pr_curve.png" alt="Curva precision–recall" caption="AP integra la curva PR ponderando la precisión con los incrementos de recall." >}}



---



## 3. Diferencia entre AP y PR-AUC



- verage_precision_score implementa la integración escalonada utilizada en recuperación de información.

- sklearn.metrics.auc(recall, precision) usa la regla del trapecio y produce el PR-AUC clásico.

- AP suele ser más estable en datos desbalanceados porque enfatiza los tramos donde el recall realmente aumenta.



---



## 4. Consejos prácticos



- **Selección de umbral** – Un AP alto sugiere que el modelo mantiene buena precisión en un rango amplio de recalls.

- **Ranking y búsqueda** – En tareas de recomendación se reporta el mean average precision (MAP), promedio de AP por consulta.

- **Complemento al F1** – F1 describe un único punto de operación, mientras que AP evalúa todo el espectro de umbrales.



---



## Resumen



- Average Precision evalúa la curva precision–recall completa y es ideal para datasets desbalanceados.

- Python 3.13 con scikit-learn permite calcularlo fácilmente mediante verage_precision_score.

- Combínalo con F1, ROC-AUC y la visualización PR para comparar modelos y definir umbrales operativos.

---

