---

title: "F1 Score | La media armónica de precisión y exhaustividad"

linkTitle: "F1 Score"

seo_title: "F1 Score | La media armónica de precisión y exhaustividad"

pre: "4.3.8 "

weight: 8

---



{{< lead >}}

El F1 score combina precisión y exhaustividad mediante la media armónica. Es ideal cuando las falsas alarmas y los falsos negativos son igual de preocupantes. A continuación lo calculamos en Python 3.13, observamos cómo cambia con el umbral y revisamos cuándo conviene usar otras variantes Fβ.

{{< /lead >}}



---



## 1. Definición



Con precisión \(P\) y exhaustividad \(R\), el F1 se define como





F_1 = 2 \cdot \frac{P \cdot R}{P + R}.





La versión general Fβ permite ponderar más el recall (\(\beta > 1\)) o la precisión (\(\beta < 1\)):





F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}.





---



## 2. Cálculo en Python 3.13



```bash

python --version        # p. ej. Python 3.13.0

pip install scikit-learn matplotlib

```



```python

import numpy as np

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, f1_score, fbeta_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



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



model = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, class_weight="balanced"),

)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(classification_report(y_test, y_pred, digits=3))

print("F1:", f1_score(y_test, y_pred))

print("F0.5:", fbeta_score(y_test, y_pred, beta=0.5))

print("F2:", fbeta_score(y_test, y_pred, beta=2.0))

```



classification_report resume precisión, exhaustividad y F1 por clase.



---



## 3. Cómo varía el F1 con el umbral



Si disponemos de probabilidades, podemos representar cómo evoluciona el F1 al mover el umbral de decisión.



```python

from sklearn.metrics import f1_score, precision_recall_curve



proba = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, proba)

thresholds = np.append(thresholds, 1.0)

f1_scores = [

    f1_score(y_test, (proba >= t).astype(int))

    for t in thresholds

]

```



{{< figure src="/images/eval/classification/f1-score/f1_vs_threshold.png" alt="F1 según el umbral" caption="Localiza el pico para elegir el umbral con mejor equilibrio entre precisión y recall." >}}



- El máximo indica el mejor compromiso entre precisión y recall cuando ambos son igual de relevantes.

- Usa F0.5 (prioriza precisión) o F2 (prioriza recall) si las necesidades del negocio así lo requieren.



---



## 4. Estrategias de promedio en escenarios multiclase



El parámetro verage de scikit-learn permite agrupar F1 en problemas multiclase o multilabel:



- macro — media simple de los F1 por clase.

- weighted — media ponderada por el soporte de cada clase.

- micro — se calcula sobre la confusión global; puede ocultar el comportamiento de clases minoritarias.



```python

from sklearn.metrics import f1_score



f1_macro = f1_score(y_test, y_pred, average="macro")

f1_weighted = f1_score(y_test, y_pred, average="weighted")

```



En multietiqueta, verage="samples" devuelve la media por muestra.



---



## Resumen



- F1 equilibra precisión y exhaustividad; conviene graficarlo frente al umbral para elegir el punto de operación.

- Los Fβ permiten inclinar la balanza hacia recall (β>1) o precisión (β<1) según el contexto.

- En multiclase, indica la estrategia de promedio y revisa F1 junto con Precision/Recall y la curva PR para comprender el comportamiento del modelo.

---

