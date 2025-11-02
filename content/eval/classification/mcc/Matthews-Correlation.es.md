---

title: "Coeficiente de correlación de Matthews (MCC) | Una métrica equilibrada"

linkTitle: "MCC"

seo_title: "Coeficiente de correlación de Matthews (MCC) | Una métrica equilibrada"

pre: "4.3.5 "

weight: 5

---



{{< lead >}}

El MCC incorpora las cuatro entradas de la matriz de confusión (TP, FP, FN, TN) en un valor único entre −1 y 1. Es mucho más resistente al desbalance de clases que la exactitud o el F1, por lo que conviene reportarlo cuando el equilibrio es clave.

{{< /lead >}}



---



## 1. Definición



En clasificación binaria:





\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}.





- **1** → predicción perfecta

- **0** → equivalente al azar

- **−1** → desacuerdo total



La versión multiclase se construye a partir de la matriz de confusión completa.



---



## 2. Cálculo con Python 3.13



```bash

python --version        # p. ej. Python 3.13.0

pip install scikit-learn matplotlib

```



```python

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import matthews_corrcoef, confusion_matrix

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



print(confusion_matrix(y_test, y_pred))

print("MCC:", matthews_corrcoef(y_test, y_pred))

```



class_weight="balanced" facilita que la clase minoritaria influya en el resultado.



---



## 3. MCC frente al umbral



{{< figure src="/images/eval/classification/mcc/mcc_vs_threshold.png" alt="MCC vs umbral" caption="Identifica el umbral donde el MCC alcanza su máximo para equilibrar todas las celdas de la matriz de confusión." >}}



A diferencia del F1, MCC incluye los verdaderos negativos. Al analizarlo a lo largo de los umbrales se detecta el punto de operación con mejor correlación.



---



## 4. Aplicaciones prácticas



- **Controlar la exactitud** – si la exactitud es alta pero el MCC bajo, alguna clase está siendo ignorada.

- **Selección de modelos** – usa make_scorer(matthews_corrcoef) en GridSearchCV para optimizarlo directamente.

- **Complementar ROC/PR** – MCC ofrece una visión global, mientras que ROC-AUC o PR-AUC se centran en el ranking y la recuperación.



---



## Resumen



- MCC ofrece una evaluación equilibrada entre −1 y 1 que considera TP, FP, FN y TN.

- En Python 3.13 basta con matthews_corrcoef; visualizarlo contra el umbral ayuda a elegir el mejor punto de operación.

- Repórtalo junto a Accuracy, F1 y las métricas PR para evitar conclusiones engañosas en datos desbalanceados.

---

