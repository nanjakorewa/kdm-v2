---

title: "Log Loss | Medir la calidad de las probabilidades"

linkTitle: "Log Loss"

seo_title: "Log Loss | Medir la calidad de las probabilidades"

pre: "4.3.4 "

weight: 4

---



{{< lead >}}

El Log Loss (o cross-entropy) penaliza con fuerza las probabilidades erróneas. Es el indicador preferido cuando dependemos de probabilidades bien calibradas. Veamos cómo calcularlo en Python 3.13 y qué nos dicen las curvas de penalización.

{{< /lead >}}



---



## 1. Definición



Para una clasificación binaria se expresa como





\mathrm{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \left[y_i \log(p_i) + (1 - y_i) \log(1 - p_i)\right],





donde \(p_i\) es la probabilidad predicha para la clase positiva y \(y_i\) es la etiqueta real (0 o 1). En multiclase se extiende sumando las probabilidades de cada clase con su etiqueta one-hot.



---



## 2. Cálculo en Python 3.13



```bash

python --version        # p. ej. Python 3.13.0

pip install scikit-learn matplotlib

```



```python

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, log_loss

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, stratify=y, random_state=42

)



model = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, solver="lbfgs"),

)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)



print(classification_report(y_test, model.predict(X_test), digits=3))

print("Log Loss:", log_loss(y_test, proba))

```



Basta pasar el array de probabilidades a log_loss.



---



## 3. Curvas de penalización



{{< figure src="/images/eval/classification/log-loss/log_loss_curves.png" alt="Penalización del Log Loss" caption="Las probabilidades cercanas a la clase incorrecta generan un coste muy alto." >}}



- Otorgar poca probabilidad a un positivo real (p. ej. 0.1) resulta en una penalización enorme.

- Predecir 0.5 sistemáticamente también se castiga: el modelo no está aprendiendo nada útil.



---



## 4. Cuándo usar Log Loss



- **Comprobar la calibración** – después de aplicar Platt scaling o isotonic regression, verifica que el Log Loss disminuya.

- **Competencias y comparativas** – en plataformas como Kaggle el Log Loss es un criterio habitual.

- **Comparación independiente del umbral** – a diferencia de la exactitud, evalúa toda la distribución de probabilidades.



log_loss permite configurar parámetros como labels, eps y 

ormalize para controlar estabilidad numérica y soportes incompletos.



---



## Resumen



- Log Loss mide cuánto se apartan las probabilidades del resultado real; cuanto menor, mejor.

- Con scikit-learn en Python 3.13 basta con predict_proba y log_loss.

- Complementa la evaluación con métricas como ROC-AUC o las curvas PR para analizar discriminación y calibración.

---

