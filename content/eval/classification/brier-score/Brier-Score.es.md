---

title: "Brier Score | Medir la calibración de probabilidades"

linkTitle: "Brier Score"

seo_title: "Brier Score | Medir la calibración de probabilidades"

pre: "4.3.10 "

weight: 10

---



{{< lead >}}

El Brier Score es el error cuadrático entre las probabilidades predichas y los resultados observados (0/1). Evalúa lo bien calibrado que está un clasificador, algo crítico cuando trabajamos con probabilidades, como en previsión meteorológica o planificación de demanda. A continuación lo calculamos en Python 3.13 y visualizamos el diagrama de fiabilidad.

{{< /lead >}}



---



## 1. Definición



En una clasificación binaria el Brier Score se expresa como





\mathrm{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2,





donde \(p_i\) es la probabilidad predicha de la clase positiva y \(y_i\) es la etiqueta real (0 o 1). En problemas multiclase se calcula el error cuadrático por clase y se promedia.



---



## 2. Implementación y visualización en Python 3.13



```bash

python --version        # p. ej. Python 3.13.0

pip install scikit-learn matplotlib

```



El siguiente script entrena una regresión logística sobre el conjunto Breast Cancer, imprime el Brier Score y dibuja un diagrama de fiabilidad. La figura se guarda en static/images/eval/classification/brier-score/reliability_curve.png, lista para regenerarse con generate_eval_assets.py.



```python

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.calibration import CalibrationDisplay

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import brier_score_loss

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, stratify=y, random_state=42

)



pipeline = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, solver="lbfgs"),

)

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_test)[:, 1]



score = brier_score_loss(y_test, proba)

print(f"Brier Score: {score:.3f}")



fig, ax = plt.subplots(figsize=(5, 5))

CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, ax=ax)

ax.set_title("Diagrama de fiabilidad (Breast Cancer Dataset)")

fig.tight_layout()

output_dir = Path("static/images/eval/classification/brier-score")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "reliability_curve.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/brier-score/reliability_curve.png" alt="Diagrama de fiabilidad" caption="Cuanto más se aleja de la diagonal, más sobre- o infra-confianza muestran las probabilidades." >}}



---



## 3. Interpretación del puntaje



- Probabilidades perfectamente calibradas producen **0**.

- Un modelo que siempre devuelve 0.5 en un conjunto balanceado se queda en **0.25**.

- Cuanto más pequeño sea el valor, mejor: el error cuadrático penaliza especialmente las probabilidades alejadas del resultado real.



---



## 4. Diagnóstico con diagramas de fiabilidad



El diagrama de fiabilidad agrupa las predicciones por bins, coloca la probabilidad media predicha en el eje x y la tasa real de positivos en el eje y.



- Puntos **por debajo** de la diagonal → el modelo es sobreconfidente (probabilidades demasiado altas).

- Puntos **por encima** de la diagonal → el modelo es subconfidente.

- Tras aplicar técnicas de calibración (Platt scaling, isotonic regression, etc.), vuelva a calcular el Brier Score y el gráfico para confirmar la mejora.



---



## Resumen



- El Brier Score mide el error cuadrático medio de las probabilidades; valores menores indican mejor calibración.

- En Python 3.13, rier_score_loss más el diagrama de fiabilidad proporcionan una comprobación rápida.

- Combínelo con ROC-AUC y métricas Precision/Recall para evaluar tanto la capacidad de ranking como la calidad de la probabilidad.

---

