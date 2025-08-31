---
title: "Random Forest"
pre: "2.4.1 "
weight: 1
title_suffix: "Intuición, fórmulas y práctica"
---

{{% youtube "ewvjQMj8nA8" %}}

<div class="pagetop-box">
  <p><b>Random Forest</b> entrena muchos árboles con <b>bootstrap</b> y <b>submuestreo de características</b>. Predice por <b>votación</b> (clasificación) o <b>promedio</b> (regresión), reduciendo la varianza y aumentando la robustez.</p>
</div>

{{% notice document %}}
- [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
{{% /notice %}}

## Cómo funciona (fórmulas)
- Árbol \(h_b(x)\) por muestra bootstrap \(\mathcal{D}_b\), para \(b=1,\dots,B\).
- Predicción:
  - Clasificación: \(\hat y = \operatorname*{arg\,max}_c \sum_{b=1}^B \mathbf{1}[h_b(x)=c]\)
  - Regresión: \(\hat y = \tfrac{1}{B}\sum_{b=1}^B h_b(x)\)

Ejemplo (Gini): \(\mathrm{Gini}(S)=1-\sum_c p(c\mid S)^2\)

---

## Entrenar y medir ROC-AUC
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

n_features = 20
X, y = make_classification(
    n_samples=2500,
    n_features=n_features,
    n_informative=10,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=4,
    random_state=777,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=777
)

model = RandomForestClassifier(
    n_estimators=50, max_depth=3, random_state=777, bootstrap=True, oob_score=True
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rf_score = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC (test) = {rf_score}")
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_6_0.png)

---

## Desempeño por árbol
```python
import japanize_matplotlib

estimator_scores = []
for i in range(10):
    est = model.estimators_[i]
    estimator_scores.append(roc_auc_score(y_test, est.predict(X_test)))

plt.figure(figsize=(10, 4))
bar_index = [i for i in range(len(estimator_scores))]
plt.bar(bar_index, estimator_scores)
plt.bar([10], rf_score)
plt.xticks(bar_index + [10], bar_index + ["RF"])
plt.xlabel("índice de árbol")
plt.ylabel("ROC-AUC")
plt.show()
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_6_0.png)

---

## Importancia de características

### Basada en impureza
```python
plt.figure(figsize=(10, 4))
feature_index = [i for i in range(n_features)]
plt.bar(feature_index, model.feature_importances_)
plt.xlabel("índice de característica")
plt.ylabel("importancia")
plt.show()
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_8_0.png)

### Permutation importance
{{% notice document %}}
[permutation_importance](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html)
{{% /notice %}}

```python
from sklearn.inspection import permutation_importance

p_imp = permutation_importance(
    model, X_train, y_train, n_repeats=10, random_state=77
).importances_mean

plt.figure(figsize=(10, 4))
plt.bar(feature_index, p_imp)
plt.xlabel("índice de característica")
plt.ylabel("importancia")
plt.show()
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_10_0.png)

---

## Visualizar árboles (opcional)
```python
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image, display

for i in range(3):
    try:
        est = model.estimators_[i]
        export_graphviz(
            est,
            out_file=f"tree{i}.dot",
            feature_names=[f"x{i}" for i in range(n_features)],
            class_names=["A", "B"],
            proportion=True,
            filled=True,
        )
        call(["dot", "-Tpng", f"tree{i}.dot", "-o", f"tree{i}.png", "-Gdpi=500"])
        display(Image(filename=f"tree{i}.png"))
    except Exception:
        pass
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_0.png)

---

## Consejos de hiperparámetros
- <b>n_estimators</b>: más árboles → más estable, más cómputo.
- <b>max_depth</b>: profundo → sobreajuste; poco profundo → infracceso.
- <b>max_features</b>: menos → menor correlación, más diversidad.
- <b>bootstrap</b>, <b>oob_score</b>: validación OOB opcional.

