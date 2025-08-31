---
title: "Clasificación Softmax"
pre: "2.2.2 "
weight: 2
title_suffix: "Multiclase en Python"
---

<div class="pagetop-box">
  <p>La <b>clasificación softmax</b> generaliza la regresión logística a <b>múltiples clases</b>. Con dos clases equivale a logística; con tres o más devuelve una probabilidad válida sobre clases.</p>
</div>

---

## 1. Función softmax

Dado $z=(z_1,\dots,z_K)$:

$$
\mathrm{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)} \quad (i=1,\dots,K)
$$

- Salida en <b>[0,1]</b>  
- Suma <b>1</b> entre clases  
- Interpretación como probabilidades

---

## 2. Modelo

Para $x$, el score de la clase $k$:

$$
z_k = w_k^\top x + b_k
$$

Probabilidad softmax:

$$
P(y=k\mid x) = \frac{\exp(w_k^\top x + b_k)}{\sum_{j=1}^{K} \exp(w_j^\top x + b_j)}
$$

Se entrena con entropía cruzada multinomial.

---

## 3. Probar en Python

Use `LogisticRegression` con `multi_class="multinomial"`.

{{% notice document %}}
- [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_classes=3,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
clf.fit(X, y)

x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.title("Clasificación softmax (logística multinomial)")
plt.show()
```

---

## 4. Notas

- Probabilidades adecuadas para decisiones  
- Fronteras lineales en el espacio de características  
- Escalado de variables recomendado; añadir no linealidades si hace falta

---

## 5. Usos típicos

- <b>Texto</b> (temas multiclase)  
- <b>Imágenes</b> (dígitos 0–9, etc.)  
- <b>Intención de usuario</b> (una entre varias)

---

## Resumen

- Softmax generaliza logística a multiclase.  
- Devuelve una distribución de probabilidad válida.  
- En scikit-learn, `multi_class="multinomial"`.  
- Base simple y fuerte para muchos problemas.

---

