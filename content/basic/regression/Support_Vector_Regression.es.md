---
title: "Regresión con máquinas de soporte vectorial (SVR)"
pre: "2.1.10 "
weight: 10
title_suffix: "Predicciones robustas con un tubo ε-insensible"
---

{{% summary %}}
- SVR extiende las SVM a regresión, tratando los errores dentro de un tubo ε-insensible como cero para reducir el impacto de los valores atípicos.
- Gracias a los kernels puede capturar relaciones no lineales mientras mantiene el modelo compacto mediante vectores de soporte.
- Los hiperparámetros `C`, `epsilon` y `gamma` controlan el equilibrio entre generalización y suavidad.
- La estandarización de las características es esencial; encapsular el preprocesado y el aprendizaje en un pipeline garantiza transformaciones consistentes.
{{% /summary %}}

## Intuición
SVR ajusta una función rodeada por un tubo de ancho ε: los puntos dentro del tubo no incurren en pérdida, mientras que los que lo atraviesan pagan una penalización. Solo los puntos que tocan o salen del tubo —los vectores de soporte— influyen en el modelo final, lo que produce aproximaciones suaves y resistentes al ruido.

## Formulación matemática
El problema de optimización es

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
$$

sujeto a

$$
\begin{aligned}
y_i - (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) &\le \epsilon + \xi_i, \\
(\mathbf{w}^\top \phi(\mathbf{x}_i) + b) - y_i &\le \epsilon + \xi_i^*, \\
\xi_i, \xi_i^* &\ge 0,
\end{aligned}
$$

donde \(\phi\) es la transformación inducida por el kernel elegido. Resolver el dual revela los vectores de soporte y sus coeficientes.

## Experimentos con Python
Ejemplo de SVR combinado con `StandardScaler` dentro de un pipeline.

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
)

svr.fit(X_train, y_train)
pred = svr.predict(X_test)
```

### Interpretación de los resultados
- El pipeline escala los datos de entrenamiento con su media y varianza y aplica la misma transformación al conjunto de prueba.
- `pred` contiene las predicciones; ajustar `epsilon` y `C` modifica el equilibrio entre sobreajuste y subajuste.
- Aumentar `gamma` en el kernel RBF enfatiza patrones locales, mientras que valores pequeños producen funciones más suaves.

## Referencias
{{% references %}}
<li>Smola, A. J., &amp; Schölkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199–222.</li>
<li>Vapnik, V. (1995). <i>The Nature of Statistical Learning Theory</i>. Springer.</li>
{{% /references %}}
