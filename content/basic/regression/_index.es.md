---
title: Regresión lineal
weight: 1
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.1 </b>"
---

{{% summary %}}
- La regresión lineal captura la relación lineal entre variables de entrada y salida y es la base para tareas de predicción e interpretación.
- Al combinar extensiones como regularización, métodos robustos o reducción de dimensionalidad, se adapta a distintos tipos de datos.
- Cada página del capítulo sigue el mismo recorrido Eresumen, intuición, fórmulas, experimentos en Python y referencias Epara pasar de los fundamentos a aplicaciones prácticas.
{{% /summary %}}

# Regresión lineal

## Intuición
La regresión lineal responde a la pregunta “¿cuánto cambia la salida cuando la entrada aumenta una unidad?” Gracias a que los coeficientes son fáciles de interpretar y el cálculo es rápido, suele ser el primer modelo que se prueba en un proyecto de aprendizaje automático.

## Formulación matemática
El método de mínimos cuadrados ordinarios estima los coeficientes minimizando la suma de los errores al cuadrado entre observaciones y predicciones. Con matriz \(\mathbf{X}\) y vector \(\mathbf{y}\),

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
$$

En el resto del capítulo ampliamos este marco con regularización, técnicas robustas y otras variantes.

## Experimentos con Python
Cada página incluye ejemplos ejecutables con `scikit-learn` que abarcan:

- Fundamentos: mínimos cuadrados, Ridge, Lasso, regresión robusta  
- Mayor expresividad: regresión polinómica, Elastic Net, regresión de cuantiles, regresión bayesiana  
- Reducción y esparsidad: regresión por componentes principales, PLS, mínimos cuadrados ponderados, Orthogonal Matching Pursuit, SVR, entre otros

Ejecútalos para observar cómo se comportan los modelos con datos sintéticos y reales.

## Referencias
{{% references %}}
<li>Draper, N. R., &amp; Smith, H. (1998). <i>Applied Regression Analysis</i> (3rd ed.). John Wiley &amp; Sons.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
