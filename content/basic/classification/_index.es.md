---
title: Clasificación lineal
weight: 3
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.2 </b>"
---

{{% summary %}}
- Los clasificadores lineales separan clases con hiperplanos, ofreciendo interpretabilidad y entrenamiento rápido.
- Perceptrón, regresión logística, SVM y métodos afines comparten ideas que conviene comparar.
- Los conceptos aprendidos conectan con regularización, kernels y aprendizaje basado en distancias.
{{% /summary %}}

# Clasificación lineal

## Intuición
Los clasificadores lineales colocan un hiperplano en el espacio de características y asignan etiquetas según el lado en el que caiga la muestra. Desde el perceptrón sencillo, pasando por la regresión logística probabilística, hasta las SVM basadas en márgenes, la diferencia clave reside en cómo se elige el hiperplano.

## Formulación matemática
Un clasificador lineal genérico predice \(\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)\). Cada método entrena \(\mathbf{w}\) de forma distinta; las variantes probabilísticas aplican una sigmoide o un softmax al puntaje lineal. Con el truco del kernel basta con reemplazar el producto interno por un kernel para extender la idea a fronteras no lineales.

## Experimentos con Python
En este capítulo recorreremos ejemplos prácticos con Python y scikit-learn:

- Perceptrón: reglas de actualización y fronteras de decisión del clasificador lineal más simple.
- Regresión logística: clasificación binaria probabilística con entropía cruzada.
- Regresión softmax: extensión multiclase que entrega todas las probabilidades.
- Análisis discriminante lineal (LDA): direcciones que maximizan la separabilidad.
- Máquinas de vectores de soporte (SVM): maximización del margen y kernel trick.
- Naive Bayes: clasificación rápida con independencia condicional.
- k vecinos más cercanos: aprendizaje perezoso basado en distancias.

Puedes ejecutar los fragmentos de código y experimentar para observar cómo se comportan las fronteras de decisión y las probabilidades.

## Referencias
{{% references %}}
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
{{% /references %}}
