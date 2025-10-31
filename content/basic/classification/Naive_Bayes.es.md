---
title: "Naive Bayes"
pre: "2.2.6 "
weight: 6
title_suffix: "Inferencia rápida con independencia condicional"
---

{{% summary %}}
- Naive Bayes asume independencia condicional entre características y combina la probabilidad a priori con la verosimilitud mediante el teorema de Bayes.
- El entrenamiento y la inferencia son muy rápidos, lo que lo vuelve una potente línea base para datos dispersos y de alta dimensión como texto o spam.
- El suavizado de Laplace y las características TF-IDF ayudan frente a palabras no vistas y diferencias de frecuencia.
- Cuando la suposición de independencia es demasiado fuerte, conviene aplicar selección de características o ensamblarlo con otros modelos.
{{% /summary %}}

## Intuición
El teorema de Bayes afirma que “prior × verosimilitud ∝ posterior”. Si las características son condicionalmente independientes, la verosimilitud se factoriza como el producto de probabilidades individuales. Naive Bayes aprovecha esta aproximación y ofrece estimaciones sólidas incluso con pocos datos de entrenamiento.

## Formulación matemática
Para una clase \\(y\\) y un vector de características \\(\mathbf{x} = (x_1, \ldots, x_d)\\),

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y).
$$

Existen distintas variantes según el tipo de datos: el modelo multinomial para frecuencias de palabras, el bernoulli para presencia/ausencia y el gaussiano para valores continuos.

## Experimentos con Python
El ejemplo siguiente entrena un clasificador Naive Bayes multinomial sobre un subconjunto del conjunto 20 Newsgroups usando TF-IDF. Aun con miles de características el entrenamiento es veloz, y el informe de clasificación resume el desempeño.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

categories = ["rec.autos", "sci.space", "talk.politics.guns"]
train = fetch_20newsgroups(subset="train", categories=categories, remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", categories=categories, remove=("headers", "footers", "quotes"))

model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB(alpha=0.5))
model.fit(train.data, train.target)
pred = model.predict(test.data)

print(classification_report(test.target, pred, target_names=categories))
```

## Referencias
{{% references %}}
<li>Manning, C. D., Raghavan, P., &amp; Schütze, H. (2008). <i>Introduction to Information Retrieval</i>. Cambridge University Press.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
