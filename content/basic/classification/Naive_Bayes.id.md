---
title: "Naive Bayes"
pre: "2.2.6 "
weight: 6
title_suffix: "Inferensi cepat dengan asumsi independensi"
---

{{% summary %}}
- Naive Bayes mengasumsikan fitur saling independen secara kondisional dan menggabungkan prior dengan likelihood menggunakan teorema Bayes.
- Pelatihan dan inferensinya sangat cepat, menjadikannya baseline kuat untuk data berdimensi tinggi dan jarang seperti teks atau deteksi spam.
- Pelapisan Laplace dan fitur TF-IDF membantu menghadapi kata yang belum pernah muncul serta ketimpangan frekuensi.
- Jika asumsi independensi terlalu kuat, gunakan seleksi fitur atau kombinasikan Naive Bayes dengan model lain.
{{% /summary %}}

## Intuisi
Teorema Bayes menyatakan bahwa “prior × likelihood ∝ posterior”. Jika fitur independen secara kondisional, likelihood dapat dipisah menjadi hasil kali probabilitas per fitur. Naive Bayes memanfaatkan pendekatan ini sehingga tetap stabil meski data latih sedikit.

## Formulasi matematis
Untuk kelas \\(y\\) dan vektor fitur \\(\mathbf{x} = (x_1, \ldots, x_d)\\),

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y).
$$

Model likelihood berbeda cocok untuk jenis data berbeda: multinomial untuk frekuensi kata, bernoulli untuk hadir/tidak hadir, dan gaussian untuk nilai kontinu.

## Eksperimen dengan Python
Contoh berikut melatih Naive Bayes multinomial pada subset data 20 Newsgroups dengan fitur TF-IDF. Walaupun jumlah fiturnya besar, pelatihan tetap cepat dan laporan klasifikasi merangkum performanya.

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

## Referensi
{{% references %}}
<li>Manning, C. D., Raghavan, P., &amp; Schütze, H. (2008). <i>Introduction to Information Retrieval</i>. Cambridge University Press.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
