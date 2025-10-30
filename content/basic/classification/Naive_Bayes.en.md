---
title: "Naive Bayes"
pre: "2.2.6 "
weight: 6
title_suffix: "Fast inference with conditional independence"
---

{{% summary %}}
- Naive Bayes assumes conditional independence between features and combines prior probabilities with likelihoods via Bayes’ rule.
- Training and inference are extremely fast, making it a strong baseline for high-dimensional sparse data such as text or spam filtering.
- Laplace smoothing and TF-IDF features mitigate issues with unseen words and frequency imbalance.
- When the independence assumption is too strong, consider feature selection or ensembling Naive Bayes with other models.
{{% /summary %}}

## Intuition
Bayes’ rule states that “prior × likelihood ∝ posterior”. If features are conditionally independent, the likelihood factorises into a product of per-feature probabilities. Naive Bayes leverages this approximation, delivering robust estimates even with small training sets.

## Mathematical formulation
For class \(y\) and features \(\mathbf{x} = (x_1, \ldots, x_d)\),

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y).
$$

Different likelihood models suit different data types: the multinomial model for word counts, the Bernoulli model for binary presence/absence, and Gaussian Naive Bayes for continuous values.

## Experiments with Python
The snippet below trains a multinomial Naive Bayes classifier on a subset of the 20 Newsgroups data set, using TF-IDF features. Even with thousands of features the model trains quickly, and the classification report summarises performance.

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

## References
{{% references %}}
<li>Manning, C. D., Raghavan, P., &amp; Schütze, H. (2008). <i>Introduction to Information Retrieval</i>. Cambridge University Press.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
