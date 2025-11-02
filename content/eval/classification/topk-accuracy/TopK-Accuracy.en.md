---
title: "Top-k Accuracy and Recall@k"
linkTitle: "Top-k Accuracy"
seo_title: "Top-k Accuracy & Recall@k | Evaluating candidate lists"
title_suffix: "Evaluating whether the correct label appears among the top candidates"
pre: "4.3.12 "
weight: 12
---

{{< lead >}}
Top-k Accuracy (also known as Recall@k) measures how often the correct label appears among the top k candidates returned by a model. It is the go-to metric for multi-class and recommendation scenarios where surfacing the correct answer in a shortlist is all that matters.
{{< /lead >}}

---

## 1. Definition
If the model produces scores per class and \\(S_k(x)\\) denotes the top-k candidates for input \\(x\\), then

$$
\mathrm{Top\text{-}k\ Accuracy} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{ y_i \in S_k(x_i) \}
$$

This is equivalent to `Recall@k`: a hit is recorded when the ground-truth label is among the top k predictions.

---

## 2. Computing in Python 3.13
```bash
python --version  # e.g. Python 3.13.0
pip install scikit-learn
```

```python
import numpy as np
from sklearn.metrics import top_k_accuracy_score

proba = model.predict_proba(X_test)  # shape: (n_samples, n_classes)
top3 = top_k_accuracy_score(y_test, proba, k=3, labels=model.classes_)
top5 = top_k_accuracy_score(y_test, proba, k=5, labels=model.classes_)

print("Top-3 Accuracy:", round(top3, 3))
print("Top-5 Accuracy:", round(top5, 3))
```

`top_k_accuracy_score` consumes class probabilities and returns the metric for any value of `k`. Passing the `labels` array makes the computation robust to class-order differences.

---

## 3. Design considerations
- **Choosing k**: Match the shortlists your UI or product can display.
- **Ties**: Equal scores can make rankings unstable; use stable sorting or add tie-breaker decimals when needed.
- **Multiple ground truths**: For genuine multi-label tasks, complement Top-k Accuracy with Precision@k and Recall@k.

---

## 4. Practical applications
- **Recommendation systems**: Check whether the items users eventually choose appear in the suggested slate.
- **Image classification**: Large-class datasets (e.g., ImageNet) traditionally report Top-5 Accuracy.
- **Search & retrieval**: Evaluate if the relevant document shows up within the top 10 results.

---

## 5. Comparison with other ranking metrics
| Metric           | What it captures                  | Typical use case                               |
| ---------------- | --------------------------------- | ---------------------------------------------- |
| Top-k Accuracy   | Whether the correct label is in k | Shortlists where any hit counts                |
| NDCG             | Rank-weighted relevance           | When placement near the top carries more value |
| MAP              | Mean precision over ranks         | Rankings with multiple relevant items          |
| Hit Rate         | Synonymous with Top-k Accuracy    | Common in recommender-system literature        |

---

## Takeaways
- Top-k Accuracy checks if the correct answer appears among the candidates—a simple yet powerful signal.
- scikit-learn’s `top_k_accuracy_score` lets you evaluate multiple k values with minimal effort.
- Combine it with NDCG or MAP to account for rank quality and cases with several relevant options.
