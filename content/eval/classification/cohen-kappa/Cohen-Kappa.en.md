---
title: "Cohen's Kappa: Agreement Beyond Chance"
linkTitle: "Cohen's Kappa"
seo_title: "Cohen's Kappa | Measuring Agreement Beyond Chance"
title_suffix: "Measuring agreement beyond chance"
pre: "4.3.11 "
weight: 11
---

{{< lead >}}
Cohen's Kappa discounts the agreement we would obtain by chance and is widely used to evaluate both annotation quality and classification models on imbalanced data. It reveals whether a model truly understands the task instead of exploiting skewed class distributions.
{{< /lead >}}

---

## 1. Definition
Let \\(p_o\\) be the observed agreement and \\(p_e\\) the expected agreement by random chance. The coefficient is

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- \\(\kappa = 1\\): perfect agreement  
- \\(\kappa = 0\\): no better than chance  
- \\(\kappa < 0\\): worse than chance

---

## 2. Computing in Python 3.13
```bash
python --version  # e.g. Python 3.13.0
pip install scikit-learn
```

```python
from sklearn.metrics import cohen_kappa_score, confusion_matrix

print("Cohen's Kappa:", cohen_kappa_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

The function works for multi-class classification. Use `weights="quadratic"` to compute the weighted version for ordinal labels.

---

## 3. Interpretation guide
Landis & Koch (1977) proposed the following rule of thumb. Adapt the thresholds to the expectations of your domain.

| κ       | Interpretation        |
| ------- | --------------------- |
| < 0     | Poor agreement        |
| 0.0–0.2 | Slight agreement      |
| 0.2–0.4 | Fair agreement        |
| 0.4–0.6 | Moderate agreement    |
| 0.6–0.8 | Substantial agreement |
| 0.8–1.0 | Almost perfect        |

---

## 4. Benefits for model evaluation
- **Robust to imbalance**: Models that simply predict the majority class receive a low κ, counteracting overly optimistic Accuracy.
- **Annotation quality checks**: Compare model predictions with human labels or agreement between annotators objectively.
- **Weighted Kappa**: For ordinal outcomes (e.g., 5-point ratings) account for how far away incorrect predictions fall.

---

## 5. Practical tips
- A high Accuracy but low κ signals that the model may rely on chance agreement. Inspect the confusion matrix for failure patterns.
- Regulated industries sometimes require κ-based reporting—document the computation pipeline for audits.
- Use κ when auditing training labels to identify annotators or subsets with inconsistent decisions.

---

## Key takeaways
- Cohen's Kappa subtracts chance agreement, making it suitable for imbalanced problems and annotation benchmarking.
- `cohen_kappa_score` in scikit-learn provides both standard and weighted versions with minimal code.
- Combine κ with Accuracy, F1, and other metrics for a well-rounded assessment of model performance and labeling quality.
