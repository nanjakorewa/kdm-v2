---
title: "Hamming Loss: Label-wise Error Rate"
linkTitle: "Hamming Loss"
seo_title: "Hamming Loss | Measuring label-wise error in multi-label tasks"
title_suffix: "Measuring label-wise error in multi-label tasks"
pre: "4.3.13 "
weight: 13
---

{{< lead >}}
Hamming Loss measures the proportion of labels that are predicted incorrectly. It is especially handy for multi-label classification where we care about the average number of labels missed per sample.
{{< /lead >}}

---

## 1. Definition
Given true label sets \\(Y_i\\), predicted label sets \\(\hat{Y}_i\\), and \\(L\\) labels in total,

$$
\mathrm{Hamming\ Loss} = \frac{1}{nL} \sum_{i=1}^n \lvert Y_i \triangle \hat{Y}_i \rvert
$$

where \\(Y \triangle \hat{Y}\\) denotes the symmetric difference (labels that appear in only one of the sets). For multi-label tasks this equals the average number of wrong labels per sample.

---

## 2. Computing in Python 3.13
```bash
python --version  # e.g. Python 3.13.0
pip install scikit-learn
```

```python
from sklearn.metrics import hamming_loss

print("Hamming Loss:", hamming_loss(y_true, y_pred))
```

Pass `y_true` and `y_pred` as 0/1 multi-label indicator arrays (the output of `MultiLabelBinarizer` works nicely).

---

## 3. Reading the score
- The closer to 0, the better. A perfect classifier yields 0.
- A value of 0.05 means “on average 5% of the labels were wrong”.
- If labels have different business impact, consider a weighted Hamming Loss.

---

## 4. Relation to other metrics
| Metric          | What it captures             | When to use it                              |
| --------------- | ---------------------------- | ------------------------------------------- |
| Exact Match     | Sample-wise perfect accuracy | Strict: requires the entire label set match |
| **Hamming Loss**| Label-wise error rate        | Track the average number of mistakes        |
| Micro F1        | Precision & recall balance   | Account for positive/negative imbalance     |
| Jaccard Index   | Set overlap                  | Evaluate the similarity of label sets       |

Hamming Loss is less strict than Exact Match and provides a smoother signal when iterating on the model.

---

## 5. Practical tips
- **Tag recommendation**: quantify how many tags per item are wrong on average.
- **Alert systems**: monitor how often multi-label alarms fire incorrectly.
- **Weighted evaluation**: apply per-label weights when the cost of mistakes varies across labels.

---

## Takeaways
- Hamming Loss captures label-wise error rates, making it ideal for monitoring multi-label improvements.
- scikit-learn’s `hamming_loss` is easy to use and complements Exact Match and F1 for a fuller picture.
- Combine the metric with per-label diagnostics to prioritise remediation where mistakes hurt the most.
