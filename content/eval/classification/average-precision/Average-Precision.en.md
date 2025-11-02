---

title: "Average Precision (AP) | Evaluating Precision–Recall curves"

linkTitle: "Average Precision"

seo_title: "Average Precision (AP) | Evaluating Precision–Recall curves"

pre: "4.3.9 "

weight: 9

---



{{< lead >}}

Average Precision (AP) summarises the Precision–Recall curve by weighting precision with the increase in recall. It captures how a model behaves across all thresholds, especially on imbalanced datasets. Let’s compute it in Python 3.13 and see how it complements F1 and ROC-AUC.

{{< /lead >}}



---



## 1. Definition



If the PR curve consists of points \((R_n, P_n)\), Average Precision is defined as





\mathrm{AP} = \sum_{n}(R_n - R_{n-1}) P_n





The change in recall acts as the weight, so AP reflects the average precision as the threshold slides from high to low.



---



## 2. Computing AP in Python 3.13



```bash

python --version        # e.g. Python 3.13.0

pip install scikit-learn matplotlib

```



Reusing the probabilities proba from the Precision–Recall example, we can obtain AP with a couple of scikit-learn calls:



```python

from sklearn.metrics import precision_recall_curve, average_precision_score



precision, recall, thresholds = precision_recall_curve(y_test, proba)

ap = average_precision_score(y_test, proba)

print(f"Average Precision: {ap:.3f}")

```



The corresponding PR curve is the same pr_curve.png we generated earlier.



{{< figure src="/images/eval/classification/precision-recall/pr_curve.png" alt="Precision–Recall curve" caption="AP integrates the PR curve by weighting precision with recall increments." >}}



---



## 3. AP versus PR-AUC



- verage_precision_score implements the step-wise integration commonly used in information retrieval.

- sklearn.metrics.auc(recall, precision) applies the trapezoidal rule and yields the classical PR-AUC.

- AP tends to be more robust on imbalanced datasets because it emphasises changes where recall actually increases.



---



## 4. Practical takeaways



- **Threshold tuning** – Higher AP implies that the model keeps precision high across a wider span of recalls.

- **Ranking tasks** – In recommendation and search, Mean Average Precision (MAP) averages AP across queries.

- **Complement to F1** – F1 reflects a single operating point; AP reflects the whole threshold spectrum.



---



## Summary



- Average Precision evaluates the entire Precision–Recall landscape, making it ideal for imbalanced problems.

- Python 3.13 + scikit-learn compute it with verage_precision_score in a few lines.

- Pair AP with F1, ROC-AUC, and PR curves when presenting model comparisons or choosing operating thresholds.

---

