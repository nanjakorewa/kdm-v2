---
title: "ROC-AUC"
pre: "4.3.1 "
weight: 1
searchtitle: "plot grafik ROC-AUC dalam python"
---

Area di bawah kurva ROC disebut AUC (Area Under the Curve) dan digunakan sebagai indeks evaluasi untuk model klasifikasi; yang terbaik adalah ketika AUC adalah 1, dan 0.5 untuk model acak dan sama sekali tidak valid.

- ROC-AUC adalah contoh tipikal dari indeks evaluasi klasifikasi biner
- 1 adalah yang terbaik, 0,5 mendekati prediksi yang benar-benar acak
- Di bawah 0,5 bisa jadi ketika prediksi adalah kebalikan dari jawaban yang benar
- Merencanakan kurva ROC dapat membantu menentukan ambang klasifikasi yang seharusnya


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
```

## Plot Kurva ROC
{{% notice document %}}
[sklearn.metrics.roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
{{% /notice %}}

### Fungsi untuk memplot Kurva ROC

```python
def plot_roc_curve(test_y, pred_y):
    """Plot Kurva ROC dari jawaban dan prediksi yang benar

    Args:
        test_y (ndarray of shape (n_samples,)): y
        pred_y (ndarray of shape (n_samples,)): Nilai prediksi untuk y
    """
    # Tingkat Positif Palsu, Tingkat Positif Sejati

    fprs, tprs, thresholds = roc_curve(test_y, pred_y)

    # plot ROC
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle="-", c="k", alpha=0.2, label="ROC-AUC=0.5")
    plt.plot(fprs, tprs, color="orange", label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Isi area yang sesuai dengan skor ROC-AUC
    y_zeros = [0 for _ in tprs]
    plt.fill_between(fprs, y_zeros, tprs, color="orange", alpha=0.3, label="ROC-AUC")
    plt.legend()
    plt.show()
```

### Membuat model dan memplot Kurva ROC terhadap data sampel

```python
X, y = make_classification(
    n_samples=1000,
    n_classes=2,
    n_informative=4,
    n_clusters_per_class=3,
    random_state=RND,
)
train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.33, random_state=RND
)

model = RandomForestClassifier(max_depth=5)
model.fit(train_X, train_y)
pred_y = model.predict_proba(test_X)[:, 1]
plot_roc_curve(test_y, pred_y)
```


    
![png](/images/eval/classification/ROC-AUC_files/ROC-AUC_6_0.png)
    


### Hitung ROC-AUC
{{% notice document %}}
[sklearn.metrics.roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
{{% /notice %}}


```python
from sklearn.metrics import roc_auc_score

roc_auc_score(test_y, pred_y)
```




    0.89069793083171


