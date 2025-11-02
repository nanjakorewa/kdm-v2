---

title: "検証曲線"

pre: "4.1.3 "

weight: 3

---



{{% notice document %}}

[sklearn.model_selection.validation_curve](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html)

{{% /notice %}}

モデルの訓練データへの適合度と検証用データへの予測性能を同時に比較するための便利機能。





```python

import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import validation_curve

from sklearn.metrics import roc_auc_score

```python

# サンプルデータに対してモデルを作成し交差検証

### 実験用データ





```python

X, y = make_classification(

    n_samples=1000,

    n_classes=2,

    n_informative=4,

    n_clusters_per_class=3,

    random_state=RND,

)



```