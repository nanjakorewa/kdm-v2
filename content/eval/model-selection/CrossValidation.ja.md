---
title: "交差検証"
pre: "4.1.1 "
weight: 1
---

> *標本データを分割し、その一部をまず解析して、残る部分でその解析のテストを行い、解析自身の妥当性の検証・確認に当てる手法* [交差検証 出典: フリー百科事典『ウィキペディア（Wikipedia）』](https://ja.wikipedia.org/wiki/%E4%BA%A4%E5%B7%AE%E6%A4%9C%E8%A8%BC)


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
```
## サンプルデータに対してモデルを作成し交差検証
### 実験用データ


```python
X, y = make_classification(
    n_samples=300,
    n_classes=2,
    n_informative=4,
    n_features=6,
    weights=[0.2, 0.8],
    n_clusters_per_class=2,
    shuffle=True,
    random_state=RND,
)

train_valid_X, test_X, train_valid_y, test_y = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RND
)
```

### 交差検証なしの場合のテストデータでのモデル精度
一度だけ `train_test_split` を実行してそのテストデータで評価します。


```python
train_X, valid_X, train_y, valid_y = train_test_split(
    train_valid_X, train_valid_y, test_size=0.2, random_state=RND
)

model = RandomForestClassifier(max_depth=4, random_state=RND)
model.fit(train_X, train_y)
pred_y = model.predict(valid_X)
rocauc = roc_auc_score(valid_y, pred_y)
print(f"ROC-AUC = {rocauc}")
```

    ROC-AUC = 0.5277777777777778


### 交差検証時のスコア
データを１０分割して交差検証をして、ROC-AUCの平均値を指標として使用します。[sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)を用いると複数の評価指標で簡単に交差検証のスコアを算出できます。
以下の例では、ROC-AUCとAccuracyを交差検証して求めます。


```python
metrics = ("roc_auc", "accuracy")
model = RandomForestClassifier(max_depth=4, random_state=RND)
cv_scores = cross_validate(
    model, train_valid_X, train_valid_y, cv=5, scoring=metrics, return_train_score=True
)

for m in metrics:
    cv_m = cv_scores[f"test_{m}"]
    print(f"{m} {np.mean(cv_m)}")
```

    roc_auc 0.8443019943019943
    accuracy 0.8583333333333334


### テストデータでの性能


```python
model = RandomForestClassifier(max_depth=4, random_state=RND).fit(
    train_valid_X, train_valid_y
)
pred_y = model.predict(test_X)
rocauc = roc_auc_score(test_y, pred_y)
print(f"test ROC-AUC = {rocauc}")
```

    test ROC-AUC = 0.8125000000000001

