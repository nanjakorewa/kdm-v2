---
title: "Boruta"
pre: "2.7.1 "
weight: 1
searchtitle: "Borutaを使い特徴選択を実行する"
---


## Boruta
Borutaを使って特徴量を選択してみます。このブロックのコードはBorutaの実行サンプルをそのまま持ってきたものです。

`Kursa, Miron B., and Witold R. Rudnicki. "Feature selection with the Boruta package." Journal of statistical software 36 (2010): 1-13.`

{{% youtube "xOkKnsqhUgw" %}}


```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy


# FIXME
np.random.seed(777)
np.int = int
np.float = float
np.bool = bool
```


```python
# load X and y
X = pd.read_csv("examples/test_X.csv", index_col=0).values
y = pd.read_csv("examples/test_y.csv", header=None, index_col=0).values
y = y.ravel()

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators="auto", verbose=2, random_state=1)

# find all relevant features - 5 features should be selected
feat_selector.fit(X, y)

# check selected features - first 5 features are selected
feat_selector.support_

# check ranking of features
feat_selector.ranking_

# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)
```

    Iteration: 	1 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	2 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	3 / 100
    Confirmed: 	0
    Tentative: 	10
    
    BorutaPy finished running.
    
    Iteration: 	9 / 100
    Confirmed: 	5
    Tentative: 	0
    Rejected: 	5
    

## 人工データでの実験


```python
from sklearn.datasets import make_classification
from xgboost import XGBClassifier


def fs_by_boruta(model, X, y):
    feat_selector = BorutaPy(model, n_estimators="auto", verbose=2, random_state=1)
    feat_selector.fit(X, y)
    X_filtered = feat_selector.transform(X)

    if X.shape[1] == X_filtered.shape[1]:
        print("不用な特徴は見つかりませんでした")
    else:
        print("不用な特徴を削除しました")
        print(f"{X.shape[1]} --> {X_filtered.shape[1]}")

    return X_filtered
```

### すべて必要な特徴ならば削除しない


```python
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=10,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
model = XGBClassifier(max_depth=4)
fs_by_boruta(model, X, y)
```

    Iteration: 	1 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	2 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	3 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	4 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	5 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	6 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	7 / 100
    Confirmed: 	0
    Tentative: 	10
    Rejected: 	0
    Iteration: 	8 / 100
    Confirmed: 	10
    Tentative: 	0
    Rejected: 	0
    
    
    BorutaPy finished running.
    
    Iteration: 	9 / 100
    Confirmed: 	10
    Tentative: 	0
    Rejected: 	0
    不用な特徴は見つかりませんでした
    




    array([[ 0.38760058, -0.4398061 ,  1.0103586 , ..., -2.11674403,
            -3.59368321, -0.43265007],
           [-2.18745511, -2.45701675,  1.99758878, ...,  1.16128752,
            -1.92766999,  3.18705784],
           [ 3.98304273,  0.06250274, -1.31136045, ...,  1.45498409,
            -4.17178063, -2.21695578],
           ...,
           [-0.44293666,  3.25707522, -0.50633794, ..., -0.72410483,
            -1.5420989 ,  0.75991518],
           [-1.12641706, -0.48636924,  0.92918889, ..., -1.01001779,
            -2.69280573, -3.47050681],
           [-2.3936814 ,  1.44048113,  1.95832126, ..., -5.15104933,
            -1.02766442,  1.4853396 ]])



### 不用な特徴は削除する
100個のうち10個だけ有用な特徴を混ぜて、何個削除できるかを試します。

[sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)の仕様は

> Without shuffling, X horizontally stacks features in the following order: the primary n_informative features, followed by n_redundant linear combinations of the informative features, followed by n_repeated duplicates, drawn randomly with replacement from the informative and redundant features. The remaining features are filled with random noise. Thus, without shuffling, all useful features are contained in the columns X[:, :n_informative + n_redundant + n_repeated].

となっているので、有用な特徴である先頭の10個の列が削除されていないかどうか確認してみます。


```python
X, y = make_classification(
    n_samples=2000,
    n_features=100,
    n_informative=10,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
model = XGBClassifier(max_depth=5)

X_b = fs_by_boruta(model, X, y)
```

    Iteration: 	1 / 100
    Confirmed: 	0
    Tentative: 	100
    Rejected: 	0
    Iteration: 	2 / 100
...
    
    BorutaPy finished running.
    
    Iteration: 	100 / 100
    Confirmed: 	10
    Tentative: 	1
    Rejected: 	88
    不用な特徴を削除しました
    100 --> 10
    

#### フィルタリング後のデータに有用な特徴が残っていることを確認する
期待どおりならば先頭10列は有用な特徴なのですべて残っているはずです。


```python
X[:, :10] == X_b[:, :10]
```




    array([[ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True],
           ...,
           [ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True]])


