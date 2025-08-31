---
title: "スタッキング"
pre: "2.4.2 "
weight: 2
title_suffix: "の直感・設定・実装"
---

{{% youtube "U5F1PYw_P3E" %}}

<div class="pagetop-box">
  <p><b>スタッキング（Stacking）</b>は、複数のベースモデルの予測を<b>メタ学習器</b>に入力し、最終予測を出す手法です。異なる性質のモデルを組み合わせることで精度・安定性の向上を狙います。</p>
</div>

```python
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
```

## データの準備

```python
n_features = 20
X, y = make_classification(
    n_samples=2500,
    n_features=n_features,
    n_informative=10,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=4,
    random_state=777,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=777
)
```

## ベースライン：RandomForest

```python
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=777)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_score = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC (RF) = {rf_score:.3f}")
```

## スタッキング（決定木のみ）

ベース学習器を全て決定木にした場合、表現力が似通っているため改善が小さいことがあります。

{{% notice document %}}
[StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
{{% /notice %}}

```python
# 前段のモデル
estimators = [
    ("dt1", DecisionTreeClassifier(max_depth=3, random_state=777)),
    ("dt2", DecisionTreeClassifier(max_depth=4, random_state=777)),
    ("dt3", DecisionTreeClassifier(max_depth=5, random_state=777)),
]

# 後段（メタ学習器）
final_estimator = DecisionTreeClassifier(max_depth=3, random_state=777)

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
clf_score = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC (Stacking) = {clf_score:.3f}")
```

## 実務のコツ

- ベース学習器は<b>異なる系統</b>（例：線形モデル＋木系＋距離ベース）を混ぜると効果が出やすい。
- メタ学習器は<b>単純なモデル</b>（ロジスティック回帰など）から試すと安定。
- scikit-learn の Stacking は内部で CV を使い、<b>リークしにくい設計</b>になっています。

