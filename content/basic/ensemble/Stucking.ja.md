---
title: "スタッキング"
pre: "2.4.2 "
weight: 2
title_suffix: "の直感・数式・実装"
---

{{< katex />}}
{{% youtube "U5F1PYw_P3E" %}}

<div class="pagetop-box">
  <p><b>スタッキング（Stacking）</b>は、複数のベースモデルの予測結果を<b>メタ学習器</b>に入力し、最終的な予測を行う手法です。異なる性質のモデルを組み合わせることで、精度や安定性を高めることができます。</p>
</div>

{{% notice document %}}
- [StackingClassifier — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
{{% /notice %}}

---

## 1. 直感：モデルの「意見」をまとめる
- それぞれの学習器（決定木やランダムフォレストなど）は「異なる視点」から予測を行う。  
- スタッキングは、それらの出力を新しい学習器（メタ学習器）に入力して、**最終判断を下す仕組み**。  
- 例えるなら「複数の専門家の意見を集めて、まとめ役が最終結論を出す」イメージ。

---

## 2. 数式でみるスタッキング

訓練データ $(x_i, y_i)$ があり、ベース学習器を $h_1, h_2, \dots, h_M$ とする。

1. 各学習器が予測を出力：
   $$
   z_i = \big(h_1(x_i), h_2(x_i), \dots, h_M(x_i)\big)
   $$

2. メタ学習器 $g$ がこれを入力として最終予測を行う：
   $$
   \hat y_i = g(z_i) = g\big(h_1(x_i), \dots, h_M(x_i)\big)
   $$

> ポイント：単純に平均や多数決するのではなく、**学習によって最適な組み合わせ方を学ぶ**ところが特徴です。

---

## 3. データの準備

```python
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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

---

## 4. ベースライン：ランダムフォレスト

```python
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=777)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_score = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC (RF) = {rf_score:.3f}")
```

---

## 5. スタッキング（決定木のみの例）

ベース学習器をすべて決定木にした場合、似た傾向の予測になるため改善は小さいことがあります。

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

---

## 6. 実務で効くコツ
- ベース学習器は<b>異なる系統</b>を混ぜる（例：線形モデル＋木系＋距離ベース）。  
- メタ学習器は<b>シンプルなモデル</b>（ロジスティック回帰など）から始めると安定。  
- scikit-learn の `StackingClassifier` は内部で交差検証（CV）を用いるため、**データリークを防ぐ設計**になっている。  
- 回帰でも `StackingRegressor` を使って同様に実装可能。  

---

## 7. まとめ
- スタッキングは **「複数モデルの予測をさらに学習して最適に組み合わせる」** 手法。  
- 単純多数決の Bagging より柔軟にモデルを融合できる。  
- 効果を出すには **性質の異なるモデルを組み合わせることが重要**。  
- 実務ではベースモデルの多様性と、メタ学習器の選択がカギ。  

---
