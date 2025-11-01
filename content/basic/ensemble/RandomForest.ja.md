---
title: "ランダムフォレスト | 複数の決定木をバギングして高精度化する"
linkTitle: "ランダムフォレスト"
seo_title: "ランダムフォレスト | 複数の決定木をバギングして高精度化する"
pre: "2.4.1 "
weight: 1
title_suffix: "多数の決定木を束ねて安定化する"
---

{{< katex />}}
{{% youtube "ewvjQMj8nA8" %}}

{{% summary %}}
- ランダムフォレストはブートストラップ標本と特徴量サブサンプリングによって多様な決定木を学習し、投票（分類）または平均（回帰）で予測する Bagging 系アンサンブル。
- 各木が扱うデータ・特徴量をランダムに変えることで、モデル間の相関を下げ、分散を減らしつつ過学習を抑える。
- `n_estimators`, `max_depth`, `max_features` の調整で精度と汎化のバランスを取るほか、OOB（Out-of-Bag）スコアで汎化性能を推定できる。
- 特徴量重要度や Permutation importance を利用すると、モデルの解釈や変数選択にも活用できる。
{{% /summary %}}

## 直感
単体の決定木は学習データに過度に適合しやすい一方で、複数の木の予測を平均すれば偶発的な誤差が打ち消し合い、安定したモデルになります。ランダムフォレストでは「データセットをブートストラップで再標本化」「各分割で候補とする特徴量をランダムに選択」の 2 つのランダム性を導入し、木ごとに異なる視点を持たせます。

## 具体的な数式
ブートストラップ標本 \\(\mathcal{D}_b\\) から学習した決定木 \\(h_b(x)\\) を \\(B\\) 本集めると、

- **分類**（多数決）  
  $$
  \hat y(x) = \operatorname*{arg\,max}_c \sum_{b=1}^B \mathbf{1}\bigl[h_b(x) = c\bigr]
  $$
- **回帰**（平均）  
  $$
  \hat y(x) = \frac{1}{B} \sum_{b=1}^B h_b(x)
  $$

となります。各木のノード分割ではジニ不純度などの指標を使って特徴量を選択し、木が深くなりすぎないように `max_depth` や `min_samples_leaf` を設定します。

## Pythonを用いた実験や説明
以下では 2 クラス分類問題でランダムフォレストを学習し、各木の ROC-AUC とアンサンブル全体の性能を比較します。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
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

model = RandomForestClassifier(
    n_estimators=50, max_depth=3, random_state=777,
    bootstrap=True, oob_score=True
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rf_score = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC (RandomForest) = {rf_score:.3f}")
```

```python
estimator_scores = []
for estimator in model.estimators_[:10]:
    estimator_pred = estimator.predict(X_test)
    estimator_scores.append(roc_auc_score(y_test, estimator_pred))

plt.figure(figsize=(10, 4))
bar_index = list(range(len(estimator_scores)))
plt.bar(bar_index, estimator_scores, label="各決定木")
plt.bar([10], rf_score, label="ランダムフォレスト")
plt.xticks(bar_index + [10], bar_index + ["RF"])
plt.xlabel("インデックス")
plt.ylabel("ROC-AUC")
plt.legend()
plt.show()
```

![randomforest block 2](/images/basic/ensemble/randomforest_block02.svg)

```python
plt.figure(figsize=(10, 4))
plt.bar(range(n_features), model.feature_importances_)
plt.xlabel("特徴量インデックス")
plt.ylabel("不純度ベースの重要度")
plt.show()
```

![randomforest block 3](/images/basic/ensemble/randomforest_block03.svg)

```python
from sklearn.inspection import permutation_importance

p_imp = permutation_importance(
    model, X_train, y_train,
    n_repeats=10, random_state=77
).importances_mean

plt.figure(figsize=(10, 4))
plt.bar(range(n_features), p_imp)
plt.xlabel("特徴量インデックス")
plt.ylabel("Permutation importance")
plt.show()
```

![randomforest block 4](/images/basic/ensemble/randomforest_block04.svg)

## 参考文献
{{% references %}}
<li>Breiman, L. (2001). Random Forests. <i>Machine Learning</i>, 45(1), 5–32.</li>
<li>scikit-learn developers. (2024). <i>RandomForestClassifier</i>. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html</li>
{{% /references %}}
