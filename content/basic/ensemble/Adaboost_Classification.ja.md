---
title: "AdaBoost（分類）"
pre: "2.4.3 "
weight: 3
title_suffix: "の直感・数式・実装"
---

{{% youtube "1K-h4YzrnsY" %}}

<div class="pagetop-box">
  <p><b>AdaBoost</b> は、誤分類したサンプルに<b>重み</b>を置いて次の弱学習器を学習する <b>Boosting</b> 手法です。弱学習器（たとえば深さの浅い決定木）を段階的に積み重ね、強い分類器にします。</p>
</div>

{{% notice document %}}
- [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
{{% /notice %}}

## アルゴリズム（数式）

反復 <code>t=1,\dots,T</code> で弱学習器 <code>\(h_t(x)\)</code> を学習。重み付き誤り率
<code>\(\varepsilon_t = \frac{\sum_i w_i^{(t)} \mathbf{1}[y_i \ne h_t(x_i)]}{\sum_i w_i^{(t)}}\)</code> から
係数 <code>\(\alpha_t = \tfrac{1}{2}\ln \frac{1-\varepsilon_t}{\varepsilon_t}\)</code> を計算。

サンプル重み更新：
<code>\(w_i^{(t+1)} = w_i^{(t)} \exp\big( \alpha_t\, \mathbf{1}[y_i \ne h_t(x_i)] \big)\)</code>（正規化あり）。

最終分類器：<code>\(H(x) = \operatorname{sign}\big( \sum_{t=1}^T \alpha_t h_t(x) \big)\)</code>

---

## 実験データの作成と学習

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

n_features = 20
X, y = make_classification(
    n_samples=2500,
    n_features=n_features,
    n_informative=10,
    n_classes=2,
    n_redundant=4,
    n_clusters_per_class=5,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

ab_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=10,
    learning_rate=1.0,
    random_state=117117,
)
ab_clf.fit(X_train, y_train)

y_pred = ab_clf.predict(X_test)
score = roc_auc_score(y_test, y_pred)
print("ROC-AUC:", score)
```

![png](/images/basic/ensemble/Adaboost_Classification_files/Adaboost_Classification_10_0.png)

---

## ハイパーパラメータの影響

### learning_rate（学習率）
小さすぎると進みが遅く、大きすぎると収束しにくい場合があります。<b>n_estimators</b> とトレードオフ。

```python
scores = []
learning_rate_list = np.linspace(0.01, 1, 50)
for lr in learning_rate_list:
    ab = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=10,
        learning_rate=lr,
        random_state=117117,
    ).fit(X_train, y_train)
    y_pred = ab.predict(X_test)
    scores.append(roc_auc_score(y_test, y_pred))

plt.figure(figsize=(5, 5))
plt.plot(learning_rate_list, scores)
plt.xlabel("learning rate")
plt.ylabel("ROC-AUC")
plt.grid()
plt.show()
```

![png](/images/basic/ensemble/Adaboost_Classification_files/Adaboost_Classification_13_0.png)

### n_estimators（反復回数）
大きくすると表現力は増しますが、過学習や計算コストに注意。適度に増やし、<b>learning_rate</b> と併せて調整します。

```python
scores = []
n_estimators_list = [int(ne) for ne in np.linspace(5, 70, 20)]
for ne in n_estimators_list:
    ab = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=ne,
        learning_rate=0.6,
        random_state=117117,
    ).fit(X_train, y_train)
    y_pred = ab.predict(X_test)
    scores.append(roc_auc_score(y_test, y_pred))

plt.figure(figsize=(5, 5))
plt.plot(n_estimators_list, scores)
plt.xlabel("n_estimators")
plt.ylabel("ROC-AUC")
plt.grid()
plt.show()
```

![png](/images/basic/ensemble/Adaboost_Classification_files/Adaboost_Classification_16_0.png)

### base_estimator（弱学習器）
深さの異なる決定木で比較してみます。深くすると個々は強くなりますが、Boosting の利点が薄れることも。

```python
scores = []
base_list = [DecisionTreeClassifier(max_depth=md) for md in [2,3,4,5,6]]
for base in base_list:
    ab = AdaBoostClassifier(
        base_estimator=base, n_estimators=10, learning_rate=0.5, random_state=117117
    ).fit(X_train, y_train)
    y_pred = ab.predict(X_test)
    scores.append(roc_auc_score(y_test, y_pred))

plt.figure(figsize=(6, 5))
idx = range(len(base_list))
plt.bar(list(idx), scores)
plt.xticks(list(idx), [str(b) for b in base_list], rotation=90)
plt.xlabel("base_estimator")
plt.ylabel("ROC-AUC")
plt.tight_layout()
plt.show()
```

---

## ポイントまとめ

- サンプル重みにより、<b>難しいサンプル</b>を重点的に学習。
- <b>learning_rate × n_estimators</b> のバランス調整が重要。
- 弱学習器は浅い木が定番。過度に強い弱学習器は Boosting 効果を損なうことも。

