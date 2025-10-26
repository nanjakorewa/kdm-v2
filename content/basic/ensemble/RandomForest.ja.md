---
title: "ランダムフォレスト"
pre: "2.4.1 "
weight: 1
title_suffix: "の直感・数式・実装"
---

{{< katex />}}
{{% youtube "ewvjQMj8nA8" %}}

<div class="pagetop-box">
  <p><b>ランダムフォレスト</b>は、多数の決定木を<b>ブートストラップ（再標本化）</b>と<b>特徴量サブサンプリング</b>を組み合わせて学習し、分類は多数決、回帰は平均で予測する <b>Bagging 系</b>のアンサンブルです。分散を下げ、頑健な予測を狙います。</p>
</div>

{{% notice document %}}
- [RandomForestClassifier — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)  
- [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)  
- [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
{{% /notice %}}

---

## 1. 直感：多数の「弱い木」を寄せ集めて安定化
- 決定木は「過学習しやすい」モデル。  
- ランダムフォレストは **データの一部**（ブートストラップ標本）と **特徴量の一部**を毎回選んで木を作る。  
- 木ごとに「バラバラな視点」を持たせることで、全体をまとめると頑健で精度の高いモデルになる。  

---

## 2. 数式でみるランダムフォレスト

ブートストラップ標本 \\(\mathcal{D}_b\\) ごとに木 \\(h_b(x)\\) を学習（\\(b=1,\dots,B\\)）。

- 分類予測（多数決）：
  $$
  \hat y(x) = \operatorname*{arg\,max}_c \sum_{b=1}^B \mathbf{1}[h_b(x) = c]
  $$

- 回帰予測（平均）：
  $$
  \hat y(x) = \frac{1}{B}\sum_{b=1}^B h_b(x)
  $$

各木の分割には「ジニ不純度」などを用いる：
$$
\mathrm{Gini}(S) = 1 - \sum_{c} p(c\mid S)^2
$$

---

## 3. 分類データで学習し ROC-AUC を確認

```python
import numpy as np
import matplotlib.pyplot as plt
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
print(f"テストデータでのROC-AUC = {rf_score:.3f}")
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_6_0.png)

---

## 4. 各決定木の性能を比較してみる

```python
import japanize_matplotlib

estimator_scores = []
for i in range(10):
    estimator = model.estimators_[i]
    estimator_pred = estimator.predict(X_test)
    estimator_scores.append(roc_auc_score(y_test, estimator_pred))

plt.figure(figsize=(10, 4))
bar_index = [i for i in range(len(estimator_scores))]
plt.bar(bar_index, estimator_scores)
plt.bar([10], rf_score)
plt.xticks(bar_index + [10], bar_index + ["RF"])
plt.xlabel("木のインデックス")
plt.ylabel("ROC-AUC")
plt.show()
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_6_0.png)

> 各木はバラバラの性能ですが、全体でまとめると安定して精度が高くなります。

---

## 5. 特徴量の重要度

### 5.1 不純度（impurity）に基づく重要度
各ノード分割での「不純度減少」を特徴量ごとに足し上げ、全木で平均した量。

```python
plt.figure(figsize=(10, 4))
feature_index = [i for i in range(n_features)]
plt.bar(feature_index, model.feature_importances_)
plt.xlabel("特徴のインデックス")
plt.ylabel("特徴の重要度")
plt.show()
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_8_0.png)

### 5.2 Permutation importance（入れ替え重要度）

```python
from sklearn.inspection import permutation_importance

p_imp = permutation_importance(
    model, X_train, y_train, n_repeats=10, random_state=77
).importances_mean

plt.figure(figsize=(10, 4))
plt.bar(feature_index, p_imp)
plt.xlabel("特徴のインデックス")
plt.ylabel("特徴の重要度")
plt.show()
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_10_0.png)

---

## 6. 木の可視化（任意）

```python
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image, display

for i in range(3):
    try:
        estimator = model.estimators_[i]
        export_graphviz(
            estimator,
            out_file=f"tree{i}.dot",
            feature_names=[f"x{i}" for i in range(n_features)],
            class_names=["A", "B"],
            proportion=True,
            filled=True,
        )
        call(["dot", "-Tpng", f"tree{i}.dot", "-o", f"tree{i}.png", "-Gdpi=500"])
        display(Image(filename=f"tree{i}.png"))
    except Exception:
        pass
```

![png](/images/basic/ensemble/RandomForest_files/RandomForest_12_0.png)

---

## 7. ハイパーパラメータの目安
- **n_estimators** : 木の本数。多いほど安定だが計算増。  
- **max_depth** : 各木の深さ。深すぎると過学習、浅すぎると精度不足。  
- **max_features** : 分割で試す特徴数。小さめにすると木の相関が減り→多様性UP。  
- **bootstrap**, **oob_score** : OOB（袋外データ）で汎化性能を評価可能。  

---

## 8. まとめ
- ランダムフォレストは **「多数の弱い木の集合で強いモデルを作る」** 手法。  
- Bagging に「特徴量のランダム化」を加えることで、木同士の相関を下げて性能を安定化。  
- 予測は分類なら多数決、回帰なら平均。  
- 特徴量重要度の計算も可能で、解釈性に活用できる。  
- 実務では **n_estimators・max_depth・max_features** の調整が重要。  

---
