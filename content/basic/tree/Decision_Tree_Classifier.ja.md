---
title: "決定木(分類)"
pre: "2.3.1 "
weight: 1
title_suffix: "について仕組みを理解する"
---

{{% youtube "qQa9Emh0pZE" %}}



<div class="pagetop-box">
  <p><b>決定木（分類）</b>とは、データを「はい/いいえ」のようなルールの組合せで分類するモデルです。  
  分岐のルールを繰り返すと<b>木の枝分かれ</b>のような構造になり、結果が木構造で表現されます。  
  人間にも読みやすいルールとして可視化できるため、<b>解釈しやすい分類モデル</b>の代表例です。</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

---

## 1. 決定木の基本アイデア

- 特徴量に基づいてデータを「条件分岐」させていく  
- 各分岐は「質問」に対応（例：年齢 > 30 か？ yes/no）  
- 最後にたどり着く葉ノードが「クラスの予測」  

### 分岐の基準（不純度）
分岐を決めるには「どの質問でクラスがよく分かれるか」を測る必要があります。  
代表的な指標は以下の2つです：

- **ジニ不純度 (Gini impurity)**  
  $$
  Gini(t) = 1 - \sum_k p_{k}^2
  $$
  （ノード \\(t\\) におけるクラス \\(k\\) の割合 \\(p_k\\) を使う）

- **エントロピー (Entropy)**  
  $$
  H(t) = -\sum_k p_k \log p_k
  $$

どちらも「クラスが混ざっているほど大きくなる」指標で、これを小さくするように分岐を選びます。  
scikit-learn では `criterion="gini"` または `"entropy"` を指定可能です。

---

## 2. サンプルデータの作成

ここでは 2 クラス分類ができる人工データを作ります。

```python
n_classes = 2
X, y = make_classification(
    n_samples=100,
    n_features=2,      # 特徴量を2つに
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_classes=n_classes,
    n_clusters_per_class=1,
)
```

---

## 3. 決定木を学習して境界を可視化

`DecisionTreeClassifier(criterion="gini")` を用いて分類モデルを作成します。  
学習後、決定境界をカラーマップで表示して、分類がどう行われているかを見てみましょう。

{{% notice document %}}
[sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
{{% /notice %}}

```python
# モデルの学習
clf = DecisionTreeClassifier(criterion="gini").fit(X, y)

# 可視化用のメッシュグリッドを作成
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 決定境界の描画
plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1, alpha=0.6)
plt.xlabel("x1")
plt.ylabel("x2")

# 元データを重ねて表示
for i, color, label_name in zip(range(n_classes), ["r", "b"], ["A", "B"]):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=label_name, edgecolor="k")

plt.legend()
plt.title("決定木による分類境界")
plt.show()
```

![decision-tree-classifier block 3](/images/basic/tree/decision-tree-classifier_block03.svg)

---

## 4. 決定木の構造を可視化

木そのものを図として描くこともできます。  
分岐条件・サンプル数・ジニ不純度・クラス比率などが表示されます。

{{% notice document %}}
[sklearn.tree.plot_tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
{{% /notice %}}

```python
plt.figure(figsize=(12, 12))
plot_tree(clf, filled=True, feature_names=["x1", "x2"], class_names=["A", "B"])
plt.show()
```

![decision-tree-classifier block 4](/images/basic/tree/decision-tree-classifier_block04.svg)

---

## 5. 決定木の長所と短所

### 長所
- ルールが木構造で表現されるので<b>解釈しやすい</b>  
- データのスケーリングが不要（正規化・標準化の前処理がなくてもOK）  
- 数値・カテゴリ変数の両方を扱える  

### 短所
- 枝分かれが増えると<b>過学習</b>しやすい  
- データに少しの変化があるだけで木の形が大きく変わる（不安定）  
- 境界は必ず「軸に平行な直線」になるので複雑な分離は苦手  

---

## 6. 実務での工夫

- **深さ制限（max_depth）** や **最小サンプル数（min_samples_split/min_samples_leaf）** を指定して過学習を防ぐ  
- **ランダムフォレストや勾配ブースティング**など「多数の木を組み合わせる手法」として発展（高精度化）  
- 可視化による説明性を活かして「意思決定の根拠」を提示できる  

---

## まとめ

- 決定木は「条件分岐の木構造」で分類するモデル  
- 分岐ルールには「ジニ不純度」や「エントロピー」が使われる  
- 可視化しやすく、解釈性が高いのが大きな利点  
- ただし単独では過学習しやすいため、正則化やアンサンブル手法と組み合わせるのが実務的  

---
