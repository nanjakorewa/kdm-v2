---
title: "決定木（分類）"
pre: "2.3.1 "
weight: 1
title_suffix: "分割ルールで直感的に分類する"
---

{{% youtube "qQa9Emh0pZE" %}}

{{% summary %}}
- 決定木分類器は「もし◯◯なら左へ、そうでなければ右へ」という条件分岐を繰り返し、葉ノードでクラスを予測する。
- 分割の評価にはジニ不純度やエントロピーなどの不純度指標を用い、クラスが純粋になるように特徴量としきい値を選ぶ。
- 木構造をそのまま図示できるため、予測根拠を人間に説明しやすい。
- 深さや葉ノードサイズを制御して過学習を防ぎ、必要に応じてアンサンブル手法へ発展させる。
{{% /summary %}}

## 直感
サンプルを特徴量に基づいて順番に分岐させ、最終的にたどり着いた葉ノードの多数派クラスを予測として返します。分岐を重ねるほど木は深くなり、データを細かく区切ることができますが、過学習もしやすくなります。木の各ノードは「質問」に相当し、枝は「はい / いいえ」で進むので、意思決定の理由を追跡しやすいのが大きな利点です。

## 具体的な数式
ノード \\(t\\) におけるジニ不純度は

$$
\mathrm{Gini}(t) = 1 - \sum_k p_k^2
$$

で定義されます（\\(p_k\\) はノード \\(t\\) に含まれるクラス \\(k\\) の割合）。エントロピーは

$$
H(t) = - \sum_k p_k \log p_k
$$

です。親ノード \\(t\\) を特徴量 \\(x_j\\) としきい値 \\(s\\) で左右に分割したときの不純度減少量

$$
\Delta I = I(t) - \frac{n_L}{n_t} I(t_L) - \frac{n_R}{n_t} I(t_R)
$$

が最大になる組み合わせを選び、葉ノードに達するまで繰り返します。ここで \\(I(\cdot)\\) は選んだ不純度指標、\\(n_t\\) はノード内のサンプル数です。

## Pythonを用いた実験や説明
人工データで決定境界と木構造を可視化するコードです。元のサンプルコードをそのまま利用しています。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

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

```python
plt.figure(figsize=(12, 12))
plot_tree(clf, filled=True, feature_names=["x1", "x2"], class_names=["A", "B"])
plt.show()
```

![decision-tree-classifier block 4](/images/basic/tree/decision-tree-classifier_block04.svg)

## 参考文献
{{% references %}}
<li>Breiman, L., Friedman, J. H., Olshen, R. A., &amp; Stone, C. J. (1984). <i>Classification and Regression Trees</i>. Wadsworth.</li>
<li>scikit-learn developers. (2024). <i>Decision Trees</i>. https://scikit-learn.org/stable/modules/tree.html</li>
{{% /references %}}
