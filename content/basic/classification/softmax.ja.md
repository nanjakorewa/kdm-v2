---
title: "ソフトマックス回帰"
pre: "2.2.2 "
weight: 2
title_suffix: "多クラスの確率を同時に求める"
---

{{% summary %}}
- ソフトマックス回帰はロジスティック回帰を多クラスへ拡張し、全クラスの出現確率を同時に推定する。
- 出力は 0〜1 の確率で合計が 1 になるため、意思決定のしきい値や期待値計算にそのまま利用できる。
- 学習はクロスエントロピー損失を最小化することで行い、確率分布のずれを直接補正する。
- scikit-learn では `LogisticRegression(multi_class="multinomial")` で簡単に扱える。
{{% /summary %}}

## 直感
二値分類ではシグモイド関数で一方のクラスの確率が求まりますが、多クラスでは「各クラスの確率がどれくらいか」を同時に知りたいことが多くあります。ソフトマックス回帰はクラスごとの線形スコアを指数関数で正規化し、確率分布に変換します。指数を使うことで、スコアが高いクラスはより大きく、低いクラスは小さく押し下げられます。

## 具体的な数式
クラス数を \\(K\\)、クラス \\(k\\) の重みベクトルを \\(\mathbf{w}_k\\)、バイアスを \\(b_k\\) とすると、

$$
P(y = k \mid \mathbf{x}) =
\frac{\exp\left(\mathbf{w}_k^\top \mathbf{x} + b_k\right)}
{\sum_{j=1}^{K} \exp\left(\mathbf{w}_j^\top \mathbf{x} + b_j\right)}
$$

で確率が得られます。目的関数はクロスエントロピー

$$
L = - \sum_{i=1}^{n} \sum_{k=1}^{K} \mathbb{1}(y_i = k) \log P(y = k \mid \mathbf{x}_i)
$$

です。確率の合計は常に 1 になるため、確率的な解釈がしやすく、正則化を加えることで係数の暴れも抑えられます。

## Pythonを用いた実験や説明
下記のコードは 3 クラスの人工データにソフトマックス回帰を適用し、決定領域を描画した例です。`multi_class="multinomial"` を指定するとソフトマックス学習が有効になります。

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 3クラスのデータを生成
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_classes=3,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# ソフトマックス回帰（多クラスロジスティック回帰）
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
clf.fit(X, y)

# 予測のためのメッシュグリッドを作成
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 描画
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.title("ソフトマックス回帰による多クラス分類")
plt.xlabel("特徴量1")
plt.ylabel("特徴量2")
plt.show()
```

![softmax block 1](/images/basic/classification/softmax_block01.svg)

## 参考文献
{{% references %}}
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
