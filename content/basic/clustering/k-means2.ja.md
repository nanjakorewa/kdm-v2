---
title: "k-means++"
weight: 2
pre: "2.5.2 "
title_suffix: "初期セントロイドを賢く選ぶ"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means++ は k-means の初期セントロイド選択を改良し、遠く離れた点を優先的に選ぶことで収束を安定させる。
- アルゴリズムは「1 点目はランダム → 距離二乗に比例した確率で次点を選択」を繰り返し、クラスタが偏らないようにする。
- 乱数初期化との挙動の違いを `scikit-learn` の `KMeans(init="k-means++")` で簡単に比較できる。
- 大規模データやオンライン処理では k-means++ をベースにした mini-batch 版が実務で広く利用される。
{{% /summary %}}

## 直感
k-means は初期セントロイドの選び方に敏感で、悪い初期値では局所解に陥ってしまいます。  
k-means++ では「セントロイド同士ができるだけ離れるように初期化」することで、クラスタの偏りを避けます。  
最初の 1 点はランダムに選びつつ、2 点目以降は既存セントロイドとの距離が大きい点ほど高い確率で選ばれます。

## 具体的な数式
既に選ばれたセントロイド集合 \\(\{\mu_1, \dots, \mu_m\}\\) に対して、新しい候補点 \\(x\\) の選択確率は

$$
P(x) = \frac{D(x)^2}{\sum_{x' \in \mathcal{X}} D(x')^2}, \qquad D(x) = \min_{1 \le j \le m} \lVert x - \mu_j \rVert.
$$

距離 \\(D(x)\\) が大きい点、すなわち既存セントロイドから遠い点ほど高い確率で選ばれます。  
この初期化により、理論的には \\(O(\log k)\\) 近似保証が得られることが知られています（Arthur & Vassilvitskii, 2007）。

## Pythonを用いた実験や説明
乱数初期化と k-means++ 初期化を比較し、クラスタリングの違いを可視化します。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 3000
random_state = 11711
X, y = make_blobs(
    n_samples=n_samples, random_state=random_state, cluster_std=1.5, centers=8
)

for i in range(3):
    rand_index = np.random.permutation(1000)
    X_rand = X[rand_index]

    km_random = KMeans(
        n_clusters=5, random_state=random_state, init="random",
        max_iter=1, n_init=1
    ).fit(X_rand)
    y_pred_random = km_random.labels_

    km_kpp = KMeans(
        n_clusters=5, random_state=random_state, init="k-means++",
        max_iter=1, n_init=1
    ).fit(X_rand)
    y_pred_kpp = km_kpp.labels_

    plt.figure(figsize=(10, 2))
    plt.subplot(1, 2, 1)
    plt.title("random 初期化")
    plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_pred_random, marker="x")

    plt.subplot(1, 2, 2)
    plt.title("k-means++ 初期化")
    plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_pred_kpp, marker="x")
    plt.tight_layout()
    plt.show()
```

![k-means2 block 1](/images/basic/clustering/k-means2_block01.svg)
![k-means2 block 1 fig 1](/images/basic/clustering/k-means2_block01_fig01.svg)
![k-means2 block 1 fig 2](/images/basic/clustering/k-means2_block01_fig02.svg)
![k-means2 block 1 fig 3](/images/basic/clustering/k-means2_block01_fig03.svg)
![k-means2 block 1 fig 4](/images/basic/clustering/k-means2_block01_fig04.svg)

## 参考文献
{{% references %}}
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>Bahmani, B. et al. (2012). Scalable k-means++. <i>Proceedings of the VLDB Endowment</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
