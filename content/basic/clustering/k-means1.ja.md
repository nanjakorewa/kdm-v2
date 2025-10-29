---
title: "k-means"
weight: 1
pre: "2.5.1 "
title_suffix: "クラスタの重心を反復更新する"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means は「近いもの同士をまとめる」という直感に基づき、クラスタ中心（セントロイド）を繰り返し更新してデータを \(k\) 個に分割する。
- 目的関数は各点と対応するセントロイドとの距離二乗和（クラスタ内平方和; WCSS）であり、これを最小化する。
- `scikit-learn` の `KMeans` を使うと、初期値の設定や反復回数を調整しながらアルゴリズムの収束過程を確認できる。
- クラスタ数 \(k\) の選択にはエルボー法やシルエット係数がよく用いられ、データの構造とビジネス要件を合わせて判断する。
{{% /summary %}}

## 直感
k-means は以下のステップを収束するまで繰り返します。

1. クラスタ数 \(k\) をあらかじめ決め、初期セントロイド（クラスタの代表点）を設定する。
2. 各サンプルを「最も近いセントロイド」に割り当てる。
3. 割り当てられたサンプルの平均を取り、新しいセントロイドを再計算する。
4. セントロイドがほとんど動かなくなるまで 2–3 を繰り返す。

セントロイドがクラスタの重心を表すため「重心を探すアルゴリズム」という感覚で捉えると理解しやすいです。  
ただし初期値の設定や外れ値の影響を受けやすいため、複数回初期化したり前処理を丁寧に行うことが重要です。

## 具体的な数式
データ集合 \(\mathcal{X} = \{x_1, \dots, x_n\}\) を \(k\) 個のクラスタ \(\{C_1, \dots, C_k\}\) に分割するとき、k-means はクラスタ内平方和（Within-Cluster Sum of Squares; WCSS）を最小化します。

$$
\min_{C_1, \dots, C_k} \sum_{j=1}^k \sum_{x_i \in C_j} \lVert x_i - \mu_j \rVert^2,
$$

ここで \(\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i\) はクラスタ \(j\) のセントロイドです。  
直感的には「各点とクラスタの重心の距離二乗をできるだけ小さくする」ことに相当します。

## Pythonを用いた実験や説明
以下では `scikit-learn` の `KMeans` を用いて、アルゴリズムの収束の様子やクラスタ数・分散の影響を確認します。

### 収束の過程を可視化する
```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 1000
random_state = 117117
n_clusters = 4
X, y = make_blobs(
    n_samples=n_samples, random_state=random_state, cluster_std=1.5, centers=8
)

# 初期セントロイドを適当に選ぶ
init = X[np.where(X[:, 1] < -8)][:n_clusters]

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c="k", marker="x", label="データ")
plt.legend()
plt.show()
```

![k-means1 block 1](/images/basic/clustering/k-means1_block01.svg)

```python
# 反復回数を変えてセントロイドが収束する様子を見る
plt.figure(figsize=(12, 12))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    max_iter = 1 + i
    km = KMeans(
        n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=1, random_state=1
    ).fit(X)
    cluster_centers = km.cluster_centers_
    y_pred = km.predict(X)

    plt.title(f"max_iter={max_iter}")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker="x")
    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c="r", marker="o", label="セントロイド"
    )
plt.tight_layout()
plt.show()
```

![k-means1 block 2](/images/basic/clustering/k-means1_block02.svg)

### クラスタ間の重なりや \(k\) の影響
```python
# クラスタ分散を徐々に大きくしてみる
plt.figure(figsize=(8, 8))
for i in range(4):
    cluster_std = (i + 1) * 1.5
    X, y = make_blobs(
        n_samples=n_samples, random_state=random_state, cluster_std=cluster_std
    )
    y_pred = KMeans(n_clusters=2, random_state=random_state, init="random").fit_predict(X)
    plt.subplot(2, 2, i + 1)
    plt.title(f"cluster_std={cluster_std}")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker="x")
plt.tight_layout()
plt.show()
```

![k-means1 block 3](/images/basic/clustering/k-means1_block03.svg)

```python
# クラスタ数 k を変えると割り当てがどう変わるか
plt.figure(figsize=(8, 8))
for i in range(4):
    n_clusters = (i + 1) * 2
    X, y = make_blobs(
        n_samples=n_samples, random_state=random_state, cluster_std=1, centers=5
    )
    y_pred = KMeans(
        n_clusters=n_clusters, random_state=random_state, init="random"
    ).fit_predict(X)
    plt.subplot(2, 2, i + 1)
    plt.title(f"#cluster={n_clusters}")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker="x")
plt.tight_layout()
plt.show()
```

![k-means1 block 4](/images/basic/clustering/k-means1_block04.svg)

### \(k\) の選び方
```python
from sklearn.metrics import silhouette_score

K = range(1, 11)
wcss = []
for k in K:
    km = KMeans(n_clusters=k, random_state=117117).fit(X)
    wcss.append(km.inertia_)  # WCSS に対応

plt.plot(K, wcss, marker="o")
plt.xlabel("クラスタ数 k")
plt.ylabel("WCSS")
plt.title("エルボー法")
plt.show()
```

![k-means1 block 5](/images/basic/clustering/k-means1_block05.svg)

```python
scores = []
K = range(2, 11)  # k=1 ではシルエット係数が定義できない
for k in K:
    km = KMeans(n_clusters=k, random_state=117117).fit(X)
    labels = km.labels_
    scores.append(silhouette_score(X, labels))

plt.plot(K, scores, marker="o")
plt.xlabel("クラスタ数 k")
plt.ylabel("シルエットスコア")
plt.title("シルエット分析")
plt.show()
```

![k-means1 block 6](/images/basic/clustering/k-means1_block06.svg)

## 参考文献
{{% references %}}
<li>MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations. <i>Proceedings of the Fifth Berkeley Symposium</i>.</li>
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
