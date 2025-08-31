---
title: "k-means"
weight: 1
pre: "2.5.1 "
title_suffix: "の仕組みの説明"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

<div class="pagetop-box">
    <p><b>k-means</b>とはクラスタリングの代表的アルゴリズムです。与えられたデータを<b>k個のクラスタ</b>に分けるために、クラスタの代表点（セントロイド）を計算し、最も近いセントロイドにデータを割り当てる作業を繰り返します。</p>
</div>

---

## 1. 直感：繰り返し「重心」を探す
- まずクラスタの数 $k$ を決める。  
- 適当に $k$ 個の点をクラスタの中心（セントロイド）として置く。  
- 各データ点を「最も近いセントロイド」に割り当てる。  
- 各クラスタの平均を計算し、新しいセントロイドに更新。  
- これを繰り返すと、クラスタが安定して分かれる。  

---

## 2. 数式でみる k-means

データセット $\{x_1, \dots, x_n\}$ を $k$ 個のクラスタに分けるとき、目的は **クラスタ内平方和（Within-Cluster Sum of Squares, WCSS）** を最小化すること：

$$
\min_{C_1, \dots, C_k} \sum_{j=1}^k \sum_{x_i \in C_j} \| x_i - \mu_j \|^2
$$

ここで：
- $C_j$ = 第 $j$ クラスタの集合
- $\mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j} x_i$ = クラスタ $j$ の重心（セントロイド）

> つまり「各点とその属するクラスタ中心の距離の二乗」をできるだけ小さくする。

---

## 3. k-means クラスタリングの例

### サンプルデータを作成
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

# 初期値用のポイント
init = X[np.where(X[:, 1] < -8)][:n_clusters]

# データと初期点を可視化
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c="k", marker="x", label="学習用データ")
plt.legend()
plt.show()
```

![png](/images/basic/clustering/k-means1_files/k-means1_5_0.png)

---

### セントロイドの計算過程

```python
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
        c="r", marker="o", label="クラスタのセントロイド"
    )
plt.tight_layout()
plt.show()
```

![png](/images/basic/clustering/k-means1_files/k-means1_7_0.png)

---

## 4. クラスタが重なり合っている場合

クラスタが重なり合うと、境界があいまいになり、誤割り当ても起きやすい。

```python
n_samples = 1000
random_state = 117117

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

![png](/images/basic/clustering/k-means1_files/k-means1_9_0.png)

---

## 5. k の数の影響

クラスタ数 $k$ を変えると、分割のされ方が変わる。

```python
n_samples = 1000
random_state = 117117

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

![png](/images/basic/clustering/k-means1_files/k-means1_11_0.png)

---

## 6. 実務でのコツ
- $k$ は事前に決める必要がある → **エルボー法**や**シルエット係数**を使って適切な値を選ぶ。  
- スケーリング（標準化）は効果的（距離計算が前提のため）。  
- 外れ値があるとセントロイドが引っ張られやすい → 事前処理が大切。  
- 初期値の違いで局所解に落ちることがある → `n_init` を大きめに設定。  

---

## 7. まとめ
- k-means は「クラスタごとの平均」を繰り返し更新してデータを分類するアルゴリズム。  
- 数式的には **クラスタ内平方和の最小化**を目指している。  
- $k$ の選び方や初期値の工夫が性能に大きく影響する。  
- シンプルかつ計算効率がよいので、クラスタリングの基本手法として広く利用されている。  

---

## 発展：クラスタ数 $k$ の決め方

### 1. エルボー法
クラスタ数を変えたときの **WCSS（クラスタ内平方和）** を計算し、減少の度合いが「肘（elbow）」のように折れ曲がる点を最適な $k$ とみなす。

- WCSS の定義：
  $$
  \mathrm{WCSS}(k) = \sum_{j=1}^k \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
  $$

```python
wcss = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=117117).fit(X)
    wcss.append(km.inertia_)  # inertia_ は WCSS に相当

plt.plot(K, wcss, marker="o")
plt.xlabel("クラスタ数 k")
plt.ylabel("WCSS")
plt.title("エルボー法")
plt.show()
```

---

### 2. シルエットスコア
各点のクラスタリングの適切さを評価する指標。  
点 $i$ に対して：

- $a(i)$ = 自分のクラスタ内の平均距離  
- $b(i)$ = 他クラスタの中で最小の平均距離  

シルエット係数：
$$
s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}
$$

- $s(i)$ が 1 に近い → 良くクラスタ分けされている  
- 0 に近い → 境界に近い  
- 負 → 間違ったクラスタに入っている可能性  

```python
from sklearn.metrics import silhouette_score

scores = []
K = range(2, 11)  # k=1 では定義できない
for k in K:
    km = KMeans(n_clusters=k, random_state=117117).fit(X)
    labels = km.labels_
    scores.append(silhouette_score(X, labels))

plt.plot(K, scores, marker="o")
plt.xlabel("クラスタ数 k")
plt.ylabel("シルエットスコア")
plt.title("シルエット法")
plt.show()
```

---

## 発展まとめ
- **エルボー法**：WCSS の減り方を見て「肘」の位置を選ぶ。  
- **シルエットスコア**：クラスタの分離度と一貫性を評価。  
- 実務では両方を参考にしつつ、業務上の解釈のしやすさも考慮して $k$ を決めるとよい。  

---
