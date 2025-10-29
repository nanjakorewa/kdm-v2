---
title: "X-means"
weight: 3
pre: "2.5.3 "
title_suffix: "の仕組みと使い方"
---

{{< katex />}}
{{% youtube "2hkyJcWctUA" %}}

<div class="pagetop-box">
    <p><b>X-means</b>は、<b>クラスタ数を自動的に決定できる</b>クラスタリング手法です。<br>
    k-means や k-means++ はクラスタ数 \\(k\\) を事前に指定する必要がありますが、X-means は統計的基準を使って「最適なクラスタ数」を選びます。</p>
</div>

{{% notice document %}}
[pyclustering.cluster.xmeans.xmeans Class Reference](https://pyclustering.github.io/docs/0.9.0/html/dd/db4/classpyclustering_1_1cluster_1_1xmeans_1_1xmeans.html)
{{% /notice %}}

{{% notice ref %}}
Pelleg, Dan, and Andrew W. Moore. "X-means: Extending k-means with efficient estimation of the number of clusters." ICML, 2000.
{{% /notice %}}

---

## 1. 直感：クラスタ数を「分割しながら」決める
- k-means：クラスタ数 \\(k\\) を人間が決める必要がある。  
- X-means：まず少ないクラスタ数から始める。  
  - その後、クラスタを分割するかどうかを **BIC や MDL の基準**で判断する。  
- これを繰り返して、最適なクラスタ数を自動決定する。  

---

## 2. 数式でみる X-means

### 2.1 BIC（ベイズ情報量基準）
クラスタ数が増えると「当てはまり」は良くなるが、複雑すぎると過学習になる。  
BIC は「モデルの当てはまり」と「複雑さ」のバランスをとる指標。

$$
\mathrm{BIC} = \ln L - \frac{p}{2} \ln n
$$

- \\(L\\)：モデルの尤度（クラスタリングの当てはまりの良さ）  
- \\(p\\)：パラメータ数（クラスタ数が増えると大きくなる）  
- \\(n\\)：データ数  

> BIC が大きくなるようにクラスタ数を選択。

---

### 2.2 MDL（最小記述長）
「モデルの複雑さ」と「データの説明力」のトレードオフを数値化。  
X-means では BIC または MDL を使ってクラスタ数を決定する。  

---

## 3. Python による実装例

### k-means でクラスタ数を指定
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def plot_by_kmeans(X, k=5):
    y_pred = KMeans(n_clusters=k, random_state=117117, init="random").fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker="x")
    plt.title(f"k-means, n_clusters={k}")

# サンプルデータ
n_samples = 1000
random_state = 117117
X, _ = make_blobs(
    n_samples=n_samples, random_state=random_state, cluster_std=1, centers=10
)

plot_by_kmeans(X)
```

![x-means block 1](/images/basic/clustering/x-means_block01.svg)

---

### X-means でクラスタ数を自動決定
```python
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

BAYESIAN_INFORMATION_CRITERION = 0
MINIMUM_NOISELESS_DESCRIPTION_LENGTH = 1

def plot_by_xmeans(X, c_min=3, c_max=10, criterion=BAYESIAN_INFORMATION_CRITERION, tolerance=0.025):
    initial_centers = kmeans_plusplus_initializer(X, c_min).initialize()
    xmeans_instance = xmeans(
        X, initial_centers, c_max, criterion=criterion, tolerance=tolerance
    )
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()

    for i, cluster_i in enumerate(clusters):
        X_ci = X[cluster_i]
        plt.scatter(X_ci[:, 0], X_ci[:, 1], marker="x")
    plt.title("x-means")

plot_by_xmeans(X, c_min=3, c_max=10, criterion=BAYESIAN_INFORMATION_CRITERION)
```

---

## 4. tolerance の影響
分割を許す「しきい値（tolerance）」を変えると結果が変わる。

```python
X, _ = make_blobs(n_samples=2000, random_state=117117, cluster_std=0.4, centers=10)

plt.figure(figsize=(25, 5))
for i, ti in enumerate(np.linspace(0.0001, 1, 5)):
    ti = np.round(ti, 4)
    plt.subplot(1, 10, i + 1)
    plot_by_xmeans(X, c_min=3, c_max=10, criterion=BAYESIAN_INFORMATION_CRITERION, tolerance=ti)
    plt.title(f"tol={ti}")
```

---

## 5. k-means と X-means の比較
いくつかのデータセットで、k-means（固定クラスタ数）と X-means（自動クラスタ数）を比較。

```python
for i in range(3):
    X, _ = make_blobs(
        n_samples=1000, random_state=117117,
        cluster_std=0.7, centers=5 + i * 5
    )
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_by_kmeans(X)
    plt.subplot(1, 2, 2)
    plot_by_xmeans(X, c_min=3, c_max=20)
    plt.show()
```

---

## 6. 実務でのコツ
- k-means は「クラスタ数を決める必要がある」のが弱点。  
- X-means は「BIC / MDL」を使い、データからクラスタ数を推定。  
- データ構造が複雑な場合や、クラスタ数を事前に決めにくい場合に有効。  
- ただし計算コストは k-means より大きい。  

---

## まとめ
- **k-means**：クラスタ数を事前に指定する必要がある。  
- **X-means**：クラスタ数を自動決定できる（BIC・MDL を用いる）。  
- 実務では「クラスタ数が不明」な場面で有用。  
- tolerance などのパラメータ調整によって結果が変わるため注意。  

---
