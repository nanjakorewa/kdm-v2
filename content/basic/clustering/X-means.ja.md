---
title: "X-means"
weight: 3
pre: "2.5.3 "
title_suffix: "クラスタ数を自動推定する"
---

{{< katex />}}
{{% youtube "2hkyJcWctUA" %}}

{{% summary %}}
- X-means は k-means を拡張し、クラスタ数 \(k\) をデータから自動的に推定するクラスタリング手法。
- 少ないクラスタから開始し、各クラスタを二分割して BIC や MDL を比較しながら分割を受け入れるか否かを判断する。
- `pyclustering` ライブラリを利用すると、`xmeans` クラスで BIC/MDL 基準や `tolerance` の調整を行いつつクラスタを生成できる。
- k-means より計算コストは増えるが、クラスタ数を人手で決めにくいケースや異質な構造を持つデータで役立つ。
{{% /summary %}}

## 直感
k-means はクラスタ数をあらかじめ指定しなければなりません。X-means はまず少ないクラスタ数で k-means を実行し、各クラスタを 2 つに分割したときのモデル選択指標（BIC など）を比較して、分割した方がよいかを判断します。  
この分割を繰り返すことで、データに合ったクラスタ数が自動的に決まります。

## 具体的な数式
Bayesian Information Criterion (BIC) は

$$
\mathrm{BIC} = \ln L - \frac{p}{2} \ln n,
$$

ここで \(L\) はモデルの尤度、\(p\) は推定パラメータ数、\(n\) はサンプル数です。クラスタを分割した場合としない場合の BIC を比較し、値が大きい（より説明力が高い）方を採用します。  
MDL（最小記述長）も同様に、モデルの複雑さと当てはまりのバランスを見る指標として利用されます。

## Pythonを用いた実験や説明
`pyclustering` に含まれる X-means 実装を使って、クラスタ数が未知のデータを分割します。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

BIC = 0

n_samples = 1000
random_state = 117117
X, _ = make_blobs(
    n_samples=n_samples,
    random_state=random_state,
    cluster_std=1,
    centers=10,
)

def plot_by_kmeans(X, k=5):
    y_pred = KMeans(n_clusters=k, random_state=117117, init="random").fit_predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, marker="x")
    plt.title(f"k-means (k={k})")

plot_by_kmeans(X)
plt.show()
```

![x-means block 1](/images/basic/clustering/x-means_block01.svg)

```python
def plot_by_xmeans(X, c_min=3, c_max=10, criterion=BIC, tolerance=0.025):
    initial_centers = kmeans_plusplus_initializer(X, c_min).initialize()
    xm = xmeans(
        X, initial_centers, c_max,
        criterion=criterion, tolerance=tolerance
    )
    xm.process()
    clusters = xm.get_clusters()

    for cluster in clusters:
        X_cluster = X[cluster]
        plt.scatter(X_cluster[:, 0], X_cluster[:, 1], marker="x")
    plt.title("X-means")

plot_by_xmeans(X, c_min=3, c_max=10, criterion=BIC)
plt.show()
```

## 参考文献
{{% references %}}
<li>Pelleg, D., &amp; Moore, A. W. (2000). X-means: Extending k-means with Efficient Estimation of the Number of Clusters. <i>ICML</i>.</li>
<li>pyclustering developers. (2023). <i>xmeans module</i>. https://pyclustering.github.io/docs/latest/html/dd/db4/classpyclustering_1_1cluster_1_1xmeans_1_1xmeans.html</li>
{{% /references %}}
