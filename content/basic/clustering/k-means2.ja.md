---
title: k-means++
weight: 2
pre: "2.5.2 "
title_suffix: "の仕組みの説明"
---

{{% youtube "ff9xjGcNKX0" %}}

<div class="pagetop-box">
    <p><b>k-means</b>とはクラスタリングのアルゴリズムの一種です。与えられたデータをk個のクラスタにまとめるために、クラスタごとの平均を計算↔最も近い代表点にデータを割り当てる作業を繰り返します。
    k-meansは初期値となるk個の代表点によって収束の仕方が変化します。k-means++では初期値の取り方を工夫して、より安定した結果が求められるようになります。</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
```
## k-means++とランダムな初期値選択の比較
randomとkmeans++との間のクラスタリングの結果を比較します。
k-means++は左上のクラスタを2つに分けてしまう場合があることを確認できます。
※この例ではあえて `max_iter=1` を指定しています。


```python
n_samples = 3000
random_state = 11711
X, y = make_blobs(
    n_samples=n_samples, random_state=random_state, cluster_std=1.5, centers=8
)

plt.figure(figsize=(8, 8))
for i in range(10):
    # random/k-means++を比較, クラスタリングに使用するデータは毎回ランダムに決める
    rand_index = np.random.permutation(1000)
    X_rand = X[rand_index]
    y_pred_rnd = KMeans(
        n_clusters=5, random_state=random_state, init="random", max_iter=1, n_init=1
    ).fit_predict(X_rand)
    y_pred_kpp = KMeans(
        n_clusters=5, random_state=random_state, init="k-means++", max_iter=1, n_init=1
    ).fit_predict(X_rand)

    plt.figure(figsize=(10, 2))
    plt.subplot(1, 2, 1)
    plt.title(f"random")
    plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_pred_rnd, marker="x")
    plt.subplot(1, 2, 2)
    plt.title(f"k-means++")
    plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_pred_kpp, marker="x")
    plt.show()

plt.tight_layout()
plt.show()
```


    
![png](/images/basic/clustering/k-means2_files/k-means2_5_1.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_2.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_3.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_4.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_5.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_6.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_7.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_8.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_9.png)
    



    
![png](/images/basic/clustering/k-means2_files/k-means2_5_10.png)
    
