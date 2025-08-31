---
title: "k-means++"
weight: 2
pre: "2.5.2 "
title_suffix: "の仕組みの説明"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

<div class="pagetop-box">
    <p><b>k-means</b>はクラスタリングの代表的アルゴリズムですが、初期値（セントロイドの配置）によって結果が大きく変わる欠点があります。<br>
    <b>k-means++</b>は初期値の選び方を工夫することで、より安定して良い解に収束しやすくした改良手法です。</p>
</div>

---

## 1. 直感：なぜ初期値が重要？
- k-means はクラスタ中心をランダムに選ぶことが多い。  
- 初期値が悪いと「局所解」に落ち、クラスタが偏ったり精度が低くなったりする。  
- k-means++ は「離れた点を優先して初期中心に選ぶ」仕組みで、よりバランス良いスタートを切れる。  

---

## 2. k-means++ のアルゴリズム（数式）

1. 最初のクラスタ中心 \(\mu_1\) をデータからランダムに 1 点選ぶ。  
2. 残りの点について、最近傍の既存中心との距離の二乗を計算：  
   $$
   D(x) = \min_{1 \le j \le m} \|x - \mu_j\|^2
   $$
3. この \(D(x)\) に比例した確率で次の中心を選ぶ。  
   - 遠い点ほど選ばれやすい。  
4. これを \(k\) 個選ぶまで繰り返す。  

---

## 3. Python による比較実験

### コード例

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

plt.figure(figsize=(8, 8))
for i in range(3):  # 繰り返し比較
    rand_index = np.random.permutation(1000)
    X_rand = X[rand_index]

    # random 初期化
    y_pred_rnd = KMeans(
        n_clusters=5, random_state=random_state, init="random",
        max_iter=1, n_init=1
    ).fit_predict(X_rand)

    # k-means++ 初期化
    y_pred_kpp = KMeans(
        n_clusters=5, random_state=random_state, init="k-means++",
        max_iter=1, n_init=1
    ).fit_predict(X_rand)

    plt.figure(figsize=(10, 2))
    plt.subplot(1, 2, 1)
    plt.title("random 初期化")
    plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_pred_rnd, marker="x")
    plt.subplot(1, 2, 2)
    plt.title("k-means++ 初期化")
    plt.scatter(X_rand[:, 0], X_rand[:, 1], c=y_pred_kpp, marker="x")
    plt.show()
```

![png](/images/basic/clustering/k-means2_files/k-means2_5_1.png)

---

## 4. 実務でのポイント
- **scikit-learn** の `KMeans` ではデフォルトが `init="k-means++"`。  
- ほとんどの場合、`k-means++` を使えば十分安定した結果が得られる。  
- データが大規模な場合は、サンプリングして初期値を決める手法（mini-batch k-means など）も有効。  

---

## 発展：k-means++ の理論的保証
- 通常のランダム初期化では、最適解から大きく外れることがある。  
- k-means++ では、**期待値的に \(O(\log k)\) 近似保証** があることが示されている（Arthur & Vassilvitskii, 2007）。  
- つまり「完全最適解ではないが、極端に悪い初期化にはならない」ことが理論的に保証されている。  

---

## まとめ
- k-means は初期値によって結果が変わる弱点がある。  
- k-means++ は「離れた点を優先して初期中心を選ぶ」ことで安定性を改善。  
- scikit-learn ではデフォルトで k-means++ が使われているので、基本的にはそのまま使えばよい。  
- 発展的には **理論的な近似保証** もあり、実務でも安心して利用できる初期化手法。  

---
