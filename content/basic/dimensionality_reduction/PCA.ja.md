---
title: "PCA"
pre: "2.6.1 "
weight: 1
title_suffix: "の仕組みの説明"
---

{{< katex />}}
{{% youtube "9sn0b3tml50" %}}

<div class="pagetop-box">
    <p><b>PCA(主成分分析)</b>とは、教師なし学習に分類される「次元削減」のアルゴリズムです。  
    多くの変数の情報をなるべく損なわずに、少数の変数（主成分）に情報をまとめ直すことができます。  
    データの可視化や特徴抽出、ノイズ除去などに広く使われています。</p>
</div>

---

## 1. 直感：データの「一番ばらつく方向」を探す
- 高次元データを扱うと「特徴量が多すぎて扱いづらい」ことが多い。  
- PCAは、データが最も広がっている方向（= 分散が最大の方向）を探して、その方向を新しい軸（主成分）として使う。  
- こうすることで「情報をなるべく失わずに次元を減らす」ことができる。

---

## 2. 数式でみる PCA

データ行列 \(X \in \mathbb{R}^{n \times d}\) を平均 0 に正規化したとする。

1. 共分散行列を計算：
   $$
   \Sigma = \frac{1}{n} X^\top X
   $$

2. 固有値分解：
   $$
   \Sigma v_j = \lambda_j v_j
   $$
   - \(v_j\)：主成分ベクトル（固有ベクトル）  
   - \(\lambda_j\)：対応する分散（固有値）

3. 主成分得点（低次元データ）：
   $$
   Z = X V_k
   $$
   ここで \(V_k\) は固有値の大きい順に \(k\) 個のベクトルを並べた行列。

---

## 3. 実験用のデータ

{{% notice document %}}
[sklearn.datasets.make_blobs](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html)
{{% /notice %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=600, n_features=3, random_state=117117)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
```

![png](/images/basic/dimensionality_reduction/PCA_files/PCA_4_1.png)

---

## 4. PCAで二次元に次元削減する

{{% notice document %}}
[sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
{{% /notice %}}

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCAによる2次元化")
plt.show()
```

![png](/images/basic/dimensionality_reduction/PCA_files/PCA_6_1.png)

---

## 5. 正規化の効果
PCAは「分散の大きさ」に敏感なので、スケールの違う特徴量を持つ場合は正規化（標準化）が必須です。

### 例1: 正規化なし vs 正規化あり

```python
from sklearn.preprocessing import StandardScaler

# 特徴量のスケールがバラバラなデータ
X, y = make_blobs(
    n_samples=200, n_features=3, random_state=11711, centers=3, cluster_std=2.0
)
X[:, 1] = X[:, 1] * 1000
X[:, 2] = X[:, 2] * 0.01

X_ss = StandardScaler().fit_transform(X)

# PCA実行
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)

pca_ss = PCA(n_components=2).fit(X_ss)
X_pca_ss = pca_ss.transform(X_ss)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("正規化なし")
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, marker="x", alpha=0.6)
plt.subplot(122)
plt.title("正規化あり")
plt.scatter(X_pca_ss[:, 0], X_pca_ss[:, 1], c=y, marker="x", alpha=0.6)
plt.show()
```

![png](/images/basic/dimensionality_reduction/PCA_files/PCA_8_0.png)

---

## 6. 実務での応用
- **可視化**：高次元データを2次元に落としてプロット  
- **ノイズ除去**：情報量の少ない主成分を削除することで精度向上  
- **特徴量圧縮**：学習時間の短縮や過学習防止  
- **前処理**：クラスタリングや回帰・分類の前段で使うと有効  

---

## 発展：PCAのさらに深い話題

### 1. 寄与率と累積寄与率
- 各主成分の寄与率は、固有値の割合で計算される：
  $$
  \text{寄与率} = \frac{\lambda_j}{\sum_i \lambda_i}
  $$
- 累積寄与率が 80〜90% を超える次元数を選ぶのが一般的。

### 2. SVDによる計算
実装上は「特異値分解 (SVD)」で計算されることが多い。  
$X = U \Sigma V^\top$ の形に分解し、\(V\) の列ベクトルが主成分に対応する。

### 3. カーネルPCA
非線形な構造をとらえるために「カーネル法」を使う拡張版。  
クラスタが曲線状に広がる場合などに有効。

---

## まとめ
- PCAは「データの一番ばらつく方向」を軸に取り直す次元削減手法。  
- 数式的には「共分散行列の固有値分解」で表される。  
- 正規化は必須。寄与率を見ながら次元数を決める。  
- 発展的には SVD・カーネルPCA なども存在し、応用範囲は広い。  

---
