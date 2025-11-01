---
title: "LDA | をpythonで実行してみる"
linkTitle: "LDA"
seo_title: "LDA | をpythonで実行してみる"
pre: "2.6.3 "
weight: 3
title_suffix: "をpythonで実行してみる"
---

{{< katex />}}

<div class="pagetop-box">
  <p><b>LDA（Linear Discriminant Analysis, 線形判別分析）</b>は、教師あり学習に基づく次元削減手法です。  
  各クラスができるだけ分離するように新しい軸（判別軸）を作り、低次元空間に射影します。  
  「データの分散」を最大化するPCAとは異なり、LDAは「クラスの分離度」を最大化するのが目的です。</p>
</div>

---

## 1. 直感：クラスを分けやすい方向を探す
- PCA：データの「ばらつき」が大きい方向を探す（ラベルなし）。  
- LDA：クラス間の距離が大きく、クラス内のばらつきが小さくなる方向を探す（ラベルあり）。  

---

## 2. 数式でみる LDA

クラス \\(C_1, C_2, \dots, C_k\\) があるとき：

1. **クラス内散布行列**（within-class scatter）：
   $$
   S_W = \sum_{j=1}^k \sum_{x_i \in C_j} (x_i - \mu_j)(x_i - \mu_j)^\top
   $$
   - 各クラス内のばらつき。

2. **クラス間散布行列**（between-class scatter）：
   $$
   S_B = \sum_{j=1}^k n_j (\mu_j - \mu)(\mu_j - \mu)^\top
   $$
   - クラス平均 \\(\mu_j\\) と全体平均 \\(\mu\\) の差。

3. **目的**：  
   射影ベクトル \\(w\\) を求め、次を最大化：
   $$
   J(w) = \frac{w^\top S_B w}{w^\top S_W w}
   $$
   これは「クラス間の分離を大きくし、クラス内のばらつきを小さくする」基準。

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

X, y = make_blobs(
    n_samples=600, n_features=3, random_state=11711, cluster_std=4, centers=3
)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c=y)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_zlabel("$x_3$")
```

![lda block 1](/images/basic/dimensionality-reduction/lda_block01.svg)

---

## 4. LDAで二次元に次元削減

{{% notice document %}}
[sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
{{% /notice %}}

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=2).fit(X, y)
X_lda = lda.transform(X)

plt.figure(figsize=(8, 8))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, alpha=0.5)
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.title("LDAによる2次元化")
plt.show()
```

![lda block 2](/images/basic/dimensionality-reduction/lda_block02.svg)

---

## 5. PCAとLDAの比較

同じデータをPCAで2次元化した場合：

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, alpha=0.5)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCAによる2次元化")
plt.show()
```

![lda block 3](/images/basic/dimensionality-reduction/lda_block03.svg)

> PCAは「ばらつき最大化」、LDAは「クラス分離最大化」という違いがある。

---

## 6. 実務での応用
- **次元削減**：分類問題の前処理として。  
- **可視化**：高次元データを2次元に落としてラベルごとにプロット。  
- **顔認識（Fisherfaces）**：PCAとLDAを組み合わせた手法が有名。  

---

## 発展：LDAの制約
- LDAの最大次元は「クラス数 - 1」。  
  （例：3クラスなら最大2次元までしか落とせない）  
- クラス分布が「ガウス分布で共分散が等しい」と仮定。  
- 多クラス拡張も可能だが、サンプル数が少ない場合は不安定になりやすい。  

---

## まとめ
- LDAは **ラベルあり次元削減** 手法。  
- 数式的には「クラス間散布を大きくし、クラス内散布を小さくする」方向を探す。  
- PCAと対比することで、教師なし（PCA）と教師あり（LDA）の違いが理解しやすい。  
- 実務でも分類前処理や可視化に活用される重要な手法。  

---
