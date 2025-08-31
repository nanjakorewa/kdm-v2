---
title: "Kernel-PCA"
pre: "2.6.4 "
weight: 4
title_suffix: "Implementing in Python"
---

{{< katex />}}

<div class="pagetop-box">
  <p><b>カーネル主成分分析（Kernel PCA）</b>は、PCAを非線形に拡張した手法です。  
  通常のPCAは「直線的な軸」でしか次元削減できませんが、Kernel PCAは「非線形な写像」を用いることで、曲線的な分離や複雑な構造をとらえることができます。</p>
</div>

---

## 1. 直感：非線形構造を扱うPCA
- PCAは「分散が最大の方向」を探すが、線形構造しか表現できない。  
- データが「円」や「渦巻き」のように非線形構造を持つと、PCAではうまく分けられない。  
- Kernel PCAでは、データを高次元空間に写像してからPCAを適用する。  

---

## 2. 数式でみる Kernel PCA

データ点 \(x_i \in \mathbb{R}^d\) を非線形写像 \(\phi(x)\) によって高次元空間へ移す。  

1. **カーネル行列**  
   $$
   K_{ij} = \langle \phi(x_i), \phi(x_j) \rangle
   $$
   カーネル関数 \(k(x_i, x_j)\) を直接計算することで、高次元空間に行かずに内積を得る。

2. **固有値問題**  
   $$
   K v = \lambda v
   $$
   を解くことで、非線形構造に対応した主成分を得る。

代表的なカーネル関数：
- RBFカーネル（ガウシアン）：
  $$
  k(x, x') = \exp(-\gamma \|x - x'\|^2)
  $$
- 多項式カーネル：
  $$
  k(x, x') = (\langle x, x' \rangle + c)^d
  $$

---

## 3. 実験用データ（円環構造）

{{% notice document %}}
[sklearn.datasets.make_circles](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)
{{% /notice %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=400, factor=0.3, noise=0.15)

plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("円環状データ")
plt.show()
```

![png](/images/basic/dimensionality_reduction/Kernel-PCA_files/Kernel-PCA_4_1.png)

---

## 4. Kernel PCAで次元削減

{{% notice document %}}
[sklearn.decomposition.KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)
{{% /notice %}}

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(kernel="rbf", gamma=3)
X_kpca = kpca.fit_transform(X)

plt.figure(figsize=(8, 8))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.title("Kernel PCAによる2次元化")
plt.show()
```

![png](/images/basic/dimensionality_reduction/Kernel-PCA_files/Kernel-PCA_6_1.png)

> 通常のPCAでは分離できなかった「同心円のクラス」が、Kernel PCAではきれいに分離できる。

---

## 5. 実務での応用
- **非線形構造の可視化**（例：円形・渦巻きデータ）  
- **パターン認識**：手書き文字や顔画像の特徴抽出  
- **非線形分類の前処理**：SVMなどと組み合わせ  

---

## 発展：Kernel PCAの注意点
- **カーネル選択が重要**：データに合うカーネルを選ぶ必要がある。  
- **計算コスト**：カーネル行列は \(O(n^2)\) の計算量で大規模データでは重い。  
- **固有値の解釈が難しい**：通常のPCAのように「寄与率」を直接解釈しにくい。  

---

## まとめ
- Kernel PCAは「非線形構造を扱えるPCA」。  
- 数式的には「カーネルトリック」で高次元に写像してPCAを実行。  
- 円環状や渦巻きのようなデータを分離できるのが大きな強み。  
- 実務では可視化や非線形分類の前処理に有効。  

---
