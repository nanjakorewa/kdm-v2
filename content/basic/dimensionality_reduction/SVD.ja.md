---
title: "特異値分解（SVD）| 行列を分解して情報を圧縮する"
linkTitle: "SVD"
seo_title: "特異値分解（SVD）| 行列を分解して情報を圧縮する"
pre: "2.6.2 "
weight: 2
title_suffix: "の仕組みの説明"
---

{{< katex />}}
{{% youtube "ni1atKuoAcE" %}}

<div class="pagetop-box">
    <p><b>SVD（特異値分解）</b>とは、任意の行列を直交行列と対角行列の積に分解する手法です。  
    行列の「特徴を抽出」し、次元削減や圧縮に応用できます。ここでは画像を例に、低次元での近似を体験します。</p>
</div>

---

## 1. 直感：行列を「基本パターン」に分解する
- データ行列 \\(A\\) を「基底ベクトル」と「重み」に分解して表す方法。  
- 画像なら「ぼやけた基本的なパターン」を重ね合わせて元画像を作るイメージ。  
- 特異値が大きい要素から順に「情報量の多いパターン」を表す。

---

## 2. 数式でみる SVD

任意の行列 \\(A \in \mathbb{R}^{m \times n}\\) は次のように分解できる：

$$
A = U \Sigma V^\top
$$

- \\(U \in \mathbb{R}^{m \times m}\\)：左特異ベクトル（直交行列）  
- \\(\Sigma \in \mathbb{R}^{m \times n}\\)：特異値を対角成分にもつ対角行列  
- \\(V \in \mathbb{R}^{n \times n}\\)：右特異ベクトル（直交行列）  

特異値 \\(\sigma_1 \ge \sigma_2 \ge \dots\\) は「データの強いパターン」順に並ぶ。  
上位の特異値だけ残すことで「低ランク近似」ができる。

---

## 3. 実験用のデータ（画像）

「実験」という文字画像をグレースケール化し、行列として扱う。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy import linalg
from PIL import Image

img = Image.open("./sample.png").convert("L").resize((163, 372)).rotate(90, expand=True)
img
```

---

## 4. 特異値分解を実行

```python
X = np.asarray(img)
U, Sigma, VT = linalg.svd(X, full_matrices=True)

print(f"A: {X.shape}, U: {U.shape}, Σ:{Sigma.shape}, V^T:{VT.shape}")
```

    A: (163, 372), U: (163, 163), Σ:(163,), V^T:(372, 372)

---

## 5. 低ランク近似で画像を復元

特異値の上位だけを残すと、元の画像を少ない情報で表現できる。

```python
for rank in [1, 2, 3, 4, 5, 10, 20, 50]:
    U_i = U[:, :rank]
    Sigma_i = np.matrix(linalg.diagsvd(Sigma[:rank], rank, rank))
    VT_i = VT[:rank, :]
    temp_image = np.asarray(U_i * Sigma_i * VT_i)

    plt.title(f"rank={rank}")
    plt.imshow(temp_image, cmap="gray")
    plt.show()
```

> ランクが大きいほど細部が復元され、ランクが小さいほどぼやける。

---

## 6. 特異ベクトルの解釈

各特異値に対応する \\(U, V\\) の列は「画像のパターン」を表している。  
例えば \\(u_1\\) は「最も大きな構造」、\\(u_2\\) は「補助的な構造」を表す。

```python
total = np.zeros((163, 372))
for rank in [1, 2, 3, 4, 5]:
    U_i = U[:, :rank]
    Sigma_i = np.matrix(linalg.diagsvd(Sigma[:rank], rank, rank))
    VT_i = VT[:rank, :]

    if rank > 1:
        for ri in range(rank - 1):
            Sigma_i[ri, ri] = 0

    temp_image = np.asarray(U_i * Sigma_i * VT_i)
    total += temp_image

    plt.figure(figsize=(5, 5))
    plt.suptitle(f"$u_{rank}$ の寄与")
    plt.subplot(211)
    plt.imshow(temp_image, cmap="gray")
    plt.subplot(212)
    plt.plot(VT[0])
    plt.show()

plt.imshow(total)
```

---

## 7. 実務での応用
- **画像圧縮**：少数の特異値だけで近似し、容量削減。  
- **ノイズ除去**：小さい特異値を削除するとノイズが減る。  
- **推薦システム**：ユーザー×アイテム行列を分解し、潜在的な好みを抽出。  
- **自然言語処理**：LSA（潜在意味解析）で単語文書行列を分解。  

---

## 発展：SVDと他手法との関係
- **低ランク近似の最適性**：ランク\\(k\\)の近似 \\(A_k\\) は
  $$
  A_k = \sum_{i=1}^k \sigma_i u_i v_i^\top
  $$
  と表せ、これが Frobenius ノルムで最良の近似になることが知られている。  

- **PCAとの関係**：データ行列を標準化して SVD を行えば、PCA の主成分と一致する。  

- **計算の工夫**：大規模データではランダム化 SVD などを使って効率化。  

---

## まとめ
- SVDは「行列を直交基底とスケールに分解する」手法。  
- 特異値が大きい順に情報量が多く、上位だけ残すことで次元削減・圧縮が可能。  
- PCAや推薦システムなど、幅広い分野で応用される基礎的アルゴリズム。  

---
