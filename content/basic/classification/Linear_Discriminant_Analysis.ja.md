---
title: "判別分析"
pre: "2.2.2 "
weight: 2
title_suffix: "の仕組みの説明"
---


{{% youtube "mw2V9rhJ0lE" %}}

<div class="pagetop-box">
  <p><b>線形判別分析（LDA: Linear Discriminant Analysis）</b>は、クラス間をもっともよく分離できる<b>射影方向</b>を見つけて、分類や次元削減に役立てる手法です。</p>
  <p>直感的には、<b>同じクラスの点をギュッと近づけ</b>、<b>異なるクラス同士は遠ざける</b>方向をデータから学習します。学習した方向に射影すると、少ない次元でもクラスが分離しやすくなります。</p>
</div>

---

## 1. 直感と数式（なぜLDAで分かれるのか）

2クラスの場合、LDAは射影ベクトル \\(\mathbf{w}\\) を選んで、スカラー \\(z=\mathbf{w}^\top \mathbf{x}\\) に写します。  
次の比（Fisher の判別基準）を最大化する \\(\mathbf{w}\\) を求めます：

$$
J(\mathbf{w}) \;=\; \frac{\text{クラス間分散}}{\text{クラス内分散}}
\;=\; \frac{\mathbf{w}^\top \mathbf{S}_B \mathbf{w}}{\mathbf{w}^\top \mathbf{S}_W \mathbf{w}},
$$

ここで、\\(\mathbf{S}_W\\) はクラス内散布行列、\\(\mathbf{S}_B\\) はクラス間散布行列です。  
直感的には **クラスの重心間距離を大きく**、**各クラス内のばらつきを小さく** する方向を選びます。

> 多クラスの場合は、最大で <b>クラス数 − 1</b> 次元までに圧縮できます（例：3クラスなら2次元まで）。

---

## 2. サンプルデータの作成と可視化

まずは 2 クラスの人工データを作成して、分布を見てみましょう。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 再現性のために乱数固定
rs = 42
X, y = make_blobs(
    n_samples=200, centers=2, n_features=2, cluster_std=2, random_state=rs
)

plt.figure(figsize=(7, 7))
plt.title("サンプルデータの散布図", fontsize=16)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
plt.xlabel("特徴 1")
plt.ylabel("特徴 2")
plt.show()
```

![linear-discriminant-analysis block 1](/images/basic/classification/linear-discriminant-analysis_block01.svg)

---

## 3. LDA を学習して、決定境界と射影方向を確認

scikit-learn の `LinearDiscriminantAnalysis` を使います。  
学習後、**決定境界**は \\(\,\mathbf{w}^\top \mathbf{x} + b = 0\,\\) の直線（2次元の場合）で表せます。

{{% notice document %}}
[sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
{{% /notice %}}

```python
# 学習
clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X, y)

# 分離超平面: w^T x + b = 0
w = clf.coef_[0]          # 形状 (2,)
b = clf.intercept_[0]     # スカラー

# 可視化用の直線（x軸の範囲に合わせて y を計算）
xs = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
# w1*x + w2*y + b = 0 -> y = -(w1/w2)*x - b/w2
ys_boundary = -(w[0]/w[1]) * xs - b / w[1]

# w ベクトルの向き（原点から伸ばして表示）
scale = 3.0
origin = X.mean(axis=0)
arrow_end = origin + scale * (w / np.linalg.norm(w))

plt.figure(figsize=(7, 7))
plt.title("LDAの決定境界と射影方向", fontsize=16)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", alpha=0.8, label="データ")
plt.plot(xs, ys_boundary, "k--", lw=1.2, label="決定境界（w^T x + b = 0）")
plt.arrow(origin[0], origin[1], (arrow_end-origin)[0], (arrow_end-origin)[1],
          head_width=0.5, length_includes_head=True, color="k", alpha=0.7, label="射影方向 w")
plt.xlabel("特徴 1")
plt.ylabel("特徴 2")
plt.legend()
plt.xlim(X[:,0].min()-2, X[:,0].max()+2)
plt.ylim(X[:,1].min()-2, X[:,1].max()+2)
plt.grid(alpha=0.25)
plt.show()
```

![linear-discriminant-analysis block 2](/images/basic/classification/linear-discriminant-analysis_block02.svg)

> メモ：もとのコードでは「\\(w\\) に垂直な直線」を描いていましたが、**決定境界は \\(w\\) に直交** します。上記のように \\(w\\) と 境界線を両方描くと、幾何の関係が直感的に掴めます。

---

## 4. LDA による次元削減（1次元へ射影）

2クラスなら 1 次元に圧縮されます。`transform` で射影後の特徴を得られます。

```python
# 1次元に変換（形状: (n_samples, 1)）
X_1d = clf.transform(X)[:, 0]

plt.figure(figsize=(8, 4))
plt.title("LDA による 1 次元への射影結果", fontsize=15)
plt.hist(X_1d[y==0], bins=20, alpha=0.7, label="クラス 0")
plt.hist(X_1d[y==1], bins=20, alpha=0.7, label="クラス 1")
plt.xlabel("射影後の特徴")
plt.legend()
plt.grid(alpha=0.25)
plt.show()
```

![linear-discriminant-analysis block 3](/images/basic/classification/linear-discriminant-analysis_block03.svg)

---

## 5. 2次元以上のデータでの例（多クラス & 次元削減）

多クラス（ここでは 3 クラス）・3 次元データを **LDA で 2 次元へ** 落としてみます。  
理論どおり、**得られる次元は最大で クラス数−1（=2）** です。

```python
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (for 3D projection)

X_3d, y_3d = make_blobs(
    n_samples=200, centers=3, n_features=3, cluster_std=3, random_state=rs
)

# 3D の分布を確認
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection="3d")
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_3d, cmap="coolwarm")
ax.set_title("3次元データの散布図", fontsize=15)
plt.show()

# LDA で 2 次元へ（3クラス → 最大2次元）
clf_3d = LinearDiscriminantAnalysis()
clf_3d.fit(X_3d, y_3d)
X_2d = clf_3d.transform(X_3d)   # 形状: (n_samples, 2)

# 2D に落とした結果を可視化
plt.figure(figsize=(7, 7))
plt.title("LDA による 2 次元への次元削減結果", fontsize=15)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_3d, cmap="coolwarm", edgecolor="k")
plt.xlabel("LDA 成分 1")
plt.ylabel("LDA 成分 2")
plt.grid(alpha=0.25)
plt.show()
```

![linear-discriminant-analysis block 4](/images/basic/classification/linear-discriminant-analysis_block04.svg)

![linear-discriminant-analysis block 4 fig 1](/images/basic/classification/linear-discriminant-analysis_block04_fig01.svg)

![linear-discriminant-analysis block 4 fig 2](/images/basic/classification/linear-discriminant-analysis_block04_fig02.svg)

---

## 6. 実務でのコツと注意点

- **前提**：各クラスが同一の分散共分散行列を持つガウス分布に近いと仮定（ズレるほど性能が落ちやすい）  
- **スケーリング**：特徴量のスケールが大きく異なる場合は標準化を（`StandardScaler` など）  
- **クラス不均衡**：`priors` で事前確率を設定できる。サンプル重み（`sample_weight`）も検討  
- **QDA との違い**：クラスごとに共分散が異なるなら **QDA（Quadratic Discriminant Analysis）** が適合することも  
- **次元削減の利点**：可視化しやすくなる／線形分類器（SVM など）の前処理としても有効

---

## まとめ

- LDA は「クラス間を最も分離する」方向を学習し、分類と次元削減の両方に使える  
- 決定境界は 2D では直線（高次元では超平面）で、\\(w^\top x + b = 0\\) で表現  
- 多クラスでは最大で <b>クラス数−1</b> 次元に圧縮  
- 仮定（ガウス性・共分散同一）と前処理（スケーリング）を意識して使うと効果的

---
