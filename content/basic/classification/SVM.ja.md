---
title: "サポートベクターマシン（SVM）"
pre: "2.2.5 "
weight: 5
title_suffix: "マージン最大化で汎化性能を高める"
---

{{% summary %}}
- SVM はクラス間のマージンを最大化する決定境界を学習し、汎化性能を重視した分類器を構成する。
- ソフトマージンにより誤分類を許容しつつ、ペナルティ係数 \\(C\\) でバランスを制御できる。
- カーネルトリックを用いることで、高次元に写像しなくても非線形な決定境界を扱える。
- 標準化とハイパーパラメータ探索（\\(C\\), \\(\gamma\\)）が性能向上の鍵。
{{% /summary %}}

## 直感
SVM はクラスを分ける境界の中でも、サンプルから最も離れた（マージンが最大の）境界を選びます。境界に接するサンプルはサポートベクターと呼ばれ、彼らだけが最終的な境界を決めます。こうすることで、多少のノイズに強い堅牢な決定境界が得られます。

## 具体的な数式
線形分離可能な場合は次の最適化問題を解きます。

$$
\min_{\mathbf{w}, b} \ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1
$$

実データでは完全には分離できないことが多いため、スラック変数 \\(\xi_i \ge 0\\) を導入したソフトマージン SVM を用います。

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}}
\ \frac{1}{2} \lVert \mathbf{w} \rVert_2^2 + C \sum_{i=1}^{n} \xi_i
\quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i
$$

カーネルトリックでは内積 \\(\mathbf{x}_i^\top \mathbf{x}_j\\) をカーネル \\(K(\mathbf{x}_i, \mathbf{x}_j)\\) に置き換えることで、非線形境界を表現できます。

## Pythonを用いた実験や説明
以下は `make_moons` で生成した非線形データに SVM を適用し、線形カーネルと RBF カーネルを比較する例です。RBF カーネルの方が複雑な境界を表現できることがわかります。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 非線形に分離可能なサンプルを生成
X, y = make_moons(n_samples=400, noise=0.25, random_state=42)

# 線形カーネル
linear_clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))
linear_clf.fit(X, y)

# RBF カーネル
rbf_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=5.0, gamma=0.5))
rbf_clf.fit(X, y)

print("Linear kernel stats:")
print(classification_report(y, linear_clf.predict(X)))

print("RBF kernel stats:")
print(classification_report(y, rbf_clf.predict(X)))

# 決定境界を描画
grid_x, grid_y = np.meshgrid(
    np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
    np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200),
)
grid = np.c_[grid_x.ravel(), grid_y.ravel()]

rbf_scores = rbf_clf.predict(grid).reshape(grid_x.shape)

plt.figure(figsize=(6, 5))
plt.contourf(grid_x, grid_y, rbf_scores, alpha=0.2, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=30)
plt.title("RBF-SVM の決定境界")
plt.xlabel("特徴量1")
plt.ylabel("特徴量2")
plt.tight_layout()
plt.show()
```

![svm block 1](/images/basic/classification/svm_block01.svg)

## 参考文献
{{% references %}}
<li>Vapnik, V. (1998). <i>Statistical Learning Theory</i>. Wiley.</li>
<li>Smola, A. J., &amp; Schölkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199–222.</li>
{{% /references %}}
