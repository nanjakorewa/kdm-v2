---
title: "サポートベクターマシン（SVM）"
pre: "2.2.3 "
weight: 3
title_suffix: "マージン最大化で汎化性能を高める"
---

{{< lead >}}
SVM は「クラス間の境界（マージン）を最大化する」ことで汎化性能を重視する分類アルゴリズムです。線形境界から非線形カーネルまで拡張でき、特徴量が多い状況でも強力に機能します。
{{< /lead >}}

---

## 1. どんなときに使う？

- 少数のサンプルでも、境界をしっかり定義したいとき  
- 特徴量が多く、線形回帰系のモデルが過学習しやすいとき  
- カーネルトリックで非線形な決定境界を描きたいとき  
- 外れ値よりも「境界付近の重要な点（サポートベクター）」を重視したいとき

---

## 2. 線形 SVM の数理

2 クラス問題で、それぞれのラベルを \\(y_i \\in \\{ -1, +1 \\}\\) とします。線形分離可能なケースでは以下を解きます。

\\[
\\min_{\\boldsymbol{w}, b} \\; \\frac{1}{2} \\lVert \\boldsymbol{w} \\rVert_2^2
\\quad \\text{s.t.} \\quad
y_i (\\boldsymbol{w}^\\top \\mathbf{x}_i + b) \\ge 1 \\quad (i = 1, \\dots, n)
\\]

マージン幅は \\(2 / \\lVert \\boldsymbol{w} \\rVert\\)。係数のノルムを小さくすることで、より広いマージンを得ます。

### ソフトマージン

実際は完全分離できないため、スラック変数 \\(\\xi_i \\ge 0\\) を導入し、ペナルティ \\(C\\) を付けます。

\\[
\\min_{\\boldsymbol{w}, b, \\boldsymbol{\\xi}} \\; \\frac{1}{2} \\lVert \\boldsymbol{w} \\rVert_2^2 + C \\sum_{i=1}^{n} \\xi_i
\\quad \\text{s.t.} \\quad
y_i (\\boldsymbol{w}^\\top \\mathbf{x}_i + b) \\ge 1 - \\xi_i
\\]

\\(C\\) が大きいほど誤分類を厳しく抑え、小さいほどマージン優先になります。

---

## 3. カーネルトリックの直感

非線形分離を扱うために、特徴量写像 \\(\\phi(\\mathbf{x})\\) で高次元に持ち上げればよいですが、明示的にベクトルを計算するとコストが高くなります。

SVM の最適化問題は内積 \\(\\phi(\\mathbf{x}_i)^\\top \\phi(\\mathbf{x}_j)\\) だけに依存するため、

\\[
K(\\mathbf{x}_i, \\mathbf{x}_j) = \\phi(\\mathbf{x}_i)^\\top \\phi(\\mathbf{x}_j)
\\]

を直接計算できるカーネル関数（RBF、Polynomial など）を使えば、次元を明示せずに非線形境界を学習できます。

---

## 4. Python で SVM を実装する

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 非線形に分離可能なテストデータ
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

![ダミー図: RBF-SVM の決定境界](/images/placeholder_regression.png)

---

## 5. ハイパーパラメータ設計

- \\(C\\): 誤分類を許容する度合い。大きいほどハードマージンに近づき、過学習リスクが上がる  
- \\(\\gamma\\)（RBF 等のカーネル）: 影響範囲の広さ。大きいと境界が細かく曲がり、小さいと滑らかになる  
- カーネル種類: `linear`, `poly`, `rbf`, `sigmoid` など。まずは `rbf` か `linear` で十分  
- グリッドサーチや `RandomizedSearchCV` で \\(C\\), \\(\\gamma\\) を同時にクロスバリデーション

---

## 6. まとめ

- SVM は「マージン最大化」という明確な原理で汎化性能を確保する分類器  
- カーネルトリックにより、非線形問題も高次元に写像せず処理できる  
- ハイパーパラメータ \\(C\\) と \\(\\gamma\\) のバランスが性能を大きく左右するため、標準化とクロスバリデーションは必須です

---
