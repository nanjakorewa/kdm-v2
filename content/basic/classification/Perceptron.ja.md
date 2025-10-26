---
title: "パーセプトロン"
pre: "2.2.4 "
weight: 4
title_suffix: "最もシンプルな線形分類器"
---

{{< lead >}}
パーセプトロンは 1950 年代に登場した一次元の線形分類器で、現在のディープラーニングの原点となったモデルです。単純ながらも勾配降下や活性化関数の直感を得るのに最適です。
{{< /lead >}}

---

## 1. 基本アイデア

- 特徴量ベクトル \\(\mathbf{x} \in \mathbb{R}^d\\) に対し、重み \\(\mathbf{w}\\) とバイアス \\(b\\) を学習
- 予測は \\(\hat{y} = \mathrm{sign}(\mathbf{w}^	op \mathbf{x} + b)\\)
- 正しく分類できなかったサンプルに対して、\\(\mathbf{w}\\) をシンプルに更新

### 更新式

\[
\mathbf{w} \leftarrow \mathbf{w} + \eta \, y_i \, \mathbf{x}_i, \quad
b \leftarrow b + \eta \, y_i
\]

ここで \\(\eta\\) は学習率、\\(y_i \in \{-1, +1\}\\)。分離可能データに対しては有限回の更新で収束することが知られています。

---

## 2. Python 実装例

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=0)
y = np.where(y == 0, -1, 1)

w = np.zeros(X.shape[1])
b = 0.0
lr = 0.1
n_epochs = 20
history = []

for epoch in range(n_epochs):
    errors = 0
    for xi, target in zip(X, y):
        update = lr * target if target * (np.dot(w, xi) + b) <= 0 else 0.0
        if update != 0.0:
            w += update * xi
            b += update
            errors += 1
    history.append(errors)
    if errors == 0:
        break

print("学習終了: w=", w, " b=", b)
print("各エポックの誤分類数:", history)

# 決定境界を描画
xx = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)
yy = -(w[0] * xx + b) / w[1]
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
plt.plot(xx, yy, color="black", linewidth=2, label="decision boundary")
plt.xlabel("特徴量1")
plt.ylabel("特徴量2")
plt.legend()
plt.tight_layout()
plt.show()
```

![ダミー図: パーセプトロンの決定境界](/images/placeholder_regression.png)

---

## 3. 特徴と注意点

- **線形分離可能性**: パーセプトロンは線形分離できるデータに対してのみ理論的収束保証
- **学習率**: 大きすぎると振動、小さすぎると収束が遅い
- **活性化関数**: ステップ関数を使用するため確率出力は得られない
- **マルチクラス**: one-vs-rest で拡張できるが、より洗練されたロジスティック回帰や SVM が一般的

---

## 4. まとめ

- パーセプトロンは「誤分類したら重みを少し動かす」だけの最小限アルゴリズム
- 線形分離できないデータにはそのままでは対応できないが、カーネルトリックや多層化へ発展
- 学習過程を観察すると、線形分類器の感覚を直感的に理解できます

---
