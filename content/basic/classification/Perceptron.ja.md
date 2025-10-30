---
title: "パーセプトロン"
pre: "2.2.3 "
weight: 3
title_suffix: "最もシンプルな線形分類器"
---

{{% summary %}}
- パーセプトロンは線形に分離可能なデータに対して有限回の更新で収束する最古参の分類アルゴリズム。
- 予測は重みベクトルと入力の内積にバイアスを加え、符号でクラスを判定するだけのシンプルな仕組み。
- 誤分類したサンプルにだけ重みを加算する更新則は、勾配降下の直感を養う入門として最適。
- 線形で分離できないデータにはそのままでは対応できず、特徴量拡張やカーネルトリックが有効になる。
{{% /summary %}}

## 直感
パーセプトロンは「誤分類したら決定境界をサンプル側に少し動かす」という考え方です。重みベクトル \(\mathbf{w}\) が決定境界の法線ベクトルを、バイアス \(b\) がオフセットを表します。学習率 \(\eta\) をかけて重みを微調整することで、境界が徐々に正しい位置へ移動します。

## 具体的な数式
予測は

$$
\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)
$$

で行います。誤分類したサンプル \((\mathbf{x}_i, y_i)\) に対しては

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta\, y_i\, \mathbf{x}_i,\qquad
b \leftarrow b + \eta\, y_i
$$

と更新します。データが線形に分離可能であれば、この更新を繰り返すだけで有限回で収束することが示されています。

## Pythonを用いた実験や説明
以下は人工データにパーセプトロンを適用し、学習の進み具合と決定境界を可視化する例です。誤分類が 0 になった時点で学習が終了します。

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

print("学習終了 w=", w, " b=", b)
print("各エポックの誤分類数:", history)

# 決定境界を描画
xx = np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200)
yy = -(w[0] * xx + b) / w[1]
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k")
plt.plot(xx, yy, color="black", linewidth=2, label="decision boundary")
plt.xlabel("特徴量 1")
plt.ylabel("特徴量 2")
plt.legend()
plt.tight_layout()
plt.show()
```

![perceptron block 1](/images/basic/classification/perceptron_block01.svg)

## 参考文献
{{% references %}}
<li>Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain. <i>Psychological Review</i>, 65(6), 386–408.</li>
<li>Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016). <i>Deep Learning</i>. MIT Press.</li>
{{% /references %}}
