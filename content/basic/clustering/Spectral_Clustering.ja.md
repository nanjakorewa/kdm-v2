---
title: "スペクトラルクラスタリング"
pre: "2.3.6 "
weight: 6
title_suffix: "グラフ固有ベクトルでクラスタを切り分ける"
---

{{< lead >}}
スペクトラルクラスタリングはデータをグラフとみなし、ラプラシアン行列の固有ベクトルを使って低次元に埋め込み、そこに k-means などを適用する手法です。非凸クラスタやグラフデータに適しています。
{{< /lead >}}

---

## 1. アルゴリズム手順

1. 類似度行列 \\(W\\) を構築（ガウスカーネルなど）
2. グラフラプラシアン \\(L = D - W\\) または正規化ラプラシアンを計算
3. \\(L\\) の小さい固有値に対応する固有ベクトルを取り出し、行ごとに正規化
4. その埋め込みに対して k-means を実行

---

## 2. Python 実装

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_circles
from sklearn.cluster import SpectralClustering

X, _ = make_circles(n_samples=500, factor=0.4, noise=0.05, random_state=0)

model = SpectralClustering(
    n_clusters=2,
    affinity="rbf",
    gamma=50,
    assign_labels="kmeans",
    random_state=0,
)
labels = model.fit_predict(X)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="coolwarm", s=20)
plt.title("スペクトラルクラスタリング")
plt.tight_layout()
plt.show()
```

![ダミー図: スペクトラルクラスタリング結果](/images/placeholder_regression.png)

---

## 3. パラメータ設計

- `affinity` と `gamma`: 類似度のスケール。`nearest_neighbors` を使う方法もある
- `n_neighbors`: グラフを疎にすることでノイズに強くなる
- `assign_labels`: 埋め込み後のクラスタリング手法（`kmeans` か `discretize`）

---

## 4. 長所と短所

| 長所 | 短所 |
| ---- | ---- |
| 非凸クラスタに強い | 類似度行列の計算コストが高い |
| グラフ構造を直接扱える | ハイパーパラメータが多くチューニングが難しい |
| 埋め込みを可視化できる | 大規模データではメモリ消費が大きい |

---

## 5. まとめ

- 固有ベクトルを使った次元圧縮 + クラスタリングという構成で、非線形構造にも対応
- 類似度スケールの調整と疎グラフ化が成功の鍵
- グラフクラスタリングや画像分割など応用範囲が広い手法です

---
