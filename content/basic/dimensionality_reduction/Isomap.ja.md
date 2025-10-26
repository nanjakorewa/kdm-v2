---
title: "Isomap"
pre: "2.4.6 "
weight: 6
title_suffix: "測地距離で多様体を展開する"
---

{{< lead >}}
Isomap はデータをグラフで結び、測地線距離を近似した上で多次元尺度法（MDS）を適用することで、非線形多様体を低次元に展開する手法です。
{{< /lead >}}

---

## 1. 手順

1. k 近傍グラフまたは \\(arepsilon\\) 近傍グラフを構築
2. グラフ上で最短経路距離（測地距離）を計算
3. その距離行列を MDS に渡して座標を得る

これにより "Swiss roll" のような巻いた多様体を平面に展開できます。

---

## 2. Python 実装

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import Isomap

X, color = make_swiss_roll(n_samples=1500, noise=0.05, random_state=0)
iso = Isomap(n_neighbors=10, n_components=2)
emb = iso.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 2], c=color, cmap="Spectral", s=5)
axes[0].set_title("元の Swiss roll")
axes[1].scatter(emb[:, 0], emb[:, 1], c=color, cmap="Spectral", s=5)
axes[1].set_title("Isomap 埋め込み")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.show()
```

![ダミー図: Isomap の可視化](/images/placeholder_regression.png)

---

## 3. パラメータ選び

- `n_neighbors`: 小さすぎるとグラフが分断、大きすぎると測地距離がユークリッド距離に近づく
- `n_components`: 可視化なら 2,3、圧縮なら必要次元まで
- ノイズに弱いので前処理や外れ値除去が重要

---

## 4. 長所と短所

| 長所 | 短所 |
| ---- | ---- |
| 非線形構造を保ちつつ展開 | グラフ最短経路の計算コストが高い |
| 埋め込み座標が安定 | 近傍グラフの選び方に敏感 |
| 可視化に向く | 高次元・大規模データでは重い |

---

## 5. まとめ

- Isomap は「測地距離 + MDS」というシンプルな構成で非線形多様体を扱える
- k 近傍グラフの品質が結果を左右するため、近傍数とノイズ処理が重要
- t-SNE/UMAP などと併用して多様体の理解を深めましょう

---
