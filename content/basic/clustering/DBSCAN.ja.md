---
title: "DBSCAN"
pre: "2.3.4 "
weight: 4
title_suffix: "密度ベースでクラスタを抽出する"
---

{{< lead >}}
DBSCAN（Density-Based Spatial Clustering of Applications with Noise）は密度の高い領域をクラスタとみなし、低密度の点を外れ値として扱うアルゴリズムです。クラスタ形状が不規則でも抽出でき、k-means では苦手な状況で力を発揮します。
{{< /lead >}}

---

## 1. 主要パラメータ

- \\(
arepsilon\\)（eps）: 近傍半径
- `min_samples`: コアポイントと見なすために必要な近傍点数
- 距離関数: ユークリッドが基本だが他の距離でも良い

### 用語
- **コアポイント**: 半径 \\(
arepsilon\\) 内に `min_samples` 以上の点がある
- **境界ポイント**: コアポイントには届かないが、コアポイントの近傍に存在
- **ノイズ**: どちらにも当てはまらない点

---

## 2. Python 実装

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

X, _ = make_moons(n_samples=600, noise=0.08, random_state=0)
X = StandardScaler().fit_transform(X)

model = DBSCAN(eps=0.3, min_samples=10)
labels = model.fit_predict(X)

print("クラスタ数 (ノイズ除く):", len(set(labels)) - (1 if -1 in labels else 0))
print("ノイズ点数:", np.sum(labels == -1))

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="Spectral", s=20)
plt.title("DBSCAN クラスタリング")
plt.xlabel("特徴量1")
plt.ylabel("特徴量2")
plt.tight_layout()
plt.show()
```

![dbscan block 1](/images/basic/clustering/dbscan_block01.svg)

---

## 3. パラメータの選び方

- k-距離グラフ（各点から `min_samples` 番目に近い点までの距離を降順に並べる）でヒザ状の位置が \\(
arepsilon\\) の目安
- `min_samples` は次元 + 1 以上にすることが推奨される
- スケールが異なる特徴が混在する場合は標準化が必須

---

## 4. 長所と短所

| 長所 | 短所 |
| ---- | ---- |
| クラスタ形状に制約がない | \\(
arepsilon\\) の設定が難しく、密度が大きく変わるデータに弱い |
| ノイズを自然に検出できる | 高次元では距離が均一になり性能が低下 |
| k-means のようにクラスタ数を事前指定しなくてよい | サンプルによっては結果が不安定 |

---

## 5. まとめ

- DBSCAN は「密度が高い領域=クラスタ」という直感を形式化した手法
- パラメータ探索と前処理を適切に行えば、非凸クラスタやノイズを伴うデータでも強力
- HDBSCAN などの派生もあり、より複雑な密度変化に対応できます

---
