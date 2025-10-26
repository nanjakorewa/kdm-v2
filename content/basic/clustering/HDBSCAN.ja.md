---
title: "HDBSCAN"
pre: "2.5.6 "
weight: 6
title_suffix: "密度クラスタを自動で抽出する"
---

{{< lead >}}
HDBSCAN（Hierarchical Density-Based Spatial Clustering of Applications with Noise）は、DBSCAN を階層化してクラスタ数を自動決定する密度ベース手法です。クラスタ密度が異なるデータでも柔軟にグルーピングでき、ノイズ点も明示的に扱えます。
{{< /lead >}}

---

## 1. DBSCAN との違い

- DBSCAN は `eps`（近傍半径）を固定するのに対し、HDBSCAN は密度の階層構造を探索しながら最適なクラスターを選択。
- クラスタに属するサンプルの「安定度」を算出し、不確実な点をノイズとして残す。
- 密度が高い領域だけでなく、疎な領域と密な領域が混ざったデータでも自然なクラスタを抽出。

---

## 2. Python で実行

```python
import numpy as np
import matplotlib.pyplot as plt
import hdbscan
from sklearn.datasets import make_moons

X, _ = make_moons(n_samples=400, noise=0.08, random_state=42)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=20,
    min_samples=10,
    cluster_selection_method="eom",
)
labels = clusterer.fit_predict(X)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=20)
plt.title("HDBSCAN によるクラスタリング")
plt.show()
```

ラベルが `-1` の点はノイズ扱いです。`clusterer.probabilities_` を見ると、各サンプルがクラスタに属する確からしさを確認できます。

---

## 3. ハイパーパラメータの直感

| パラメータ | 役割 | 変えるとどうなるか |
| --- | --- | --- |
| `min_cluster_size` | 1 クラスタの最小サイズ | 小さくすると細かなクラスタが生まれるがノイズも増えやすい。大きくすると安定した大きなクラスタにまとまる |
| `min_samples` | コアとみなす近傍点数（密度の堅さ） | 大きくするとノイズが増える反面、密度の高い点だけをクラスタに残せる |
| `cluster_selection_method` | クラスタ選択方式 (`eom`, `leaf`) | `eom` は安定度重視、`leaf` は階層の末端クラスタを保持して細分化 |
| `metric` | 距離指標 | `euclidean` が標準。コサイン距離など他の距離を使うと方向性に基づくクラスタリングが可能 |

`min_samples` を指定しない場合は `min_cluster_size` と同じ値が使われます。クラスタがノイジーなら `min_samples` を `min_cluster_size` より少し大きくするのがコツ。

---

## 4. クラスタの安定度を確認

```python
for cid, stability in clusterer.cluster_persistence_.items():
    print(f"Cluster {cid}: stability={stability:.2f}")
```

安定度（Persistence）が高いクラスタほど密度が一貫しており信頼度が高いと解釈できます。閾値を設けて信頼できるクラスタのみを採用する運用も可能です。

---

## 5. 実務上のヒント

- **ノイズをそのまま利用**：`-1` ラベルの点を異常値として扱うと、外れ点検知とクラスタリングを同時にこなせる。
- **特徴量のスケール**：距離ベースなので標準化は必須。特にカテゴリーの埋め込みや PCA 後のベクトルが使いやすい。
- **可視化**：UMAP と組み合わせて高次元データを 2 次元に落としてから HDBSCAN を適用する手法が人気。
- **計算量**：データ数が非常に多い場合は時間が掛かる。サブサンプリングで初期探索 → パラメータ決定後に全データへ適用すると良い。

---

## まとめ

- HDBSCAN は密度ベースのクラスタリングを階層的に行い、クラスタ数を自動で決定できる。
- `min_cluster_size` と `min_samples` の調整で、ノイズ耐性とクラスタの粒度を直感的にコントロール可能。
- ノイズ点の検出やクラスタ安定度の指標が提供されるため、実務での説明責任を果たしやすい。

---
