---
title: "階層的クラスタリング"
pre: "2.5.5 "
weight: 5
title_suffix: "デンドログラムで構造を読み解く"
---

{{< lead >}}
階層的クラスタリングは、サンプル同士を徐々に結合または分割しながらクラスタ構造を可視化する手法です。デンドログラムを用いることで、クラスター数を事前に決めなくても適切な切り方を検討できます。
{{< /lead >}}

---

## 1. アルゴリズムの流れ

1. 各サンプルを 1 クラスタとしてスタート。
2. 最も近いクラスタ同士を結合（凝集型）するか、最大のクラスタを分割（分割型）する。
3. すべてのサンプルが 1 つになるか、指定した距離しきい値まで繰り返す。

一般的には凝集型（Agglomerative）が使われます。距離計算の仕方やクラスタ間距離の定義（リンケージ）によって結果が変わるため、データの形状に合わせた選択が大切です。

---

## 2. Python でデンドログラムを描く

```python
import numpy as np
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=50, centers=3, cluster_std=0.7, random_state=42)

Z = linkage(X, method="ward")  # 分散最小化（Ward法）

plt.figure(figsize=(8, 4))
dendrogram(Z, truncate_mode="level", p=5)
plt.title("階層的クラスタリング（Ward）")
plt.xlabel("サンプル")
plt.ylabel("距離")
plt.show()
```

`truncate_mode="level"` と `p=5` により、上位 5 階層までに要約して表示。距離が大きく離れているところで切ると、クラスタが自然に分割できます。

---

## 3. ハイパーパラメータと直感

| パラメータ | 意味 | 調整したときの効果 |
| --- | --- | --- |
| `linkage` | クラスタ間距離の定義 (`ward`, `complete`, `average`, `single` など) | `ward` は球状クラスタに強い、`single` は鎖状に繋がりやすい、`complete` は疎なクラスタに強い |
| `affinity` | 距離指標 (`euclidean`, `manhattan`, `cosine` など) | `cosine` を使うと方向性に注目したクラスタを形成。`euclidean` は距離を重視 |
| `distance_threshold` | デンドログラムをどこで切るか | 小さくすると細かく分割、大きくすると大きなクラスタが残る。`n_clusters` と片方のみ指定 |
| `compute_full_tree` | 完全な木を保持するか | サンプル数が多いときは `False` で高速化。必要に応じて `True` にして細部を確認 |

`linkage="ward"` を選ぶと自動的に距離がユークリッドのみ対応になる点に注意。それ以外の距離を使う場合は他のリンク法を試しましょう。

---

## 4. scikit-learn でクラスタ割り当て

```python
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(
    n_clusters=3,
    linkage="average",
    affinity="euclidean",
)
labels = clustering.fit_predict(X)
```

`n_clusters` を指定すれば自動でクラスタ割り当てが得られます。`distance_threshold` と排他的なので、目的に合わせてどちらかを設定します。

---

## 5. 実務での活用

- **可視化で理解を深める**：デンドログラムをマーケティングや医療現場で共有すると、クラスタ間の距離感が伝わりやすい。
- **特徴選択とセットで**：距離指標の選び方はスケールに敏感。標準化や主成分分析と組み合わせて使うと効果的。
- **ノイズへの注意**：外れ値があるとクラスタ間距離が歪む。事前に除外するか、`complete`/`average` を使って影響を抑える。

---

## まとめ

- 階層的クラスタリングはクラスタ構造を段階的に可視化し、自然なクラスタ数を探るのに役立つ。
- リンケージと距離指標の組み合わせによって結果が大きく変わるため、データの特性を踏まえて選択する。
- デンドログラムを活用した説明がしやすく、洞察を共有しやすい手法として実務でも重宝する。

---
