---
title: "k近傍法 (k-NN)"
pre: "2.2.6 "
weight: 6
title_suffix: "距離に基づくシンプルな分類器"
---

{{< lead >}}
k-NN は学習時にパラメータを推定せず、予測時に近いサンプルを多数決するだけの「怠惰学習」モデルです。直感的で、高次元すぎないデータであれば堅実に機能します。
{{< /lead >}}

---

## 1. アルゴリズム

1. 学習データを丸ごと記憶
2. 予測したい点に対して距離を計算
3. 最も近い \\(k\\) 個のラベルを集計し、多数決または重み付き投票

距離にはユークリッド距離がよく使われますが、コサイン距離やマンハッタン距離に変えることも可能です。

---

## 2. Python 実装

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# まず k の違いで性能がどれくらい変わるかを観察
X_full, y_full = make_blobs(
    n_samples=600,
    centers=3,
    cluster_std=[1.1, 1.0, 1.2],
    random_state=7,
)
ks = [1, 3, 5, 7, 11]
for k in ks:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k, weights="distance"))
    scores = cross_val_score(model, X_full, y_full, cv=5)
    print(f"k={k}: CV accuracy={scores.mean():.3f} +/- {scores.std():.3f}")

# 2 次元のデータで決定境界を可視化
X_vis, y_vis = make_blobs(
    n_samples=450,
    centers=[(-2, 3), (1.8, 2.2), (0.8, -2.5)],
    cluster_std=[1.0, 0.9, 1.1],
    random_state=42,
)
vis_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))
vis_model.fit(X_vis, y_vis)

fig, ax = plt.subplots(figsize=(6, 4.5))
xx, yy = np.meshgrid(
    np.linspace(X_vis[:, 0].min() - 1.5, X_vis[:, 0].max() + 1.5, 300),
    np.linspace(X_vis[:, 1].min() - 1.5, X_vis[:, 1].max() + 1.5, 300),
)
grid = np.column_stack([xx.ravel(), yy.ravel()])
pred = vis_model.predict(grid).reshape(xx.shape)
ax.contourf(xx, yy, pred, levels=np.arange(0, 4) - 0.5, cmap="Pastel1", alpha=0.9)

scatter = ax.scatter(
    X_vis[:, 0],
    X_vis[:, 1],
    c=y_vis,
    cmap="Set1",
    edgecolor="#1f2937",
    linewidth=0.6,
)
ax.set_title("k-NN (k=5, 距離重みあり) の決定境界例")
ax.set_xlabel("特徴量 1")
ax.set_ylabel("特徴量 2")
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.grid(alpha=0.15)

legend = ax.legend(
    handles=scatter.legend_elements()[0],
    labels=[f"クラス {i}" for i in range(len(np.unique(y_vis)))],
    loc="upper right",
    frameon=True,
)
legend.get_frame().set_alpha(0.9)

fig.tight_layout()
```

![knn block 1](/images/basic/classification/knn_block01.svg)

---

## 3. ハイパーパラメータ

- \\(k\\): 小さいほど境界が細かくなるがノイズに弱い。奇数にすると多数決がタイになりにくい
- `weights="distance"`: 近い点に大きな票を与える設定。密度が異なるデータで有効
- 距離関数: `metric` パラメータで指定。スケール差がある場合は標準化必須

---

## 4. 長所と欠点

| 長所 | 欠点 |
| ---- | ---- |
| 実装が非常に簡単 | 予測時に全データとの距離を計算するため遅い |
| 非線形境界を自然に扱える | 次元の呪いで高次元になるほど距離の差が縮む |
| 学習が不要 | ノイズに敏感で、外れ値が投票結果を乱す |

---

## 5. まとめ

- k-NN は「近くのものは似ている」という仮定をそのまま利用する基本手法
- 前処理（標準化、特徴選択）と \\(k\\) の探索だけで堅実なベースラインを構築できる
- 大規模データでは kd-tree や ball-tree、近似最近傍探索を検討しましょう

---
