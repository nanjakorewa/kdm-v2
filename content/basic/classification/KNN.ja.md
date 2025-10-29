---
title: "k近傍法（k-NN）"
pre: "2.2.7 "
weight: 7
title_suffix: "距離に基づく怠惰学習"
---

{{% summary %}}
- k-NN は学習時にモデルを作らず、予測時に近いサンプルの多数決でクラスを決めるシンプルな手法。
- ハイパーパラメータは主に近傍数 \\(k\\) と距離の重み付けで、探索が容易。
- 非線形な決定境界を自然に表現できる一方、次元が高くなると距離の差が縮む「次元の呪い」に注意。
- 前処理として標準化や特徴選択を行うと距離計算が安定しやすい。
{{% /summary %}}

## 直感
「近くのサンプルは似たラベルを持つだろう」という仮定の下、予測したい点から最も近い \\(k\\) 個のサンプルを探し、多数決（または距離重み付き投票）でラベルを決めます。モデルを事前に学習する必要がないため、怠惰学習（lazy learning）と呼ばれます。

## 具体的な数式
テスト点 \\(\mathbf{x}\\) に対して訓練集合 \\(D\\) から距離 \\(d(\mathbf{x}, \mathbf{x}_i)\\) が小さい順に \\(k\\) 個を選び、クラス \\(c\\) の票数は

$$
v_c = \sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \,\mathbb{1}(y_i = c)
$$

で計算します。ここで \\(\mathcal{N}_k(\mathbf{x})\\) は近傍集合、重み \\(w_i\\) は距離の逆数などで設定できます。最も票の多いクラスを予測ラベルとします。

## Pythonを用いた実験や説明
次のコードは k の値を変えたときの交差検証精度を比較し、2 次元データで決定境界を可視化する例です。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# k の違いで性能がどれくらい変わるかを観察
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
ax.set_title("k-NN (k=5, 距離重み) の決定境界例")
ax.set_xlabel("特徴量1")
ax.set_ylabel("特徴量2")
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

## 参考文献
{{% references %}}
<li>Cover, T. M., &amp; Hart, P. E. (1967). Nearest Neighbor Pattern Classification. <i>IEEE Transactions on Information Theory</i>, 13(1), 21–27.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
