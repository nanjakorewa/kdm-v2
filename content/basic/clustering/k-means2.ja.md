---
title: "k-means++"
weight: 2
pre: "2.5.2 "
title_suffix: "初期セントロイドを賢く選ぶ"
---

{{< katex />}}
{{% youtube "ff9xjGcNKX0" %}}

{{% summary %}}
- k-means++ はセントロイド同士の距離が離れるように初期化し、k-means が局所解に陥るリスクを下げます。
- 新しいセントロイドは既存のセントロイドからの距離二乗に比例する確率で選ばれるため、代表点が偏りにくくなります。
- `scikit-learn` では `KMeans(init="k-means++")` を指定するだけで利用でき、ランダム初期化との違いを容易に比較できます。
- 大規模データやオンライン処理では、k-means++ を土台にした mini-batch k-means などの派生手法が広く使われています。
{{% /summary %}}

## 直感
k-means は初期セントロイドの選び方に敏感で、悪い初期値だとクラスタが偏り、最小化したい WCSS が大きいまま収束してしまうことがあります。k-means++ では、最初の1点をランダムに選んだあと、既存のセントロイドから遠いサンプルほど高い確率で選択されるようにすることで、この問題を緩和します。

## 数式で理解する
すでに選ばれているセントロイド集合 \\(\{\mu_1, \dots, \mu_m\}\\) に対して、候補点 \\(x\\) が次のセントロイドとして選ばれる確率は

$$
P(x) = \frac{D(x)^2}{\sum_{x' \in \mathcal{X}} D(x')^2}, \qquad
D(x) = \min_{1 \le j \le m} \lVert x - \mu_j \rVert
$$

で与えられます。距離 \\(D(x)\\) が大きい、すなわち既存のセントロイドから離れた点ほど選ばれやすくなるため、クラスターの代表点が散らばり、より良い初期配置が得られます（Arthur & Vassilvitskii, 2007）。

## Pythonで確かめる
乱数初期化と k-means++ 初期化を 3 回ずつ比較し、ラベルの分布と初期収束の違いを可視化します。

```python
from __future__ import annotations

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def create_blobs_dataset(
    n_samples: int = 3000,
    n_centers: int = 8,
    cluster_std: float = 1.5,
    random_state: int = 11711,
) -> tuple[np.ndarray, np.ndarray]:
    """人工データを生成し、クラスタ初期化の比較に用いる。"""
    return make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )


def compare_initialisation_strategies(
    data: np.ndarray,
    n_clusters: int = 5,
    subset_size: int = 1000,
    n_trials: int = 3,
    random_state: int = 11711,
) -> dict[str, list[float]]:
    """k-means の初期化手法を比較し、可視化と指標を返す。

    Args:
        data: クラスタリング対象の特徴量行列。
        n_clusters: 求めるクラスタ数。
        subset_size: 各試行でサンプルするデータ数。
        n_trials: 比較する試行回数。
        random_state: 乱数シード。

    Returns:
        辞書形式でランダム初期化と k-means++ 初期化の WCSS を格納。
    """
    japanize_matplotlib.japanize()
    rng = np.random.default_rng(random_state)
    inertia_random: list[float] = []
    inertia_kpp: list[float] = []

    fig, axes = plt.subplots(
        n_trials,
        2,
        figsize=(10, 3.2 * n_trials),
        sharex=True,
        sharey=True,
    )

    for trial in range(n_trials):
        indices = rng.choice(len(data), size=subset_size, replace=False)
        subset = data[indices]

        random_model = KMeans(
            n_clusters=n_clusters,
            init="random",
            n_init=1,
            max_iter=1,
            random_state=random_state + trial,
        ).fit(subset)
        kpp_model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=1,
            max_iter=1,
            random_state=random_state + trial,
        ).fit(subset)

        inertia_random.append(float(random_model.inertia_))
        inertia_kpp.append(float(kpp_model.inertia_))

        ax_random = axes[trial, 0] if n_trials > 1 else axes[0]
        ax_kpp = axes[trial, 1] if n_trials > 1 else axes[1]

        ax_random.scatter(subset[:, 0], subset[:, 1], c=random_model.labels_, s=10)
        ax_random.set_title(f"ランダム初期化（試行{trial + 1}）")
        ax_random.grid(alpha=0.2)

        ax_kpp.scatter(subset[:, 0], subset[:, 1], c=kpp_model.labels_, s=10)
        ax_kpp.set_title(f"k-means++ 初期化（試行{trial + 1}）")
        ax_kpp.grid(alpha=0.2)

    fig.suptitle("初期化手法によるラベル分布の違い（1 イテレーション）")
    fig.tight_layout()
    plt.show()

    return {"random": inertia_random, "k-means++": inertia_kpp}


FEATURES, _ = create_blobs_dataset()
metrics = compare_initialisation_strategies(
    data=FEATURES,
    n_clusters=5,
    subset_size=1000,
    n_trials=3,
    random_state=2024,
)
for method, values in metrics.items():
    print(f"{method} の平均 WCSS: {np.mean(values):.1f}")
```


![初期化手法の比較](/images/basic/clustering/k-means2_block01_ja.png)

## 参考文献
{{% references %}}
<li>Arthur, D., &amp; Vassilvitskii, S. (2007). k-means++: The Advantages of Careful Seeding. <i>ACM-SIAM SODA</i>.</li>
<li>Bahmani, B., Moseley, B., Vattani, A., Kumar, R., &amp; Vassilvitskii, S. (2012). Scalable k-means++. <i>VLDB</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
