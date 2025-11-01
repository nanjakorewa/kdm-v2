---
title: "X-means"
weight: 3
pre: "2.5.3 "
title_suffix: "クラスタ数を自動推定する"
---

{{< katex />}}
{{% youtube "2hkyJcWctUA" %}}

{{% summary %}}
- X-means は k-means の初期化を賢く行い、必要に応じてクラスタを分割しながら適切なクラスタ数を推定します。
- 各クラスタを二分割したときの BIC（ベイズ情報量基準）や MDL を比較し、分割した方が良い場合のみ採用します。
- 実装は複雑に見えますが、シンプルな BIC 計算と k-means を組み合わせるだけでも挙動を再現できます。
- 人手で \\(k\\) を決めにくいケースでも、X-means ならデータの構造に合わせてクラスタ数を調整できます。
{{% /summary %}}

## 直感
k-means はクラスタ数 \\(k\\) を事前に指定する必要があります。ところが、妥当な \\(k\\) が分からない場面は少なくありません。X-means は小さな \\(k\\) で k-means を走らせ、各クラスタを二分割して BIC などのモデル選択基準を比較し、より説明力が高ければ分割を受け入れます。この手続きを繰り返すことで、データに適したクラスタ数が自動的に決まります。

## 数式で理解する
クラスタ集合 \\(\{C_1, \dots, C_k\}\\) と中心 \\(\{\mu_1, \dots, \mu_k\}\\) が得られたとき、BIC は

$$
\mathrm{BIC} = \ln L - \tfrac{p}{2} \ln n
$$

で与えられます。ここで \\(L\\) はモデルの尤度、\\(p\\) はパラメータ数、\\(n\\) はサンプル数です。クラスタを分割した場合としなかった場合の BIC を比較し、値が大きい（＝対数尤度が高くペナルティが小さい）方を採用します。こうして「分割した方がよい」クラスタだけを細分化し、自然と最適なクラスタ数に近づきます。

## Pythonで確かめる
以下では `scikit-learn` だけを使って簡易版の X-means を実装し、ランダムに決めた \\(k=4\\) の k-means と比較します。

```python
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def generate_dataset(
    n_samples: int = 1200,
    n_centers: int = 9,
    cluster_std: float = 1.1,
    random_state: int = 1707,
) -> NDArray[np.float64]:
    """Generate synthetic blobs for clustering experiments."""
    features, _ = make_blobs(
        n_samples=n_samples,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return features


def compute_bic(
    data: NDArray[np.float64],
    labels: NDArray[np.int64],
    centers: NDArray[np.float64],
) -> float:
    """Compute a BIC score for k-means style clusters."""
    n_samples, n_features = data.shape
    n_clusters = centers.shape[0]
    if n_clusters <= 0:
        raise ValueError("Number of clusters must be positive.")

    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    sse = 0.0
    for idx, center in enumerate(centers):
        if cluster_sizes[idx] == 0:
            continue
        diffs = data[labels == idx] - center
        sse += float((diffs**2).sum())

    variance = sse / max(n_samples - n_clusters, 1)
    if variance <= 0:
        variance = 1e-6

    log_likelihood = 0.0
    norm_const = -0.5 * n_features * np.log(2 * np.pi * variance)
    for size in cluster_sizes:
        if size > 0:
            log_likelihood += size * norm_const - 0.5 * (size - 1)

    n_params = n_clusters * (n_features + 1)
    bic = log_likelihood - 0.5 * n_params * np.log(n_samples)
    return float(bic)


def xmeans_split(
    data: NDArray[np.float64],
    k_max: int = 20,
    random_state: int = 42,
) -> NDArray[np.int64]:
    """Approximate X-means by recursively splitting clusters when BIC improves."""
    rng = np.random.default_rng(random_state)
    pending = [np.arange(len(data))]
    accepted: list[NDArray[np.int64]] = []

    while pending:
        indices = pending.pop()
        subset = data[indices]

        if len(indices) <= 2:
            accepted.append(indices)
            continue

        parent_labels = np.zeros(len(indices), dtype=int)
        parent_center = subset.mean(axis=0, keepdims=True)
        bic_parent = compute_bic(subset, parent_labels, parent_center)

        split_model = KMeans(
            n_clusters=2,
            n_init=10,
            max_iter=200,
            random_state=rng.integers(0, 1_000_000),
        ).fit(subset)
        bic_children = compute_bic(
            subset,
            split_model.labels_,
            split_model.cluster_centers_,
        )

        if bic_children > bic_parent and (len(accepted) + len(pending) + 1) < k_max:
            pending.append(indices[split_model.labels_ == 0])
            pending.append(indices[split_model.labels_ == 1])
        else:
            accepted.append(indices)

    labels = np.empty(len(data), dtype=int)
    for cluster_id, cluster_indices in enumerate(accepted):
        labels[cluster_indices] = cluster_id
    return labels


def run_xmeans_demo(
    random_state: int = 2024,
    initial_k: int = 4,
    k_max: int = 18,
) -> dict[str, object]:
    """Compare vanilla k-means and the simple X-means splitter."""
    data = generate_dataset(random_state=random_state)

    kmeans_labels = KMeans(
        n_clusters=initial_k,
        n_init=10,
        random_state=random_state,
    ).fit_predict(data)
    xmeans_labels = xmeans_split(data, k_max=k_max, random_state=random_state + 99)

    unique_xmeans = np.unique(xmeans_labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].scatter(data[:, 0], data[:, 1], c=kmeans_labels, cmap="tab20", s=10)
    axes[0].set_title(f"k-means (k={initial_k})")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(data[:, 0], data[:, 1], c=xmeans_labels, cmap="tab20", s=10)
    axes[1].set_title(f"X-means (k={len(unique_xmeans)})")
    axes[1].grid(alpha=0.2)

    fig.suptitle("k-means と X-means のクラスタ割り当て比較")
    fig.tight_layout()
    plt.show()

    return {
        "kmeans_clusters": int(initial_k),
        "xmeans_clusters": int(len(unique_xmeans)),
        "cluster_sizes": np.bincount(xmeans_labels).tolist(),
    }


results = run_xmeans_demo()
print(f"k-means で指定したクラスタ数: {results['kmeans_clusters']}")
print(f"X-means が推定したクラスタ数: {results['xmeans_clusters']}")
print(f"X-means 各クラスタのサイズ: {results['cluster_sizes']}")
```


![k-means と X-means の比較](/images/basic/clustering/x-means_block01_ja.png)

## 参考文献
{{% references %}}
<li>Pelleg, D., &amp; Moore, A. W. (2000). X-means: Extending k-means with Efficient Estimation of the Number of Clusters. <i>ICML</i>.</li>
<li>Bahmani, B., Moseley, B., Vattani, A., Kumar, R., &amp; Vassilvitskii, S. (2012). Scalable k-means++. <i>VLDB</i>.</li>
<li>scikit-learn developers. (2024). <i>Clustering</i>. https://scikit-learn.org/stable/modules/clustering.html</li>
{{% /references %}}
