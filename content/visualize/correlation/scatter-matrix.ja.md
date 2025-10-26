---
title: "散布図行列で多次元を俯瞰する"
pre: "6.4.2 "
weight: 2
title_suffix: "ペアプロットで相関と分布を同時に確認"
---

`seaborn.pairplot` を使うと、特徴量の組み合わせごとの散布図と対角の分布をまとめて描画できます。クラス別に色分けすると、分類の分離度も同時に把握できます。

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

g = sns.pairplot(
    iris,
    vars=["sepal_length", "sepal_width", "petal_length", "petal_width"],
    hue="species",
    plot_kws={"alpha": 0.8, "s": 50},
    diag_kind="hist",
    corner=True,
)
g.fig.suptitle("Iris データの散布図行列", y=1.02)
g.fig.savefig("static/images/visualize/correlation/scatter_matrix.svg")
```

![scatter matrix](/images/visualize/correlation/scatter_matrix.svg)

### 使いどころ

- 分類タスクでクラスタが綺麗に分かれているかを事前に確認する。
- モデル化前に外れ値や非線形な関係がないかチェックする。
- `corner=True` にすると上三角のダブりを省けるため、大きな特徴量数でも見やすい。
