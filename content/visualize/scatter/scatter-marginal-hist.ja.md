---
title: "散布図にマージナルヒストグラムを追加"
pre: "6.5.5 "
weight: 5
title_suffix: "JointGridで柔軟にカスタマイズ"
---

`seaborn.JointGrid` を使うと、散布図と上部・右側のヒストグラムを自由に組み合わせられます。

```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")
g = sns.JointGrid(
    data=iris,
    x="petal_length",
    y="petal_width",
    height=4.5,
)
g.plot_joint(sns.scatterplot, hue=iris["species"], palette="Set2", alpha=0.7, s=50)
g.plot_marginals(sns.histplot, element="step", color="#9ca3af", alpha=0.6)

g.fig.suptitle("花弁長と花弁幅の分布", y=1.02)
g.fig.tight_layout()
g.fig.savefig("static/images/visualize/scatter/marginal_hist.svg")
```

![marginal hist](/images/visualize/scatter/marginal_hist.svg)

### 読み方のポイント

- 中央の散布図で相関を、周辺のヒストグラムで単独分布を同時に確認できる。
- `plot_joint` と `plot_marginals` を別々に呼び出すことで、プロット種類の組み合わせを細かく選べる。
- 色分けした場合は `JointGrid` 内で凡例を追加する必要があるので `ax_joint.legend()` を活用しよう。
