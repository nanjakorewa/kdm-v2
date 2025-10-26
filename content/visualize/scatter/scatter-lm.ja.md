---
title: "回帰直線を重ねた散布図"
pre: "6.5.3 "
weight: 3
title_suffix: "線形関係をざっくりモデル化"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` 実行で
`static/images/visualize/scatter/lm.svg` が更新されます。
{{% /notice %}}

`seaborn.lmplot` を使うと、散布図と回帰直線を同時に描画できます。信頼区間も自動で付与されます。

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
g = sns.lmplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="time",
    height=4,
    aspect=1.2,
    scatter_kws={"alpha": 0.6, "s": 40},
)
g.fig.suptitle("会計額とチップの関係（回帰線付き）", y=1.04)
g.fig.savefig("static/images/visualize/scatter/lm.svg")
```

![lm scatter](/images/visualize/scatter/lm.svg)

### 読み方のポイント

- 回帰線の傾きで関係性の強弱を判断できる。信頼区間が広い箇所はデータが少ない。
- カテゴリごとに色分けすることで、グループ間の傾向の違いが分かる。
- 非線形が疑われる場合は `order` 引数で多項式回帰を試してみる。
