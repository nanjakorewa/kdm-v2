---
title: "バンプチャートで順位推移を可視化"
pre: "6.7.3 "
weight: 3
title_suffix: "ランキングの入れ替わりを線で表す"
---

四半期ごとの売上順位を線でつないだバンプチャートです。順位を上下反転して描くのがポイントです。

```python
import numpy as np
import matplotlib.pyplot as plt

quarters = ["Q1", "Q2", "Q3", "Q4"]
brands = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
ranks = np.array(
    [
        [1, 2, 3, 3],
        [3, 1, 1, 2],
        [2, 3, 2, 1],
        [4, 4, 5, 4],
        [5, 5, 4, 5],
    ]
)

fig, ax = plt.subplots(figsize=(7, 4))

for brand, rank in zip(brands, ranks):
    ax.plot(quarters, rank, marker="o", linewidth=2, label=brand)

ax.set_ylim(5.5, 0.5)
ax.set_ylabel("順位")
ax.set_title("ブランド別 売上順位の推移")
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/bump.svg")
```

![bump chart](/images/visualize/advanced/bump.svg)

### 読み方のポイント

- 順位は値が小さいほど上位のため、軸を反転させると視覚的に理解しやすい。
- ラインの色やマーカーでブランドを識別し、凡例を右側に配置して重なりを避ける。
- 特定のブランドを強調したい場合は太さや色を変えると効果的。
