---
title: "ワッフルチャートで構成比をタイル表示"
pre: "6.3.8 "
weight: 8
title_suffix: "100 マスでシェアを直感的に見せる"
---

100 個のタイルにカテゴリのシェアを割り当てるワッフルチャートは、比率をざっくり伝えるのに便利です。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

labels = ["A プラン", "B プラン", "C プラン", "解約"]
share = np.array([0.45, 0.30, 0.18, 0.07])
colors = ["#2563eb", "#22c55e", "#fbbf24", "#ef4444"]

grid_size = 10
tiles = np.round(share * grid_size * grid_size).astype(int)
tiles[-1] = grid_size * grid_size - tiles[:-1].sum()  # 誤差調整

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect("equal")

start = 0
for color, count in zip(colors, tiles):
    for n in range(count):
        row = (start + n) // grid_size
        col = (start + n) % grid_size
        ax.add_patch(
            Rectangle((col, grid_size - 1 - row), 1, 1, facecolor=color, edgecolor="white")
        )
    start += count

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_title("プラン構成比のワッフルチャート")

legend_handles = [Rectangle((0, 0), 1, 1, color=c) for c in colors]
ax.legend(legend_handles, [f"{l} {s*100:.0f}%" for l, s in zip(labels, share)], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.12))

fig.tight_layout()
fig.savefig("static/images/visualize/bar/waffle.svg")
```

![waffle](/images/visualize/bar/waffle.svg)

### 読み方のポイント

- ブロック数を 100 個に固定すると割合が直感的に理解できる。
- 凡例にパーセントを併記することで正確な値も伝えられる。
- グリッド線を消すと柔らかい印象、残すと正方形の並びが強調される。
