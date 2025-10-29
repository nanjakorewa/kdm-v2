---
title: "リスクマトリクスで優先対策領域を明確に"
pre: "6.7.26 "
weight: 26
title_suffix: "発生確率×影響度を5段階ヒートマップに"
---

リスクアセスメントでは、発生確率と影響度の2軸で優先度を決めることが多いです。ヒートマップでマトリクス化すると、緊急で対処すべき項目が一目瞭然になります。

```python
import numpy as np
import matplotlib.pyplot as plt

levels = ["低", "やや低", "中", "やや高", "高"]
impact = ["軽微", "限定的", "中程度", "重大", "致命的"]

risk_matrix = np.array(
    [
        [1, 1, 2, 3, 4],
        [1, 2, 2, 3, 4],
        [2, 2, 3, 4, 4],
        [2, 3, 4, 4, 5],
        [3, 4, 4, 5, 5],
    ]
)

fig, ax = plt.subplots(figsize=(5.8, 5.2))
im = ax.imshow(risk_matrix, cmap="YlOrRd", vmin=1, vmax=5)

ax.set_xticks(range(len(levels)), labels=levels)
ax.set_yticks(range(len(impact)), labels=impact)
ax.set_xlabel("発生確率")
ax.set_ylabel("影響度")
ax.set_title("リスクマトリクス")

for i in range(risk_matrix.shape[0]):
    for j in range(risk_matrix.shape[1]):
        risk = risk_matrix[i, j]
        color = "white" if risk >= 4 else "#0f172a"
        ax.text(j, i, risk, ha="center", va="center", color=color, fontsize=12)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("優先度レベル", rotation=270, labelpad=15)

ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", bottom=False, left=False)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/risk-matrix.svg")
```

![risk matrix](/images/visualize/advanced/risk-matrix.svg)

### 読み方のポイント
- 右上の赤に近いセルが最優先対応領域です。リスク対応計画の順序付けに使えます。
- 左下の低リスク領域は監視のみ、といったルールを添えると運用が明確になります。
- セルに番号を振っておけば、詳細表や対応チケットへのリンクを紐づけやすくなります。
