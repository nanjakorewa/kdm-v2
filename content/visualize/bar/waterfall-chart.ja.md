---
title: "ウォーターフォールチャート"
pre: "6.3.6 "
weight: 6
title_suffix: "増減の積み上げを段階的に示す"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` を実行すると
`static/images/visualize/bar/waterfall.svg` が生成されます。
{{% /notice %}}

売上の増減要因を順番に足し合わせ、最終値に至るまでの寄与を可視化します。

```python
import numpy as np
import matplotlib.pyplot as plt

labels = ["基準", "新規顧客", "既存アップセル", "解約", "値引き", "最終"]
changes = np.array([300, 80, 40, -60, -30, 0])
cumulative = np.cumsum(np.insert(changes[1:-1], 0, changes[0]))

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(labels[0], changes[0], color="#64748b")
ax.bar(labels[1:-1], changes[1:-1], bottom=cumulative, color=np.where(changes[1:-1] >= 0, "#22c55e", "#f97316"))
ax.bar(labels[-1], cumulative[-1], color="#2563eb")

for x, y in zip(labels[:-1], np.append(cumulative, cumulative[-1])):
    ax.text(x, y + 10, f"{y:.0f}", ha="center", va="bottom")

ax.set_ylabel("売上（百万円）")
ax.set_title("売上増減要因のウォーターフォール")
ax.axhline(0, color="#9ca3af", linewidth=1)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/bar/waterfall.svg")
```

![waterfall](/images/visualize/bar/waterfall.svg)

### 読み方のポイント

- 増加要因を左側、減少要因を右側に配置すると流れが自然になる。
- 基準値と最終値は異なる色で強調する。
- 途中のステップに注釈（% など）を加えると意思決定者に伝わりやすい。
