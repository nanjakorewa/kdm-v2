---
title: "スロープグラフで順位変化を一目に"
pre: "6.7.7 "
weight: 7
title_suffix: "2時点の推移を線で直感的に示す"
---

プロダクト比較で「昨年と今年でどれだけ伸びたか」を示したいときは、2時点を線で結ぶスロープグラフが有効です。順位と変化幅を同時に伝えられます。

```python
import matplotlib.pyplot as plt
import numpy as np

brands = ["サービスA", "サービスB", "サービスC", "サービスD", "サービスE"]
score_2023 = np.array([62, 55, 48, 44, 38])
score_2024 = np.array([75, 58, 64, 40, 42])
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(brands)))

fig, ax = plt.subplots(figsize=(6, 4))
for idx, name in enumerate(brands):
    ax.plot(
        [0, 1],
        [score_2023[idx], score_2024[idx]],
        color=colors[idx],
        linewidth=2.5,
    )
    ax.scatter([0, 1], [score_2023[idx], score_2024[idx]], color=colors[idx], s=60)
    ax.text(
        -0.05,
        score_2023[idx],
        f"{name} {score_2023[idx]:.0f}",
        ha="right",
        va="center",
    )
    ax.text(
        1.05,
        score_2024[idx],
        f"{score_2024[idx]:.0f}",
        ha="left",
        va="center",
    )

ax.set_xticks([0, 1], labels=["2023年", "2024年"])
ax.set_title("主要サービスのNPS変化（ポイント）")
ax.set_ylim(30, 80)
ax.spines[["top", "right", "bottom"]].set_visible(False)
ax.tick_params(left=False, bottom=False)
ax.set_yticks([])
ax.grid(axis="y", alpha=0.15)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/slopegraph.svg")
```

![slopegraph](/images/visualize/advanced/slopegraph.svg)

### 読み方のポイント
- 左右の高さで各時点のスコアが、線の傾きで伸び幅が分かります。
- 線が交差していればランキング逆転。注目ポイントとして解説すると伝わりやすいです。
- 変化を強調したい場合は順序を変化量順に並べ替えるのも手です。
