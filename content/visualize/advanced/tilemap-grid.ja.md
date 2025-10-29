---
title: "タイルマップで地域スコアをコンパクト表示"
pre: "6.7.14 "
weight: 14
title_suffix: "地図代わりにグリッド上へ値を並べる"
---

日本地図のような形状を使うほどでもないが地域差を見せたい、そんなときに便利なのがタイルマップです。小さな正方形にスコアを配置するだけで比較しやすい表が作れます。

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

regions = {
    "北海道": (0, 3, 68),
    "東北": (1, 2, 58),
    "関東": (2, 2, 82),
    "中部": (2, 1, 74),
    "近畿": (3, 1, 79),
    "中国": (3, 0, 63),
    "四国": (4, 0, 55),
    "九州": (4, -1, 60),
    "沖縄": (5, -1, 52),
}

fig, ax = plt.subplots(figsize=(6, 4))
for name, (x, y, score) in regions.items():
    color = plt.cm.Blues((score - 50) / 35)
    tile = Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=1.5)
    ax.add_patch(tile)
    ax.text(
        x + 0.5,
        y + 0.5,
        f"{name}\n{score}",
        ha="center",
        va="center",
        color="white" if score > 70 else "#0f172a",
        fontsize=9,
    )

ax.set_xlim(-0.2, 6.2)
ax.set_ylim(-1.5, 4)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("地域別満足度スコアのタイルマップ")
ax.set_aspect("equal")
ax.set_frame_on(False)

sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(50, 85))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("スコア")

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/tilemap-grid.svg")
```

![tilemap grid](/images/visualize/advanced/tilemap-grid.svg)

### 読み方のポイント
- エリアの相対位置だけざっくり把握できればよい場合、詳細な地図を用意せずとも地域比較ができます。
- 枠線を白にすると、小さな差も視覚的に見分けやすくなります。
- 値のレンジを限定しておくと、極端なスコアに引っ張られずバランス良く色が割り当てられます。
