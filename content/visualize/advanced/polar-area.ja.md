---
title: "極座標の面積チャート"
pre: "6.7.5 "
weight: 5
title_suffix: "季節性や周期データの可視化に使う"
---

月別の問い合わせ件数を極座標上で表現したチャートです。季節性の傾向を 360 度の円で確認できます。

```python
import numpy as np
import matplotlib.pyplot as plt

months = np.arange(12)
angles = months / 12 * 2 * np.pi
volume = np.array([120, 140, 200, 260, 310, 350, 330, 280, 220, 180, 150, 130])

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax.bar(angles, volume, width=2 * np.pi / 12, color="#3b82f6", alpha=0.7, edgecolor="white")

ax.set_xticks(angles)
ax.set_xticklabels([f"{m+1}月" for m in months])
ax.set_title("月別問い合わせ件数（極座標チャート）")
ax.set_yticks([])

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/polar_area.svg")
```

![polar area](/images/visualize/advanced/polar_area.svg)

### 読み方のポイント

- 半径が長いほど値が大きい。冬場の件数が夏場より少ないなど季節性を直感的に比較できる。
- 棒の幅（角度）を均一に保つことで月別比較がしやすくなる。
- 複数年を重ねる場合は透明度を変えるか、折れ線で表示すると見やすい。
