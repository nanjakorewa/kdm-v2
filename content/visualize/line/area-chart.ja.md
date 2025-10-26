---
title: "積み上げエリアチャート"
pre: "6.3.2 "
weight: 2
title_suffix: "デバイス別アクセス比率を可視化する"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` を実行すると
`static/images/visualize/line/area_chart.svg` が更新されます。
{{% /notice %}}

時間帯ごとのトラフィック量を、PC・スマホ・タブレットの3カテゴリで積み上げ面グラフにしました。
面積がそのままアクセス量に対応するため、構成比の変化を直感的に把握できます。

```python
import numpy as np
import matplotlib.pyplot as plt

hours = np.array([0, 4, 8, 12, 16, 20, 24])
pc = np.array([120, 150, 180, 210, 240, 220, 180])
mobile = np.array([200, 220, 240, 260, 280, 300, 260])
tablet = np.array([80, 90, 120, 150, 180, 170, 140])

fig, ax = plt.subplots(figsize=(6, 4))
ax.stackplot(
    hours,
    tablet,
    mobile,
    pc,
    labels=["タブレット", "スマホ", "PC"],
    colors=["#bfdbfe", "#93c5fd", "#3b82f6"],
    alpha=0.85,
)
ax.set_xlim(0, 24)
ax.set_xticks(hours)
ax.set_ylabel("PV 数")
ax.set_xlabel("時間帯")
ax.set_title("時間帯別 トラフィック量")
ax.legend(loc="upper left", frameon=False)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/visualize/line/area_chart.svg")
```

![area chart](/images/visualize/line/area_chart.svg)

### 注意点

- 積み上げ面は色の順番に意味があるため、凡例の順序も揃える。
- 各カテゴリの値が極端に異なる場合は比率 100% スタックに切り替えると読みやすい。
- 面グラフは重なる順序を入れ替えることで強調したいカテゴリを前面に出せる。
