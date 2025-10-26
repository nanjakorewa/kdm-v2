---
title: "積み上げエリアチャート"
pre: "6.4.2 "
weight: 2
title_suffix: "デバイス別アクセス比率を可視化する"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` 実行時に
`static/images/visualize/line/area_chart.svg` が更新されます。
{{% /notice %}}

時間帯ごとのトラフィック量を、PC・スマホ・タブレットの 3 カテゴリで積み上げ面グラフにしました。面積がそのままアクセス量に対応するため、構成比の変化を直感的に把握できます。

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

### 読み方のポイント

- 面積が大きい時間帯がアクセスピーク。構成比の変化も同時に確認できる。
- 100% 表示にしたい場合は各系列を合計で割ると割合の推移が明確になる。
- 強調したいカテゴリがある場合は色の濃さや順序を工夫する。
