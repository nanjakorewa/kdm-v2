---
title: "二軸グラフで異なる指標を同時に表示"
pre: "6.4.4 "
weight: 4
title_suffix: "売上と転換率など単位が異なるデータ向け"
---

片方の軸で売上高、もう片方でコンバージョン率を表示する例です。`Axes.twinx()` を使って右軸を追加します。

```python
import numpy as np
import matplotlib.pyplot as plt

months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep"]
revenue = np.array([120, 140, 160, 180, 195, 210])
conversion = np.array([2.3, 2.5, 2.7, 2.9, 3.0, 3.2])

fig, ax1 = plt.subplots(figsize=(6.5, 4))
ax2 = ax1.twinx()

ax1.plot(months, revenue, marker="o", color="#2563eb", label="売上")
ax2.plot(months, conversion, marker="s", color="#f97316", label="CVR")

ax1.set_ylabel("売上（百万円）", color="#2563eb")
ax1.tick_params(axis="y", labelcolor="#2563eb")
ax2.set_ylabel("CVR（%）", color="#f97316")
ax2.tick_params(axis="y", labelcolor="#f97316")
ax1.set_title("売上とコンバージョン率の推移")

lines = ax1.get_lines() + ax2.get_lines()
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left")
ax1.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/visualize/line/dual_axis.svg")
```

![dual axis](/images/visualize/line/dual_axis.svg)

### 読み方のポイント

- 目盛りの色を線と揃えると軸の対応が分かりやすい。
- スケールが極端に異なる場合はデータを標準化するか、別図に分けることも検討する。
- 過剰に使うと混乱を招くので、関係性を強調したい指標に限定する。
