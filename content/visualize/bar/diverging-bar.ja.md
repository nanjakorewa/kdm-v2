---
title: "ダイバージング横棒で差分を表す"
pre: "6.3.5 "
weight: 5
title_suffix: "プラスとマイナスを一目で把握"
---

前年との増減を横向きの棒で表現すると、プラス・マイナスの傾向がすぐに読み取れます。

```python
import numpy as np
import matplotlib.pyplot as plt

departments = ["営業", "開発", "サポート", "マーケ", "管理"]
change = np.array([12, -5, 8, -3, 4])  # 前年比 (ポイント)
colors = np.where(change >= 0, "#10b981", "#f87171")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.barh(departments, change, color=colors)

ax.axvline(0, color="#9ca3af", linewidth=1)
ax.set_xlabel("前年差（ポイント）")
ax.set_title("部門別 NPS の前年差")
ax.bar_label(bars, fmt=lambda v: f"{v:+.0f}", padding=4)
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/bar/diverging_bar.svg")
```

![diverging bar](/images/visualize/bar/diverging_bar.svg)

### 読み方のポイント

- 中央線を 0 にすることで増減の方向が明確になる。
- 色をプラスとマイナスで変えると、直感的に把握しやすい。
- 変化量が小さい場合はパーセント表示や注釈を加えて強調する。
