---
title: "スパイラルプロットで年内の周期を描き出す"
pre: "6.7.15 "
weight: 15
title_suffix: "極座標に日次データを並べて季節性を発見"
---

年間のトレンドを一周の円に押し込み、経過日数で螺旋状に積み重ねると、季節性や周期のズレが見えやすくなります。極座標に変換するだけでユニークな可視化ができます。

```python
import numpy as np
import matplotlib.pyplot as plt

days = np.arange(365)
theta = 2 * np.pi * days / 30  # 30日ごとに一周
r = 5 + 0.01 * days + 1.2 * np.sin(2 * np.pi * days / 7)
rng = np.random.default_rng(12)
r += rng.normal(0, 0.3, size=days.size)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="polar")
ax.plot(theta, r, color="#38bdf8", linewidth=1.2)
ax.fill_between(theta, 0, r, color="#bae6fd", alpha=0.5)

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rticks([5, 7, 9])
ax.set_title("2024年 日次ユーザー滞在時間のスパイラルプロット", pad=16)

month_ticks = np.linspace(0, 2 * np.pi, 12, endpoint=False)
for idx, ang in enumerate(month_ticks):
    ax.text(ang, r.max() + 0.6, f"{idx + 1}月", ha="center", va="center")

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/spiral-plot.svg")
```

![spiral plot](/images/visualize/advanced/spiral-plot.svg)

### 読み方のポイント
- 螺旋上で半径が大きくなるほど、値が伸びた期間。外側で太っている月がピークです。
- 周期性が強ければ、同じ角度付近で波形が繰り返されます。
- 一周を何日で割るかを調整すると、週次・月次など好みの周期で可視化できます。
