---
title: "季節性を極座標で表現"
pre: "2.8.24 "
weight: 24
title_suffix: "月別の強弱を円形で可視化する"
---

{{< lead >}}
極座標グラフを使うと、月ごとの平均値や季節性の強弱を直感的に読み取ることができます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(9)
dates = pd.date_range("2020-01-01", periods=3 * 365, freq="D")
trend = 0.03 * np.arange(len(dates))
seasonal = 12 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise = rng.normal(0, 2.5, size=len(dates))
series = pd.Series(80 + trend + seasonal + noise, index=dates)

monthly_mean = series.groupby(series.index.month).mean()
angles = np.linspace(0, 2 * np.pi, num=13)
values = np.concatenate([monthly_mean.values, monthly_mean.values[:1]])

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection="polar")
ax.plot(angles, values, color="#2563eb", linewidth=2)
ax.fill(angles, values, color="#93c5fd", alpha=0.4)
ax.set_xticks(np.linspace(0, 2 * np.pi, num=12, endpoint=False))
ax.set_xticklabels([f"{m}月" for m in range(1, 13)])
ax.set_title("月別平均値を極座標で表示")
ax.set_rlabel_position(90)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/seasonal_polar.svg")
```

![plot](/images/timeseries/seasonal_polar.svg)

### 読み方のポイント

- 半径が大きい月ほど値が高く、低い月ほど値が低い。年周波数の強弱を一目で確認できる。
- 連続した線でつなぐことで、季節性の滑らかな変化と急激な谷や山を同時に把握できる。
- 複数年の平均を使うと長期傾向を強調でき、年度ごとに重ねれば年ごとの違いも比較できる。

