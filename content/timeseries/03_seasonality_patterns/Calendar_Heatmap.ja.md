---
title: "カレンダーヒートマップ"
pre: "2.8.13 "
weight: 13
title_suffix: "月間のホットスポットを探る"
---

{{< lead >}}
日次データをカレンダー形式で並べると、曜日や月内で偏っている日を一目で発見できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(28)
dates = pd.date_range("2021-01-01", periods=365, freq="D")
trend = 0.05 * np.arange(len(dates))
weekday_effect = np.array([1.0, 0.9, 0.95, 1.1, 1.2, 1.4, 0.7])
weekday_idx = dates.weekday.to_numpy()
values = 50 + trend + weekday_effect[weekday_idx] * rng.normal(10, 2, size=len(dates))
series = pd.Series(values, index=dates)

fig, axes = plt.subplots(3, 4, figsize=(10, 6))
vmin, vmax = series.min(), series.max()

for month, ax in zip(range(1, 13), axes.flatten()):
    month_series = series[series.index.month == month]
    matrix = np.full((6, 7), np.nan)
    for date, value in month_series.items():
        week = (date.day - 1) // 7
        weekday = date.weekday()
        matrix[week, weekday] = value

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(f"{month}月")
    ax.set_xticks(np.arange(7) + 0.5)
    ax.set_xticklabels(["月", "火", "水", "木", "金", "土", "日"])
    ax.set_yticks(np.arange(6) + 0.5)
    ax.set_yticklabels([f"{w+1}週" for w in range(6)])
    ax.tick_params(axis="both", which="both", length=0)

fig.suptitle("カレンダーヒートマップ（2021年）")
fig.tight_layout()
fig.savefig("static/images/timeseries/calendar_heatmap.svg")
```

![plot](/images/timeseries/calendar_heatmap.svg)

### 読み方のポイント

- 暑い色（赤系）の日が集中している週や曜日は、需要が高い日として注目する。
- 休日やイベントが特定曜日に偏っている場合、色の偏りとして表れる。
- ヒートマップで異常な日を見つけたら、詳細分析や原因究明につなげる。
