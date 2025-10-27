---
title: "季節サブシリーズプロット"
pre: "2.8.10 "
weight: 10
title_suffix: "月ごとの平均を年次で比較する"
---

{{< lead >}}
月ごとの系列を年次で並べると、季節性の形が年によってどれだけ変わるかを把握できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(14)
dates = pd.date_range("2018-01-01", periods=5 * 365, freq="D")
trend = 0.05 * np.arange(len(dates))
seasonal = 8 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise = rng.normal(0, 2.0, size=len(dates))
series = pd.Series(100 + trend + seasonal + noise, index=dates)

fig, axes = plt.subplots(3, 4, figsize=(10, 6), sharey=True)
for month, ax in zip(range(1, 13), axes.flatten()):
    monthly = series[series.index.month == month]
    ax.plot(monthly.index.year, monthly.values, marker="o", linewidth=1.2, color="#2563eb")
    ax.set_title(f"{month}月")
    ax.set_xticks(sorted(monthly.index.year.unique()))
    ax.grid(alpha=0.3)
    if month in (1, 5, 9):
        ax.set_ylabel("値")

fig.suptitle("季節サブシリーズプロット")
fig.tight_layout()
fig.savefig("static/images/timeseries/monthly_subseries.svg")
```

![plot](/images/timeseries/monthly_subseries.svg)

### 読み方のポイント

- 各月の線がほぼ平行なら季節性は安定しており、形が年によって違えば変動要因が存在する。
- 大きな離散点があれば異常年またはデータ欠損の可能性がある。
- 似た月どうしを比較することで、季節調整や特徴量作成のヒントを得られる。

