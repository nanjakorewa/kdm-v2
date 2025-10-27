---
title: "移動平均と移動標準偏差"
pre: "2.8.9 "
weight: 9
title_suffix: "変動の安定性をざっくり確認する"
---

{{< lead >}}
移動平均と移動標準偏差を重ねて描画すると、トレンドだけでなく値のばらつきまで同時に確認できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
dates = pd.date_range("2021-01-01", periods=200, freq="D")
base = 80 + 0.12 * np.arange(len(dates)) + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 20)
noise = rng.normal(0, 1.8, size=len(dates))
series = pd.Series(base + noise, index=dates)

rolling_mean = series.rolling(window=14, center=True).mean()
rolling_std = series.rolling(window=14, center=True).std()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, label="原系列", color="#94a3b8", linewidth=1)
ax.plot(rolling_mean.index, rolling_mean, label="14日移動平均", color="#2563eb", linewidth=2)
ax.fill_between(
    rolling_std.index,
    rolling_mean - rolling_std,
    rolling_mean + rolling_std,
    color="#93c5fd",
    alpha=0.35,
    label="±1σ"
)
ax.set_title("移動平均と移動標準偏差")
ax.set_xlabel("日付")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/rolling_stats.svg")
```

![plot](/images/timeseries/rolling_stats.svg)

### 読み方のポイント

- 移動平均が大きく傾くと非定常性の可能性が高く、差分やトレンド除去を検討する。
- ±1σの帯が広がる期間は変動が激しくなっているため、原因事象や季節性の変化を確認する。
- 窓幅は業務の周期（週・月など）に合わせると解釈しやすい。

