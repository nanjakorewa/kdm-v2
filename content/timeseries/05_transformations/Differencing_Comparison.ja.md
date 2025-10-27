---
title: "差分による定常化を比較"
pre: "2.8.18 "
weight: 18
title_suffix: "原系列と1次差分を並べて確認する"
---

{{< lead >}}
原系列と差分系列を並べると、定常化の効果や季節性の残り具合を直感的に把握できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(27)
dates = pd.date_range("2020-01-01", periods=240, freq="D")
trend = 0.2 * np.arange(len(dates))
seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
noise = rng.normal(0, 1.5, size=len(dates))
series = pd.Series(50 + trend + seasonal + noise, index=dates)

diff1 = series.diff().dropna()

fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
axes[0].plot(series.index, series, color="#2563eb")
axes[0].set_title("原系列")
axes[0].set_ylabel("値")
axes[0].grid(alpha=0.3)

axes[1].plot(diff1.index, diff1, color="#f97316")
axes[1].axhline(0, color="#475569", linewidth=1, linestyle="--")
axes[1].set_title("1次差分系列")
axes[1].set_xlabel("日付")
axes[1].set_ylabel("差分")
axes[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/differencing.svg")
```

![plot](/images/timeseries/differencing.svg)

### 読み方のポイント

- 差分後に平均がほぼゼロで変動が一定なら定常性に近づいたと判断できる。
- それでも周期的なパターンが残る場合は季節差分やさらなる変換を検討する。
- 差分しすぎるとノイズが増えるため、必要最小限の次数に留めることが大切。

