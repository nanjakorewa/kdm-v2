---
title: "季節成分とトレンドの分解"
pre: "2.8.4 "
weight: 4
title_suffix: "STL で系列を分けて理解する"
---

{{< lead >}}
季節性・トレンド・残差を分解すると、データに潜む周期と変化を個別に検証できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

rng = np.random.default_rng(10)
dates = pd.date_range("2019-01-01", periods=4 * 365, freq="D")
trend = 0.02 * np.arange(len(dates))
seasonal = 6 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise = rng.normal(0, 1.5, size=len(dates))
series = pd.Series(50 + trend + seasonal + noise, index=dates)

decomp = seasonal_decompose(series, model="additive", period=365)

fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
axes[0].plot(series.index, series, color="#1d4ed8")
axes[0].set_ylabel("観測値")
axes[0].set_title("原系列")

axes[1].plot(decomp.trend.index, decomp.trend, color="#9333ea")
axes[1].set_ylabel("トレンド")

axes[2].plot(decomp.seasonal.index, decomp.seasonal, color="#f97316")
axes[2].set_ylabel("季節")

axes[3].plot(decomp.resid.index, decomp.resid, color="#64748b")
axes[3].set_ylabel("残差")
axes[3].set_xlabel("日付")

for ax in axes:
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/seasonal_decompose.svg")
```

![plot](/images/timeseries/seasonal_decompose.svg)

### 読み方のポイント

- トレンド成分で長期的な増減を確認し、季節成分で周期性の強さを評価する。
- 残差が周期的に偏っていないかをチェックし、モデル化で残差にパターンが残っていないか確認する。
- 分解結果を踏まえて差分や季節調整を行うと、予測モデルが扱いやすくなる。

