---
title: "ホルト・ウィンター法 (Holt-Winters)"
pre: "2.8.6 "
weight: 6
title_suffix: "トレンドと季節性を同時に平滑化"
---

{{< lead >}}
ホルト・ウィンター法は指数平滑法を拡張し、トレンドと季節性を同時に更新しながら予測します。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

rng = np.random.default_rng(25)
dates = pd.date_range("2019-01-01", periods=4 * 12, freq="M")
trend = 2.0 * np.arange(len(dates))
seasonal = 12 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
noise = rng.normal(0, 3, size=len(dates))
series = pd.Series(80 + trend + seasonal + noise, index=dates)

model = ExponentialSmoothing(
    series,
    trend="add",
    seasonal="mul",
    seasonal_periods=12,
).fit()

fitted = model.fittedvalues
forecast_steps = 12
forecast_index = pd.date_range(series.index[-1] + pd.offsets.MonthEnd(), periods=forecast_steps, freq="M")
forecast = model.forecast(forecast_steps)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#94a3b8", label="実測値")
ax.plot(fitted.index, fitted, color="#2563eb", linewidth=1.6, label="フィット値")
ax.plot(forecast_index, forecast, color="#f97316", linewidth=1.6, label="12か月先予測")
ax.set_title("ホルト・ウィンター法（乗法季節モデル）")
ax.set_xlabel("年月")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/holt_winters.svg")
```

![plot](/images/timeseries/holt_winters.svg)

### 読み方のポイント

- 季節性を乗法型にすると、値が大きくなるにつれて季節振幅も大きくなるデータを表現できる。
- トレンドや季節性の種類（加法／乗法）を組み合わせて、データの性質に最も近いモデルを選ぶ。
- 平滑化パラメータ（α, β, γ）はグリッドサーチや AIC を利用してチューニングする。

