---
title: "VAR モデルの多変量予測"
pre: "2.8.35 "
weight: 35
title_suffix: "複数系列の相互作用を考慮"
---

{< lead >}2 系列を VAR モデルに適用し、相互依存を考慮した短期予測を行います。{< /lead >}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

rng = np.random.default_rng(27)
dates = pd.date_range("2020-01-01", periods=120, freq="M")
x = np.sin(2 * np.pi * np.arange(len(dates)) / 12)
y = 0.5 * np.roll(x, 1) + rng.normal(0, 0.2, len(dates))
df = pd.DataFrame({"x": x + rng.normal(0, 0.1, len(dates)), "y": y}, index=dates)

model = VAR(df)
results = model.fit(2)
forecast = results.forecast(df.values[-results.k_ar:], steps=6)
forecast_index = pd.date_range(dates[-1] + pd.offsets.MonthEnd(), periods=6, freq="M")
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(df.index, df["x"], label="x 観測", color="#2563eb")
ax.plot(df.index, df["y"], label="y 観測", color="#22c55e")
ax.plot(forecast_df.index, forecast_df["x"], label="x 予測", color="#f97316", linestyle="--")
ax.plot(forecast_df.index, forecast_df["y"], label="y 予測", color="#facc15", linestyle="--")
ax.set_title("VAR モデルの予測")
ax.set_xlabel("年月")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/var_forecast.svg")
```

![plot](/images/timeseries/var_forecast.svg)

### 読み方のポイント

- VAR は系列間の相互作用を同時に扱える。
- ラグ数は情報量基準 (AIC/BIC) で選択することが多い。
- 予測結果は各系列ごとに可視化すると効果がわかりやすい。
