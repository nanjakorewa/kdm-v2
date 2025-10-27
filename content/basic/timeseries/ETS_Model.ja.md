---
title: "ETS モデルでトレンドと季節性を分離"
pre: "2.8.32 "
weight: 32
title_suffix: "Error-Trend-Seasonal アプローチ"
---

{{< lead >}}
ETS（Error-Trend-Seasonal）は指数平滑法を拡張し、誤差・トレンド・季節性を組み合わせて予測します。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

rng = np.random.default_rng(23)
dates = pd.date_range("2018-01-01", periods=5 * 12, freq="M")
trend = 1.5 * np.arange(len(dates))
seasonal = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
noise = rng.normal(0, 2.5, size=len(dates))
series = pd.Series(100 + trend + seasonal + noise, index=dates)

train = series.iloc[:-12]
test_index = series.index[-12:]

model = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
).fit()

forecast = model.forecast(steps=12)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#cbd5f5", label="実測値（全体）")
ax.plot(train.index, train, color="#2563eb", linewidth=1.2, label="学習区間")
ax.plot(test_index, forecast, color="#f97316", linewidth=1.6, label="ETS 予測")
ax.set_title("ETS モデルによる予測")
ax.set_xlabel("年月")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/ets_model.svg")
```

![plot](/images/timeseries/ets_model.svg)

### 読み方のポイント

- トレンドと季節性を別々に更新するため、季節変動が大きいデータにも柔軟に対応できる。
- 季節性を加法型・乗法型から選べるので、変動幅がスケールに比例する場合は乗法型を検討する。
- Holt-Winters の発展形として理解すると、パラメータ（α, β, γ）の直感がつきやすい。

