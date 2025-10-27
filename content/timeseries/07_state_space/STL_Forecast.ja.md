---
title: "STL 分解とフォーキャスト"
pre: "2.8.34 "
weight: 34
title_suffix: "STLForecast で季節性を扱う"
---

{{< lead >}}
STL 分解で季節性を取り除き、残差に ARIMA などをあてて予測するのが STLForecast です。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA

rng = np.random.default_rng(11)
dates = pd.date_range("2018-01-01", periods=5 * 12, freq="M")
trend = 0.8 * np.arange(len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
noise = rng.normal(0, 3, size=len(dates))
series = pd.Series(100 + trend + seasonal + noise, index=dates)

train = series.iloc[:-12]
test_index = series.index[-12:]

model = STLForecast(train, ARIMA, period=12, model_kwargs={"order": (1, 1, 1)})
result = model.fit()
forecast = result.forecast(steps=12)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#94a3b8", label="実測値")
ax.plot(train.index, train, color="#2563eb", linewidth=1.4, label="学習区間")
ax.plot(test_index, forecast, color="#ef4444", linewidth=1.6, label="STLForecast 予測")
ax.fill_between(test_index, forecast - 6, forecast + 6, color="#fecaca", alpha=0.4, label="±6の目安")
ax.set_title("STLForecast による季節調整と予測")
ax.set_xlabel("年月")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/stl_forecast.svg")
```

![plot](/images/timeseries/stl_forecast.svg)

### 読み方のポイント

- STL 分解で季節性を先に取り除くため、ARIMA などのモデルがトレンドと残差に集中できる。
- 季節性が大きく変動するケースでもロバストに分解できるのが STL の強み。
- 予測区間の不確実性はモデルの分散推定に依存するため、必要なら分布を仮定して区間を計算する。

