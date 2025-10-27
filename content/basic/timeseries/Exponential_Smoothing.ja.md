---
title: "単純指数平滑法 (Simple Exponential Smoothing)"
pre: "2.8.5 "
weight: 5
title_suffix: "トレンドが弱い系列をなめらかに予測"
---

{{< lead >}}
指数平滑法は過去の値に指数的な重みをつけて平均し、滑らかな予測値を作ります。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

rng = np.random.default_rng(24)
dates = pd.date_range("2020-01-01", periods=200, freq="D")
series = pd.Series(50 + np.cumsum(rng.normal(0, 0.6, len(dates))), index=dates)

model = SimpleExpSmoothing(series).fit(smoothing_level=0.3, optimized=False)
fitted = model.fittedvalues
forecast = model.forecast(steps=14)
forecast_index = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=14, freq="D")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#cbd5f5", label="実測値")
ax.plot(fitted.index, fitted, color="#2563eb", linewidth=1.6, label="指数平滑の推定値")
ax.plot(forecast_index, forecast, color="#f97316", linewidth=1.6, label="14日先予測")
ax.set_title("単純指数平滑法の例")
ax.set_xlabel("日付")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/simple_exponential_smoothing.svg")
```

![plot](/images/timeseries/simple_exponential_smoothing.svg)

### 読み方のポイント

- 平滑係数 α（ここでは0.3）が大きいほど最新値を重視し、小さいほど長期平均を重視する。
- トレンドや季節性が弱い系列を滑らかにする基本手法で、異常検知やベースラインモデルとしても有用。
- トレンドがある場合は Holt 法、季節性がある場合は Holt-Winters へ拡張すると良い。

