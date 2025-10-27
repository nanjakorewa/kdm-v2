---
title: "外生変数付き SARIMAX"
pre: "2.8.31 "
weight: 31
title_suffix: "プロモーション効果を加味する"
---

{{< lead >}}
SARIMAX は季節 ARIMA に外生変数（天候や施策など）を追加して予測精度を高められるモデルです。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

rng = np.random.default_rng(31)
dates = pd.date_range("2019-01-01", periods=4 * 52, freq="W")
promo = (np.sin(2 * np.pi * np.arange(len(dates)) / 8) > 0.7).astype(int)
weather = rng.normal(0, 1, len(dates))

baseline = 120 + 0.2 * np.arange(len(dates)) + 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 52)
series = baseline + 10 * promo + 3 * weather + rng.normal(0, 4, len(dates))
series = pd.Series(series, index=dates)

exog = pd.DataFrame({"promo": promo, "weather": weather}, index=dates)
train = series.iloc[:-12]
test_index = series.index[-12:]
train_exog = exog.iloc[:-12]
test_exog = exog.iloc[-12:]

model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 52), exog=train_exog)
result = model.fit(disp=False)
forecast = result.predict(start=test_index[0], end=test_index[-1], exog=test_exog)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(series.index, series, color="#cbd5f5", label="実測値（全体）")
ax.plot(train.index, train, color="#2563eb", linewidth=1.2, label="学習区間")
ax.plot(test_index, forecast, color="#f97316", linewidth=1.6, label="SARIMAX 予測")
ax.set_title("外生変数付き SARIMAX の例")
ax.set_xlabel("週")
ax.set_ylabel("売上指数")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/sarimax_exog.svg")
```

![plot](/images/timeseries/sarimax_exog.svg)

### 読み方のポイント

- 外生変数の係数を確認すると、施策や気候が売上に与える影響の大きさが定量化できる。
- 季節性と外生効果の両方を取り込めるため、プロモーションの重み付けや在庫調整の意思決定に役立つ。
- 外生データの将来値が必要になるため、予測時点で利用可能かどうかも考慮する。

