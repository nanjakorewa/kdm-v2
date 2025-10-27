---
title: "ローリング予測の比較"
pre: "2.8.27 "
weight: 27
title_suffix: "ウォークフォワード法を図にする"
---

{{< lead >}}
ウォークフォワード検証では、学習期間を少しずつ進めながら予測を繰り返し、時点ごとの誤差や傾向を比較します。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)
dates = pd.date_range("2021-01-01", periods=300, freq="D")
trend = 0.1 * np.arange(len(dates))
seasonal = 6 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
noise = rng.normal(0, 1.8, size=len(dates))
series = pd.Series(120 + trend + seasonal + noise, index=dates)

window = 90
horizon = 7
step = 21

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#1d4ed8", linewidth=1.2, label="実測値")

for origin in range(window, len(series) - horizon, step):
    train = series.iloc[origin - window : origin]
    forecast_dates = series.index[origin : origin + horizon]
    naive_forecast = np.full(horizon, train.iloc[-1])
    ax.plot(
        forecast_dates,
        naive_forecast,
        color="#f97316",
        linewidth=1.2,
        alpha=0.7,
    )
    ax.scatter(
        forecast_dates[-1],
        naive_forecast[-1],
        color="#f97316",
        s=24,
    )

ax.set_title("ウォークフォワード検証のイメージ")
ax.set_xlabel("日付")
ax.set_ylabel("数値")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/rolling_forecast.svg")
```

![plot](/images/timeseries/rolling_forecast.svg)

### 読み方のポイント

- 学習窓を固定したまま起点だけを進めると、最新情報を反映した状態での性能変化を確認できる。
- 水平線が多いのはナイーブ予測を使っているためであり、モデルを差し替えれば線の形が変わる。
- ウォークフォワードを可視化すると「どの時点で予測が外れやすかったか」を具体的に議論できる。

