---
title: "自己回帰 (AR) モデルの予測"
pre: "2.8.28 "
weight: 28
title_suffix: "AutoReg で短期予測を試す"
---

{{< lead >}}
自己回帰モデルは過去の値を線形結合して未来を推定します。短期の依存が強い系列で特に有効です。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

rng = np.random.default_rng(20)
dates = pd.date_range("2020-01-01", periods=220, freq="D")
values = []
prev = 0.0
for t in range(len(dates)):
    prev = 0.7 * prev + rng.normal(0, 1)
    values.append(prev)

series = pd.Series(values, index=dates)
train = series.iloc[:-20]
test_index = series.index[-20:]

model = AutoReg(train, lags=3).fit()
forecast = model.forecast(steps=20)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#cbd5f5", label="実測値（全体）")
ax.plot(train.index, train, color="#2563eb", linewidth=1.2, label="学習区間")
ax.plot(test_index, forecast, color="#f97316", linewidth=1.6, label="AR(3) 予測")
ax.set_title("自己回帰モデルによる短期予測")
ax.set_xlabel("日付")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/ar_model.svg")
```

![plot](/images/timeseries/ar_model.svg)

### 読み方のポイント

- AR モデルは直前の値との依存が強いほど効果的。ラグ数は AIC/BIC や PACF を参考に決める。
- 予測区間が短いほど安定しやすく、長期予測は誤差が累積するので要注意。
- トレンドや季節性が強い場合は事前に差分をとるか、より複雑なモデル（ARIMA など）に切り替える。

