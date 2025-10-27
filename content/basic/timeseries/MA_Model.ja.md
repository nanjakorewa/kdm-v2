---
title: "移動平均 (MA) モデル"
pre: "2.8.29 "
weight: 29
title_suffix: "ARIMA(0,0,q) でショックを表現する"
---

{{< lead >}}
移動平均モデルは直近の誤差の組み合わせで未来を表します。突発的な揺らぎを捉えるのに向いています。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

rng = np.random.default_rng(21)
dates = pd.date_range("2020-01-01", periods=220, freq="D")
errors = rng.normal(0, 1, len(dates))
values = []
for t in range(len(dates)):
    prev1 = errors[t - 1] if t - 1 >= 0 else 0
    prev2 = errors[t - 2] if t - 2 >= 0 else 0
    values.append(errors[t] + 0.6 * prev1 + 0.3 * prev2)

series = pd.Series(values, index=dates)

train = series.iloc[:-20]
test_index = series.index[-20:]

model = ARIMA(train, order=(0, 0, 2)).fit()
forecast = model.forecast(steps=20)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#cbd5f5", label="実測値（全体）")
ax.plot(train.index, train, color="#2563eb", linewidth=1.2, label="学習区間")
ax.plot(test_index, forecast, color="#f97316", linewidth=1.6, label="MA(2) 予測")
ax.set_title("移動平均モデルによる予測")
ax.set_xlabel("日付")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/ma_model.svg")
```

![plot](/images/timeseries/ma_model.svg)

### 読み方のポイント

- MA モデルは誤差項の組み合わせで表現するため、急なスパイクの後の戻りを表現しやすい。
- ACF が q ラグ後に急激にゼロへ近づく場合、MA(q) が適しているサインになる。
- AR 成分が必要か迷ったら、ARIMA で AR と MA を両方試し AIC/BIC を比較する。
