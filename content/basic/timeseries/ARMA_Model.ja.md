---
title: "ARMA モデルで周期とショックを同時に捉える"
pre: "2.8.30 "
weight: 30
title_suffix: "AR と MA の組み合わせ"
---

{{< lead >}}
ARMA モデルは自己回帰と移動平均を組み合わせ、連続的な依存と突発的な揺らぎを同時に表現します。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

rng = np.random.default_rng(22)
dates = pd.date_range("2020-01-01", periods=220, freq="D")
values = []
prev = 0.0
errors = rng.normal(0, 1, len(dates))
for t in range(len(dates)):
    prev_error = errors[t - 1] if t - 1 >= 0 else 0
    prev = 0.6 * prev + errors[t] + 0.5 * prev_error
    values.append(prev)

series = pd.Series(values, index=dates)
train = series.iloc[:-20]
test_index = series.index[-20:]

model = ARIMA(train, order=(1, 0, 1)).fit()
forecast = model.forecast(steps=20)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#cbd5f5", label="実測値（全体）")
ax.plot(train.index, train, color="#2563eb", linewidth=1.2, label="学習区間")
ax.plot(test_index, forecast, color="#f97316", linewidth=1.6, label="ARMA(1,1) 予測")
ax.set_title("ARMA モデルによる予測")
ax.set_xlabel("日付")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/arma_model.svg")
```

![plot](/images/timeseries/arma_model.svg)

### 読み方のポイント

- AR 係数が連続性を、MA 係数が突発的なショックへの追従を担っている。
- ACF と PACF の両方を確認し、どちらもゆっくり減衰する場合に ARMA を試す価値が高い。
- データが非定常なら差分をとって ARIMA や SARIMA へ発展させる。

