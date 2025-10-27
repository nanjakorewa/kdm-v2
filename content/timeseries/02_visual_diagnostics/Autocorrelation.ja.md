---
title: "自己相関関数 (ACF) のプロット"
pre: "2.8.14 "
weight: 14
title_suffix: "モデル次数を決める手がかり"
---

{{< lead >}}
自己相関関数（ACF）は過去との関係を示し、ARIMA モデルなどの次数選択に役立ちます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

rng = np.random.default_rng(17)
dates = pd.date_range("2020-01-01", periods=200, freq="D")
values = []
prev = 0.0
for t in range(len(dates)):
    seasonal = 1.5 * np.sin(2 * np.pi * t / 12)
    prev = 0.6 * prev + seasonal + rng.normal(0, 1)
    values.append(prev)

series = pd.Series(values, index=dates)

fig, ax = plt.subplots(figsize=(7, 4))
plot_acf(series, lags=30, ax=ax, color="#2563eb")
ax.set_title("自己相関関数 (ACF)")
ax.set_xlabel("ラグ")
ax.set_ylabel("自己相関")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/acf.svg")
```

![plot](/images/timeseries/acf.svg)

### 読み方のポイント

- 棒がゆっくり減衰するなら AR 成分が強い。特定ラグで急に消えるなら MA 成分が有力。
- 週次や月次など周期があると、その周期の倍数でピークが現れる。
- 信頼区間を超えているラグのみが有意なので、モデルに含める候補になる。
