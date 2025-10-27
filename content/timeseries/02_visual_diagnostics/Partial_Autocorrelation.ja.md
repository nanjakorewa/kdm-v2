---
title: "偏自己相関 (PACF) のプロット"
pre: "2.8.15 "
weight: 15
title_suffix: "AR モデルの次数を推定する"
---

{{< lead >}}
偏自己相関 (PACF) は中間の影響を除去した相関を示し、AR モデルのラグ数を決める指標になります。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

rng = np.random.default_rng(30)
dates = pd.date_range("2020-01-01", periods=200, freq="D")
values = []
prev = 0.0
for t in range(len(dates)):
    seasonal = 1.0 * np.sin(2 * np.pi * t / 24)
    prev = 0.5 * prev + seasonal + rng.normal(0, 1)
    values.append(prev)

series = pd.Series(values, index=dates)

fig, ax = plt.subplots(figsize=(7, 4))
plot_pacf(series, lags=30, ax=ax, color="#f97316", method="ywm")
ax.set_title("偏自己相関関数 (PACF)")
ax.set_xlabel("ラグ")
ax.set_ylabel("偏自己相関")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/pacf.svg")
```

![plot](/images/timeseries/pacf.svg)

### 読み方のポイント

- 有意な棒が途切れるラグ数が AR モデルの次数の目安になる。
- ACF と合わせて確認することで、AR と MA のどちらを採用するか判断しやすい。
- 歪んだ分布や非線形性が強い場合は PACF だけで判断せず、モデルを試行しながら比較する。

