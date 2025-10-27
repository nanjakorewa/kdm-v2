---
title: "ラグプロットのグリッド"
pre: "2.8.16 "
weight: 16
title_suffix: "自己相関の形を視覚的に確認する"
---

{{< lead >}}
複数のラグを並べて描くと、周期性や自己相関の癖を直感的に読み取れます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(16)
dates = pd.date_range("2020-01-01", periods=300, freq="D")
series = pd.Series(
    10
    + 0.04 * np.arange(len(dates))
    + 2.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
    + rng.normal(0, 0.8, size=len(dates)),
    index=dates,
)

lags = [1, 2, 7, 30]
fig, axes = plt.subplots(2, 2, figsize=(6, 6))

for lag, ax in zip(lags, axes.flatten()):
    ax.scatter(series[:-lag], series[lag:], alpha=0.6, s=20, color="#2563eb")
    ax.set_title(f"ラグ {lag}")
    ax.set_xlabel("過去の値")
    ax.set_ylabel("現在の値")
    ax.grid(alpha=0.3)

fig.suptitle("複数ラグのラグプロット")
fig.tight_layout()
fig.savefig("static/images/timeseries/lag_grid.svg")
```

![plot](/images/timeseries/lag_grid.svg)

### 読み方のポイント

- ラグ1や2で対角線上に並ぶなら強い自己相関があり、AR モデルが効きやすい。
- ラグ7や30で円弧状に並ぶなら週次・月次の季節性がある可能性が高い。
- 点が散らばっていればそのラグに相関がないと判断できる。

