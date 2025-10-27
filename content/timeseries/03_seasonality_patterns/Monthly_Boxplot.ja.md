---
title: "月別ボックスプロット"
pre: "2.8.11 "
weight: 11
title_suffix: "季節性のばらつきを比較する"
---

{{< lead >}}
月ごとにボックスプロットを描くと、季節性の中心値とばらつきの大きさを把握できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(15)
dates = pd.date_range("2019-01-01", periods=4 * 365, freq="D")
seasonal = 5 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise = rng.normal(0, 2.2, size=len(dates))
series = pd.Series(150 + seasonal + noise, index=dates)

data = [series[series.index.month == month].values for month in range(1, 13)]

fig, ax = plt.subplots(figsize=(9, 4))
ax.boxplot(data, labels=[f"{m}月" for m in range(1, 13)], patch_artist=True)

colors = plt.cm.Blues(np.linspace(0.4, 0.8, 12))
for patch, color in zip(ax.artists, colors):
    patch.set_facecolor(color)

ax.set_title("月別ボックスプロット")
ax.set_xlabel("月")
ax.set_ylabel("値")
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("static/images/timeseries/monthly_boxplot.svg")
```

![plot](/images/timeseries/monthly_boxplot.svg)

### 読み方のポイント

- 中央線が高い月ほど値の水準が高く、箱が広い月ほど変動が大きい。
- ひげや外れ値が極端な月は、イベントや長期休暇など特殊要因がある可能性が高い。
- 比較したい季節区間を絞って描き直すと、より詳細な議論ができる。

