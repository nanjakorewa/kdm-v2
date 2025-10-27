---
title: "曜日別の平均をバーで確認"
pre: "2.8.12 "
weight: 12
title_suffix: "曜日効果の有無を可視化する"
---

{{< lead >}}
曜日ごとの平均やばらつきを棒グラフで示すと、週次の季節性がどれだけ強いかを一目で判断できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(13)
dates = pd.date_range("2021-01-01", periods=210, freq="D")
weekday_effect = {0: 8, 1: 6, 2: 5, 3: 4, 4: 7, 5: 12, 6: 3}
values = 70 + np.array([weekday_effect[d.weekday()] for d in dates]) + rng.normal(0, 2.5, len(dates))

series = pd.Series(values, index=dates)
weekday_mean = series.groupby(series.index.weekday).mean()
weekday_std = series.groupby(series.index.weekday).std()

labels = ["月", "火", "水", "木", "金", "土", "日"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(labels, weekday_mean, yerr=weekday_std, color="#2563eb", alpha=0.8, capsize=5)
ax.set_title("曜日別平均と標準偏差")
ax.set_xlabel("曜日")
ax.set_ylabel("平均値")
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("static/images/timeseries/weekday_average.svg")
```

![plot](/images/timeseries/weekday_average.svg)

### 読み方のポイント

- 土曜が突出して高く、日曜が低いなど曜日による差が明確に分かる。
- 誤差バー（標準偏差）が大きい曜日は、その曜日の値が安定していないことを示唆する。
- ここで得た知見をもとに、特徴量として曜日ダミーを追加するか判断する。

