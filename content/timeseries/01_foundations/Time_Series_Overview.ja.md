---
title: "日次データの基本プロット"
pre: "2.8.8 "
weight: 8
title_suffix: "折れ線と散布で全体感をつかむ"
---

{{< lead >}}
日次売上データにトレンドと季節性が混ざったケースを想定し、まずは素朴な折れ線と散布図で全体像をつかみます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
dates = pd.date_range("2022-01-01", periods=240, freq="D")
trend = 0.15 * np.arange(len(dates))
seasonal = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
noise = rng.normal(0, 2, size=len(dates))
series = 120 + trend + seasonal + noise

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dates, series, color="#2563eb", linewidth=1.6, label="日次売上")
ax.scatter(dates[::30], series[::30], color="#f97316", s=32, label="月末観測")
ax.set_title("日次売上の推移")
ax.set_xlabel("日付")
ax.set_ylabel("売上（千万円）")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/overview.svg")
```

![plot](/images/timeseries/overview.svg)

### 読み方のポイント

- 折れ線に散布図を重ねると、節目の挙動や外れ値を直感的に把握できる。
- 季節性とトレンドが混ざると増減の解釈が難しいため、まずは素朴な可視化でパターンを確認する。
- 明らかなギャップや欠測がないかを最初にチェックしておくと、後段の分析がスムーズになる。

