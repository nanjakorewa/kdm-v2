---
title: "ローリング相関で系列間の関係を追う"
pre: "2.8.17 "
weight: 17
title_suffix: "相関が時間とともに変わるかを確認する"
---

{{< lead >}}
2系列の相関を固定窓で計算すると、関係が強まる期間と弱まる期間を時系列で把握できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)
dates = pd.date_range("2021-01-01", periods=260, freq="D")
driver = np.sin(2 * np.pi * np.arange(len(dates)) / 45)
series_a = 120 + 8 * driver + rng.normal(0, 2.0, size=len(dates))
series_b = 100 + 6 * driver + rng.normal(0, 2.2, size=len(dates))

rolling_corr = (
    pd.Series(series_a, index=dates)
    .rolling(window=30, min_periods=20)
    .corr(pd.Series(series_b, index=dates))
)

fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
axs[0].plot(dates, series_a, color="#2563eb", label="系列A")
axs[0].plot(dates, series_b, color="#f97316", label="系列B", alpha=0.85)
axs[0].set_ylabel("値")
axs[0].set_title("系列Aと系列Bの推移")
axs[0].legend(loc="upper left")
axs[0].grid(alpha=0.3)

axs[1].plot(rolling_corr.index, rolling_corr, color="#10b981", linewidth=1.8, label="30日ローリング相関")
axs[1].axhline(0, color="#475569", linewidth=1, linestyle=":")
axs[1].axhline(0.5, color="#94a3b8", linewidth=1, linestyle="--")
axs[1].axhline(-0.5, color="#94a3b8", linewidth=1, linestyle="--")
axs[1].set_ylabel("相関")
axs[1].set_xlabel("日付")
axs[1].set_title("ローリング相関（30日窓）")
axs[1].grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/rolling_corr.svg")
```

![plot](/images/timeseries/rolling_corr.svg)

### 読み方のポイント

- 窓幅を広げると滑らかになる一方で短期の変化を捉えにくくなる。業務サイクルに合わせて調整する。
- 相関が急に落ち込む区間は、外的イベントや季節性の変化が影響していないか確認する。
- 正負の境界線（0）を跨ぐタイミングは、関係性が反転した可能性があるサインとして監視する。

