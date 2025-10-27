---
title: "パーセント変化の推移"
pre: "2.8.22 "
weight: 22
title_suffix: "増減率で動きを評価する"
---

{{< lead >}}
元の値ではなく増減率で可視化すると、トレンドの強さや変化の勢いを比較しやすくなります。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
dates = pd.date_range("2021-05-01", periods=160, freq="D")
level = 100 + np.cumsum(rng.normal(0, 1.8, size=len(dates)))
series = pd.Series(level, index=dates)

returns = series.pct_change().mul(100)
rolling_mean = returns.rolling(window=14, min_periods=5).mean()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(returns.index, returns, width=0.8, color="#cbd5f5", label="日次の増減率")
ax.plot(rolling_mean.index, rolling_mean, color="#ef4444", linewidth=2, label="14日移動平均")
ax.axhline(0, color="#475569", linewidth=1)
ax.set_title("日次パーセント変化の推移")
ax.set_xlabel("日付")
ax.set_ylabel("変化率（%）")
ax.legend()
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("static/images/timeseries/percent_change.svg")
```

![plot](/images/timeseries/percent_change.svg)

### 読み方のポイント

- 値が大きい期間でも小さい期間でも比較できるため、スケールの異なる製品や店舗を並べたいときに便利。
- 移動平均を重ねると短期のボラティリティに惑わされず、勢いの変化を追える。
- 急激な正負転換が発生したタイミングは、外部イベントやキャンペーンの影響を疑って深掘りする。

