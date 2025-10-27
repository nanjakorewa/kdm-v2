---
title: "構造時系列 (UCM) モデル"
pre: "2.8.33 "
weight: 33
title_suffix: "レベル・トレンド・季節性を状態空間で表現"
---

{{< lead >}}
Unobserved Components Model（UCM）は状態空間モデルの一種で、トレンドや季節性を明示的な成分として推定します。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents

rng = np.random.default_rng(33)
dates = pd.date_range("2015-01-01", periods=8 * 12, freq="M")
trend = 1.5 * np.arange(len(dates))
cycle = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
seasonal = 6 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
noise = rng.normal(0, 3, len(dates))
series = pd.Series(200 + trend + cycle + seasonal + noise, index=dates)

model = UnobservedComponents(
    series,
    level="local linear trend",
    seasonal=12,
    cycle=True,
)
result = model.fit(disp=False)

fig, axes = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
axes[0].plot(series.index, series, color="#1d4ed8")
axes[0].set_ylabel("観測値")
axes[0].set_title("構造時系列モデルの分解")

axes[1].plot(series.index, result.level_smoothed, color="#9333ea")
axes[1].set_ylabel("レベル")

axes[2].plot(series.index, result.seasonal_smoothed, color="#f97316")
axes[2].set_ylabel("季節")

axes[3].plot(series.index, result.cycle_smoothed, color="#0ea5e9")
axes[3].set_ylabel("サイクル")
axes[3].set_xlabel("年月")

for ax in axes:
    ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/ucm_model.svg")
```

![plot](/images/timeseries/ucm_model.svg)

### 読み方のポイント

- レベル・季節・サイクル成分を別々に推定できるため、どの成分が動きの主因かが明確になる。
- 状態空間モデルとして扱えるので、カルマンフィルタで逐次推定や欠損補間にも強い。
- ハイパーパラメータの多さが難点だが、AIC/BIC や事前知識を活用して適切に制約をかけると安定する。
