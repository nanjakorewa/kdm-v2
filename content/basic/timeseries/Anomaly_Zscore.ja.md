---
title: "Zスコアで異常値を検出"
pre: "2.8.25 "
weight: 25
title_suffix: "閾値を超える点をハイライトする"
---

{{< lead >}}
平均と標準偏差から Z スコアを計算し、閾値を超えた観測を異常値として可視化します。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(17)
dates = pd.date_range("2022-01-01", periods=180, freq="D")
series = 100 + np.sin(2 * np.pi * np.arange(len(dates)) / 20) * 10 + rng.normal(0, 2, len(dates))
series[60] += 25
series[140] -= 20
values = pd.Series(series, index=dates)

zscore = (values - values.mean()) / values.std()
anomalies = zscore.abs() > 2.5

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.plot(values.index, values, color="#2563eb", label="系列")
ax.scatter(values.index[anomalies], values[anomalies], color="#ef4444", s=60, label="異常値")
ax.set_title("Zスコアによる異常検知")
ax.set_xlabel("日付")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/anomaly_zscore.svg")
```

![plot](/images/timeseries/anomaly_zscore.svg)

### 読み方のポイント

- 閾値（ここでは ±2.5σ）は業務要件や許容する誤差に応じて調整する。
- 季節性が強い場合は事前に季節調整してから Z スコアを計算すると誤検知が減る。
- 異常と判定されたデータ点は、原因調査やアラート通知のトリガーとして活用する。

