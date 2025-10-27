---
title: "欠損区間をハイライト"
pre: "2.8.20 "
weight: 20
title_suffix: "データの抜けを事前に把握する"
---

{{< lead >}}
欠損や測定停止区間を色付けすると、分析前にデータ品質を共有できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(26)
dates = pd.date_range("2021-01-01", periods=200, freq="D")
series = pd.Series(100 + np.cumsum(rng.normal(0, 1.2, size=len(dates))), index=dates)

missing_ranges = [
    (pd.Timestamp("2021-03-15"), pd.Timestamp("2021-03-28")),
    (pd.Timestamp("2021-05-20"), pd.Timestamp("2021-06-05")),
]

for start, end in missing_ranges:
    series.loc[start:end] = np.nan

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#2563eb", linewidth=1.2)

for start, end in missing_ranges:
    ax.axvspan(start, end, color="#f87171", alpha=0.3)
    ax.text(start + (end - start) / 2, series.min(), "欠損", color="#b91c1c", ha="center", va="bottom")

ax.set_title("欠損区間の可視化")
ax.set_xlabel("日付")
ax.set_ylabel("値")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/missing_highlight.svg")
```

![plot](/images/timeseries/missing_highlight.svg)

### 読み方のポイント

- 欠損区間を共有すると、補完・削除・外部データで埋めるなど次のアクションが取りやすい。
- 欠損の程度を可視化しておくと、モデルの信頼性や予測への影響を議論しやすい。
- 欠損が連続して長い場合は、モデルを再学習する前にデータ収集プロセスの改善が必要かもしれない。

