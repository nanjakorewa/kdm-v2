---
title: "平均の変化点を可視化"
pre: "2.8.26 "
weight: 26
title_suffix: "手動で境界を描いて共有する"
---

{{< lead >}}
平均値が切り替わるポイントを可視化すると、施策や環境の変化をチームで共有しやすくなります。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(19)
dates = pd.date_range("2020-01-01", periods=240, freq="D")
segments = [
    (0, 80, 1.5),
    (70, 95, 1.8),
    (140, 110, 2.2),
]

values = np.empty(len(dates))
segment_bounds = [seg[0] for seg in segments[1:]] + [len(dates)]
for (start, mean, scale), end in zip(segments, segment_bounds):
    values[start:end] = mean + rng.normal(0, scale, end - start)

series = pd.Series(values, index=dates)
rolling = series.rolling(window=14, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(series.index, series, color="#64748b", alpha=0.7, label="観測値")
ax.plot(rolling.index, rolling, color="#2563eb", linewidth=2, label="14日移動平均")

for start, _, _ in segments[1:]:
    ax.axvline(dates[start], color="#ef4444", linestyle="--", linewidth=1.5)

ax.set_title("平均の変化点を確認")
ax.set_xlabel("日付")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/change_point.svg")
```

![plot](/images/timeseries/change_point.svg)

### 読み方のポイント

- 移動平均と縦線を重ねると、どこでレベルが切り替わっているかが視覚的に伝わる。
- 境界線の前後で平均や分散が大きく変わる場合は、モデルを分割するかダミー変数を追加する。
- 実務では自動検出アルゴリズム（BOCPD など）と組み合わせると再現性が高まる。
