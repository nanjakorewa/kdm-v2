---
title: "自己相関のヒートマップ"
pre: "2.8.37 "
weight: 37
title_suffix: "複数ラグの相関をまとめて表示する"
---

{{< lead >}}
ラグごとの相関を行列にすると、どの遅れ同士が似た動きをするかを視覚的に把握できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(18)
dates = pd.date_range("2020-01-01", periods=250, freq="D")
values = []
prev = 0.0
for t in range(len(dates)):
    seasonal = 2.0 * np.sin(2 * np.pi * t / 30)
    prev = 0.5 * prev + seasonal + rng.normal(0, 1)
    values.append(prev)

series = pd.Series(values, index=dates)

max_lag = 20
lagged = pd.concat(
    {f"lag_{lag}": series.shift(lag) for lag in range(max_lag + 1)},
    axis=1,
).dropna()

corr_matrix = lagged.corr()

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    corr_matrix,
    cmap="RdBu_r",
    center=0,
    annot=False,
    ax=ax,
    cbar_kws={"label": "相関係数"},
)
ax.set_title("自己相関ヒートマップ（ラグ 0〜20）")
ax.set_xlabel("ラグ")
ax.set_ylabel("ラグ")

fig.tight_layout()
fig.savefig("static/images/timeseries/acf_heatmap.svg")
```

![plot](/images/timeseries/acf_heatmap.svg)

### 読み方のポイント

- 対角線より上を見れば、異なる遅れどうしの関係性を把握できる。
- 赤（正の相関）が続く領域は周期的なリズムの存在を、青（負の相関）は逆位相を示唆する。
- ラグ 0 と特定ラグの相関が高ければ、そのラグを特徴量として加える価値がある。

