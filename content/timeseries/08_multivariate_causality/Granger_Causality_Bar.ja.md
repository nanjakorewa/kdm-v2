---
title: "グレンジャー因果性の検定結果を棒グラフで"
pre: "2.8.36 "
weight: 36
title_suffix: "p値を可視化して比較する"
---

{{< lead >}}
グレンジャー因果性検定は、ある系列の過去が別の系列の予測に寄与するかを調べる手法です。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

rng = np.random.default_rng(29)
dates = pd.date_range("2020-01-01", periods=240, freq="D")
driver = np.sin(2 * np.pi * np.arange(len(dates)) / 20) + rng.normal(0, 0.8, len(dates))
series_x = pd.Series(driver, index=dates)
series_y = pd.Series(0.6 * driver + rng.normal(0, 1, len(dates)), index=dates)

test_data = pd.DataFrame({"y": series_y, "x": series_x})
maxlag = 6
results = grangercausalitytests(test_data[["y", "x"]], maxlag=maxlag, verbose=False)
p_values = [results[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)]

fig, ax = plt.subplots(figsize=(6.5, 3.5))
ax.bar(range(1, maxlag + 1), p_values, color="#2563eb", alpha=0.8)
ax.axhline(0.05, color="#ef4444", linestyle="--", linewidth=1.2, label="有意水準 0.05")
ax.set_title("グレンジャー因果性検定の p 値（x → y）")
ax.set_xlabel("ラグ数")
ax.set_ylabel("p 値")
ax.set_xticks(range(1, maxlag + 1))
ax.legend()
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("static/images/timeseries/granger_bar.svg")
```

![plot](/images/timeseries/granger_bar.svg)

### 読み方のポイント

- p 値が有意水準より低ければ、ラグがその方向の予測に役立っていると判断できる。
- ラグが大きくなるほど自由度が下がり p 値が上がることもあるので、解釈は上限ラグを決めて行う。
- 双方向の因果性がないかを確認するため、逆方向（y → x）の検定もセットで実施する。

