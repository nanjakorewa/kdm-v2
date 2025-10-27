---
title: "パワースペクトルで周期を検出"
pre: "2.8.23 "
weight: 23
title_suffix: "FFTで支配的な周波数を確認する"
---

{{< lead >}}
フーリエ変換でパワースペクトルを求めると、どの周期成分が強いかを視覚的に把握できます。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(8)
dates = pd.date_range("2021-01-01", periods=365, freq="D")
signal = (
    50
    + 4 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)  # 週次成分
    + 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)  # 月次成分
)
noise = rng.normal(0, 1.5, size=len(dates))
series = pd.Series(signal + noise, index=dates)

fft_values = np.fft.rfft(series - series.mean())
power = np.abs(fft_values) ** 2
freq = np.fft.rfftfreq(len(series), d=1)  # 1日を単位とする

nonzero = freq > 0
period = 1 / freq[nonzero]

fig, ax = plt.subplots(figsize=(7, 4))
ax.stem(period, power[nonzero], basefmt=" ", use_line_collection=True)
ax.set_xlim(1, 60)
ax.set_title("パワースペクトルの例（周期表示）")
ax.set_xlabel("周期（日）")
ax.set_ylabel("パワー")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/power_spectrum.svg")
```

![plot](/images/timeseries/power_spectrum.svg)

### 読み方のポイント

- 棒が高い周期ほどそのリズムが強く現れていることを意味する。例では7日と30日付近が支配的。
- 週次や月次など事前に期待している周期が本当に存在するかを確かめる診断として有用。
- パワースペクトルで主要周期が分かったら、季節分解や差分の周期を決める手がかりになる。

