---
title: "移動平均でノイズをならす"
pre: "6.4.3 "
weight: 3
title_suffix: "短期変動とトレンドを同時に表示"
---

日次データに移動平均を重ねるとノイズを抑えつつトレンドを把握できます。`pandas.Series.rolling` を使うと簡単に計算できます。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = pd.date_range("2024-01-01", periods=60, freq="D")
sales = pd.Series(np.random.normal(loc=200, scale=25, size=len(rng))).cumsum() + 500
rolling = sales.rolling(window=7, center=True).mean()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(rng, sales, label="日次売上", color="#9ca3af", linewidth=1.5, alpha=0.7)
ax.plot(rng, rolling, label="7 日移動平均", color="#2563eb", linewidth=2.5)

ax.set_ylabel("売上（万円）")
ax.set_title("移動平均で売上トレンドを把握する")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/visualize/line/rolling_average.svg")
```

![rolling average](/images/visualize/line/rolling_average.svg)

### 読み方のポイント

- 移動平均の窓幅は週単位・月単位など分析目的に合わせて設定する。
- 原系列も同時に描くと、季節性や突発的な変化を見逃しにくい。
- 時系列の欠損がある場合は前処理で補完しておくと滑らかな曲線になる。
