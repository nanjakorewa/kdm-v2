---
title: "積み上げ棒グラフで構成比を示す"
pre: "6.3.4 "
weight: 4
title_suffix: "チャネル別の内訳を１本にまとめる"
---

月別総売上をチャネル別に積み上げると、全体量と構成比を同時に見ることができます。

```python
import numpy as np
import matplotlib.pyplot as plt

months = ["Apr", "May", "Jun", "Jul"]
online = np.array([150, 180, 190, 210])
store = np.array([100, 120, 130, 150])
wholesale = np.array([60, 70, 80, 85])

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(months, online, label="オンライン", color="#60a5fa")
ax.bar(months, store, bottom=online, label="店舗", color="#fbbf24")
ax.bar(months, wholesale, bottom=online + store, label="卸", color="#34d399")

ax.set_ylabel("売上（百万円）")
ax.set_title("販路別売上の構成比")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/bar/stacked_bar.svg")
```

![stacked bar](/images/visualize/bar/stacked_bar.svg)

### 読み方のポイント

- パーセント表示にしたい場合は、各系列を合計で割った 100% スタックにすると良い。
- 月ごとの合計値に注目してもらいたい場合は、合計線を上に重ねるのも一案。
- 下から順に主要チャネルを並べると凡例との対応が分かりやすい。
