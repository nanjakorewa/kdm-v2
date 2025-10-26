---
title: "グループ化棒グラフ"
pre: "6.3.3 "
weight: 3
title_suffix: "複数カテゴリを並列で比較する"
---

月別の売上をチャネル別に並べ、増減を比較する例です。幅を調整してグループ間に余白を持たせます。

```python
import numpy as np
import matplotlib.pyplot as plt

months = ["Jan", "Feb", "Mar", "Apr"]
online = np.array([120, 140, 155, 170])
store = np.array([90, 105, 110, 120])
width = 0.35
x = np.arange(len(months))

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - width / 2, online, width, label="オンライン", color="#3b82f6")
ax.bar(x + width / 2, store, width, label="店舗", color="#f59e0b")

ax.set_xticks(x, months)
ax.set_ylabel("売上（百万円）")
ax.set_title("チャネル別売上の推移")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/bar/grouped_bar.svg")
```

![grouped bar](/images/visualize/bar/grouped_bar.svg)

### 読み方のポイント

- 凡例の順序は棒の描画順と合わせる。
- グリッドや注釈で差分を補助すれば、数値に注目してもらいやすい。
- 棒が増えすぎると読みにくくなるため、チャネル数は3～4程度に抑えるのが無難。
