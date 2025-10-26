---
title: "スパークラインでコンパクトに推移を表示"
pre: "6.4.8 "
weight: 8
title_suffix: "ダッシュボードや表内で活躍するミニチャート"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` を実行すると
`static/images/visualize/line/sparkline.svg` が更新されます。
{{% /notice %}}

スパークラインは目盛りや軸を取り除いた小さな折れ線グラフで、一覧表などに組み込むとトレンドを素早く確認できます。

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(loc=100, scale=8, size=24).cumsum()
fig, ax = plt.subplots(figsize=(3.5, 0.8))
ax.plot(data, color="#22c55e", linewidth=1.5)
ax.fill_between(range(len(data)), data, np.min(data), color="#bbf7d0", alpha=0.6)

ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

fig.tight_layout(pad=0.2)
fig.savefig("static/images/visualize/line/sparkline.svg")
```

![sparkline](/images/visualize/line/sparkline.svg)

### 読み方のポイント

- 軸や目盛りを省く代わりに、直近値や最大値などを隣に数値で表示すると誤解が少ない。
- 重要なポイントだけマーカーで強調するとメリハリが付く。
- 大量のチャートを並べる際は配色を統一し、背景を白にして視認性を高める。
