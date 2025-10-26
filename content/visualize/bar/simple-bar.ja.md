---
title: "基本の縦型棒グラフ"
pre: "6.3.1 "
weight: 1
title_suffix: "カテゴリ別の値を比較する"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` を実行すると
`static/images/visualize/bar/simple_bar.svg` が生成されます。
{{% /notice %}}

カテゴリごとの売上額を比較する単純な棒グラフです。値が少ない場合はラベルを直接付けると読みやすくなります。

```python
import matplotlib.pyplot as plt

stores = ["東京", "名古屋", "大阪", "福岡", "札幌"]
sales = [320, 210, 280, 190, 160]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(stores, sales, color="#2563eb")

ax.set_ylabel("売上（百万円）")
ax.set_title("主要拠点の売上比較")
ax.bar_label(bars, fmt="%.0f", padding=4)
ax.set_ylim(0, 360)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/bar/simple_bar.svg")
```

![bar](/images/visualize/bar/simple_bar.svg)

### 読み方のポイント

- 目盛りは切りの良い値を使い、0 から始める。
- 棒の順序は大きい順または意味のある並びにすると比較しやすい。
- 値が数個ならラベル表示、数十個ならソートや絞り込みを検討する。
