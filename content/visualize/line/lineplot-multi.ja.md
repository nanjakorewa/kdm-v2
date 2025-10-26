---
title: "複数ラインの推移比較"
pre: "6.3.1 "
weight: 1
title_suffix: "2つの系列を同じ軸で比較する"
---

{{% notice tip %}}
図版は `python scripts/generate_visualize_assets.py` を実行すると
`static/images/visualize/line/lineplot_multi.svg` に再生成されます。
{{% /notice %}}

週ごとの売上推移を、東西2拠点で比較する折れ線グラフです。`matplotlib` の素直な
API だけで見やすいグラフが作れます。

```python
import matplotlib.pyplot as plt

weeks = ["W1", "W2", "W3", "W4", "W5"]
east = [180, 150, 170, 140, 160]
west = [120, 130, 140, 125, 135]

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(weeks, east, marker="o", linewidth=2.5, color="#2563eb", label="東エリア")
ax.plot(weeks, west, marker="o", linewidth=2.5, color="#10b981", label="西エリア")

ax.set_ylabel("売上（万円）")
ax.set_title("週別 売上推移")
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False, loc="lower right")

fig.tight_layout()
fig.savefig("static/images/visualize/line/lineplot_multi.svg")
```

![lineplot](/images/visualize/line/lineplot_multi.svg)

### 読み方のポイント

- 2系列を同じスケールで描くことで、上下動のタイミングの違いが一目で分かる。
- 指標の単位を必ず縦軸に明記し、凡例はプロットと同じ色で揃える。
- 重要な区間に縦の補助線や注釈を加えるとさらに把握しやすくなる。
