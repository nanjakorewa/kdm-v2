---
title: "ストリームグラフで構成比のうねりを見る"
pre: "6.7.10 "
weight: 10
title_suffix: "baseline=\"wiggle\"で柔らかな積層曲線を描く"
---

カテゴリ構成が時間とともにどのようにシフトしているかを強調したい場合は、ストリームグラフが便利です。上下対称の波で割合の変化が視覚的になります。

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 24)
rng = np.random.default_rng(24)
segments = []
for phase in range(4):
    trend = 12 + 6 * np.sin(2 * np.pi * (x + phase * 3) / 12)
    noise = rng.normal(0, 1.2, size=x.size)
    segments.append(np.clip(trend + noise, 0, None))
stacked = np.vstack(segments)

fig, ax = plt.subplots(figsize=(6.4, 4))
colors = ["#0ea5e9", "#22d3ee", "#818cf8", "#f472b6"]
ax.stackplot(x, stacked, baseline="wiggle", colors=colors, alpha=0.9)
ax.set_xticks(range(0, 24, 3), labels=[f"{h}時" for h in range(0, 24, 3)])
ax.set_title("チャネル別流入構成のストリームグラフ")
ax.set_ylabel("セッション数")
ax.set_xlabel("時間帯")
ax.spines[["right", "top"]].set_visible(False)
ax.grid(axis="x", alpha=0.15)

legend_labels = ["自然検索", "広告", "SNS", "紹介"]
ax.legend(legend_labels, loc="upper left", frameon=False)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/streamgraph.svg")
```

![streamgraph](/images/visualize/advanced/streamgraph.svg)

### 読み方のポイント
- 波の厚みで構成比の増減が分かります。昼帯でSNSが厚くなっていれば拡散タイミングの示唆になります。
- 中央線が水平なら全体規模は安定、上下に大きく振れていれば総量が変動しているサインです。
- 連続データが少ないと波がガタつくので、平滑化した系列を使うと読みやすくなります。
