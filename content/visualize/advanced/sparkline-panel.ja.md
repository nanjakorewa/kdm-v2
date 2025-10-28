---
title: "スパークラインパネルで複数指標をミニ表示"
pre: "6.7.20 "
weight: 20
title_suffix: "小さな折れ線を並べて全体感を把握"
---

ダッシュボードで主要指標の傾向だけざっと掴みたいとき、ミニ折れ線を並べるスパークラインパネルが役立ちます。数値ラベルを添えて、細部はコンパクトにまとめます。

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(9)
metrics = ["UU", "PV", "CVR", "注文数", "解約率", "NPS"]
series = []
for base in [40, 120, 3.2, 60, 1.8, 25]:
    trend = base + np.linspace(-3, 5, 12)
    noise = rng.normal(0, base * 0.05, size=12)
    series.append(trend + noise)

fig, axes = plt.subplots(2, 3, figsize=(6.4, 4.2), sharex=True)
for ax, metric, values in zip(axes.flat, metrics, series):
    ax.plot(values, color="#0ea5e9", linewidth=1.8)
    ax.fill_between(range(len(values)), values, alpha=0.15, color="#bae6fd")
    ax.set_title(metric, fontsize=11, pad=6)
    ax.set_xticks([0, 5, 11], labels=["1月", "6月", "12月"], fontsize=8)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    latest = values[-1]
    ax.text(len(values) - 0.4, latest, f"{latest:.1f}", ha="right", va="bottom", fontsize=9)

for ax in axes[-1]:
    ax.set_xlabel("月")

fig.suptitle("主要指標のスパークラインパネル", fontsize=14, y=0.98)
fig.tight_layout()
fig.savefig("static/images/visualize/advanced/sparkline-panel.svg")
```

![sparkline panel](/images/visualize/advanced/sparkline-panel.svg)

### 読み方のポイント
- 小さな折れ線を並べることで、一覧性とタイムリーな変化を両立できます。
- 目盛りを省略しつつ最新値だけ表示すると、視線が最新の状態に集中します。
- 詳細分析は個別ページで行い、トップ画面ではスパークラインで全体の空気感を共有するのがおすすめです。
