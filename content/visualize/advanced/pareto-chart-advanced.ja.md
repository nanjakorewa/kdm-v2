---
title: "パレートチャートで主要要因を素早く特定"
pre: "6.7.17 "
weight: 17
title_suffix: "累積比率と棒を重ねてABC分析"
---

不良要因や問合せカテゴリなど、累積寄与の大きさを伝えるならパレートチャートが王道です。棒グラフと累積折れ線を組み合わせ、80/20 ルールの分岐点を明確にします。

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

categories = ["設定ミス", "操作不明", "バグ", "仕様質問", "連携エラー", "その他"]
counts = np.array([120, 95, 70, 45, 30, 18])
sorted_idx = np.argsort(counts)[::-1]
counts = counts[sorted_idx]
categories = [categories[i] for i in sorted_idx]

cumulative = counts.cumsum() / counts.sum()

fig, ax1 = plt.subplots(figsize=(6.4, 4))
ax1.bar(categories, counts, color="#38bdf8")
ax1.set_ylabel("件数")
ax1.set_title("問い合わせカテゴリのパレート分析")
ax1.grid(axis="y", alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(categories, cumulative, color="#ef4444", marker="o")
ax2.set_ylabel("累積比率")
ax2.set_ylim(0, 1.05)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))

threshold = np.argmax(cumulative >= 0.8)
ax2.axhline(0.8, color="#475569", linestyle="--", linewidth=1)
ax1.axvline(threshold + 0.5, color="#475569", linestyle=":", linewidth=1)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/pareto-chart-advanced.svg")
```

![pareto chart](/images/visualize/advanced/pareto-chart-advanced.svg)

### 読み方のポイント
- 棒の高さで個別件数、折れ線で累積寄与率を合わせて確認できます。
- 80%ラインを引くと、重点的に対策すべきカテゴリが可視化されます。
- 累積線が緩やかに伸びる場合は、要因が分散しているため横断的な改善が必要だと判断できます。
