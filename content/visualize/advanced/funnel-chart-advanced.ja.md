---
title: "ファネルチャートで段階ごとの離脱を把握"
pre: "6.7.22 "
weight: 22
title_suffix: "ステップごとの減衰を梯形で表示"
---

流入→CVまでの段階ごとの減衰を示すにはファネルチャートが便利です。梯形を使うと帯の幅で残存数の減り具合が直観的に伝わります。

```python
import numpy as np
import matplotlib.pyplot as plt

steps = ["訪問", "商品閲覧", "カート投入", "決済情報", "購入完了"]
counts = np.array([12000, 5400, 2600, 1800, 1250])
max_width = counts[0]

fig, ax = plt.subplots(figsize=(6, 4))
y_positions = np.arange(len(steps), 0, -1)
for idx, (step, count) in enumerate(zip(steps, counts)):
    width = count / max_width
    left = 0.5 - width / 2
    ax.fill_between(
        [left, left + width],
        [y_positions[idx]] * 2,
        [y_positions[idx] - 0.8] * 2,
        color=plt.cm.Blues(0.3 + idx * 0.12),
    )
    ax.text(
        0.5,
        y_positions[idx] - 0.4,
        f"{step}\n{count:,}",
        ha="center",
        va="center",
        color="white",
        fontsize=11,
    )

ax.set_xlim(0, 1)
ax.set_ylim(0, len(steps) + 0.5)
ax.axis("off")
ax.set_title("EC購入ファネルの離脱状況")

conversion = counts[-1] / counts[0]
ax.text(0.02, 0.3, f"CVR: {conversion:.1%}", fontsize=11, fontweight="bold")

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/funnel-chart-advanced.svg")
```

![funnel chart](/images/visualize/advanced/funnel-chart-advanced.svg)

### 読み方のポイント
- 各ステップの幅が残存ユーザー数を示します。ギャップが大きい段階がボトルネックです。
- トップとボトムの値からCVRを算出し、図内に表示するとインパクトが伝わります。
- サービスによってステップ名を変えるだけで応用できます。
