---
title: "ロリポップチャートでシンプルに比較"
pre: "6.3.7 "
weight: 7
title_suffix: "棒グラフより軽量な表現"
---

棒の代わりに線と点で表現するロリポップチャートは、項目が多いときでも軽やかに比較できます。

```python
import numpy as np
import matplotlib.pyplot as plt

metrics = ["UX", "性能", "機能性", "信頼性", "コスパ", "サポート"]
score = np.array([4.6, 4.1, 4.4, 4.2, 3.9, 4.3])

fig, ax = plt.subplots(figsize=(6, 4))
ax.hlines(y=metrics, xmin=0, xmax=score, color="#94a3b8", linewidth=2)
ax.plot(score, metrics, "o", color="#1d4ed8", markersize=10)

ax.set_xlabel("満足度（5 点満点）")
ax.set_xlim(0, 5)
ax.set_title("機能別満足度のロリポップチャート")
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/bar/lollipop.svg")
```

![lollipop](/images/visualize/bar/lollipop.svg)

### 読み方のポイント

- 点の色や形でカテゴリーを追加で表現できる。
- 長さに集中してほしい場合は背景をシンプルに保つ。
- 値を表示するときは点の右側に小さく添えると視線の流れを邪魔しない。
