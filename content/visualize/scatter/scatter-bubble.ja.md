---
title: "バブルチャートで第三の変数を表現"
pre: "6.5.2 "
weight: 2
title_suffix: "サイズで重要度や数量を追加する"
---

売上（x 軸）、利益率（y 軸）、顧客数（バブルサイズ）の 3 変数を同時に表示する例です。

```python
import numpy as np
import matplotlib.pyplot as plt

segments = ["Enterprise", "SMB", "Startup", "Consumer", "Partner"]
revenue = np.array([320, 180, 90, 70, 150])
margin = np.array([28, 22, 15, 12, 20])
customers = np.array([45, 110, 260, 520, 160])

fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(
    revenue,
    margin,
    s=customers,
    c=margin,
    cmap="Blues",
    alpha=0.7,
    edgecolors="#1f2937",
)

for x, y, label in zip(revenue, margin, segments):
    ax.text(x, y, label, fontsize=10, ha="center", va="center")

ax.set_xlabel("売上（百万円）")
ax.set_ylabel("利益率（%）")
ax.set_title("セグメント別 売上・利益率・顧客数")
ax.grid(alpha=0.3)

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("利益率（%）")

fig.tight_layout()
fig.savefig("static/images/visualize/scatter/bubble.svg")
```

![bubble](/images/visualize/scatter/bubble.svg)

### 読み方のポイント

- バブルサイズは面積が比例するようにデータをスケーリングする。
- オーバーラップが多い場合はズラす・透明度を下げるなどの工夫が必要。
- カテゴリ名を中央に置くと凡例を省略でき、一覧性が高まる。
