---
title: "サークルマトリクスで指標推移を一望する"
pre: "7.2.2 "
weight: 2
searchtitle: "EPS と売上の前期比をサークルで比較"
---

四半期ごとに変化する EPS や売上の伸び率は、表形式よりも色と配置で見せるとトレンドがつかみやすくなります。ここでは銘柄 × 指標 × 四半期の 3 軸をサークルマトリクスで描き、どこで成長が加速・減速したのかを直感的に把握できるグラフを作ります。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

plt.style.use("scripts/k_dm.mplstyle")

data = pd.DataFrame(
    {
        "company": ["AAA"] * 8 + ["BBB"] * 8 + ["CCC"] * 8,
        "quarter": list(range(1, 9)) * 3,
        "eps_growth": [
            0.12, 0.18, 0.05, -0.08, 0.03, 0.11, 0.16, 0.21,
            0.04, -0.12, 0.08, 0.15, 0.02, 0.05, 0.09, 0.12,
            0.20, 0.24, 0.18, 0.12, 0.08, 0.05, 0.04, 0.06,
        ],
        "revenue_growth": [
            -0.05, 0.08, 0.12, 0.20, 0.06, 0.09, 0.14, 0.19,
            0.02, -0.06, 0.04, 0.11, 0.05, 0.07, -0.03, 0.02,
            0.18, 0.14, 0.10, 0.07, 0.12, 0.16, 0.09, 0.05,
        ],
    }
)

norm = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=0.25)
companies = data["company"].unique()
metrics = [("eps_growth", "EPS"), ("revenue_growth", "Revenue")]

fig, ax = plt.subplots(figsize=(10, 5.5))

for idx, company in enumerate(companies):
    subset = data[data["company"] == company]
    for offset, (metric, label) in enumerate(metrics):
        y_position = idx + (offset - 0.5) * 0.3
        sc = ax.scatter(
            subset["quarter"],
            np.full(len(subset), y_position),
            c=subset[metric],
            cmap="RdYlGn",
            s=420,
            norm=norm,
            edgecolor="#1f2937",
            linewidth=0.5,
        )
        for q, val in zip(subset["quarter"], subset[metric]):
            ax.text(
                q + 0.12,
                y_position,
                f"{val:+.1%}",
                fontsize=10,
                va="center",
                ha="left",
            )
        ax.text(
            0.3,
            y_position,
            label,
            fontsize=11,
            color="#475569",
            va="center",
        )

ax.set_yticks(range(len(companies)))
ax.set_yticklabels(companies, fontsize=12)
ax.set_xticks(range(1, 9))
ax.set_xticklabels([f"2023Q{q}" for q in range(1, 9)], rotation=30, ha="right")
ax.set_xlim(0.4, 8.6)
ax.set_title("四半期ごとの EPS / 売上 前期比", fontsize=14, pad=12)
ax.set_xlabel("四半期")
ax.grid(axis="x", color="#cbd5f5", alpha=0.4)

cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
cbar.ax.set_ylabel("前期比", rotation=90)

output = Path("static/images/finance/visualize/eps_revenue_circle.svg")
output.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(output)
```

![EPS・売上のサークルマトリクス](/images/finance/visualize/eps_revenue_circle.svg)

### 読み方のポイント
- 行が銘柄、列が四半期、円の色が伸び率、円のラベルが具体的なパーセンテージです。緑が濃いほどプラス成長、赤が濃いほどマイナス成長を示します。
- EPS と売上を上下に 2 レイヤーで並べているため、利益率だけが伸びているのか、売上も伸びているのかがひと目で分かります。
- 四半期を通じてマイナスが続いている銘柄は早期に検知できるため、決算シーズンの速報ダッシュボードとしても便利です。
