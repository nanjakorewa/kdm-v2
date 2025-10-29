---
title: "カントリーリスクプレミアムを可視化する"
pre: "7.2.1 "
weight: 1
searchtitle: "国別リスクプレミアムを棒グラフで比較"
---

国債利回りや CDS スプレッドから推計した「カントリーリスクプレミアム」は、同じ地域でも大きな差があります。ここではサンプルデータを使って、国別のリスクプレミアムを棒グラフとハイライトで比較します。

```python
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("scripts/k_dm.mplstyle")

data = pd.DataFrame(
    {
        "country": ["Japan", "United States", "Brazil", "India", "South Africa", "Turkey", "Mexico"],
        "risk_premium": [0.012, 0.018, 0.038, 0.031, 0.042, 0.055, 0.029],
    }
)

data = data.sort_values("risk_premium")
fig, ax = plt.subplots(figsize=(8, 4.8))
bars = ax.bar(data["country"], data["risk_premium"] * 100, color="#3b82f6")

average = data["risk_premium"].mean() * 100
ax.axhline(average, color="#ef4444", linestyle="--", linewidth=1.2, label=f"平均 {average:.1f}%")

for bar, value in zip(bars, data["risk_premium"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value * 100 + 0.6,
        f"{value*100:.1f}%",
        ha="center",
        fontsize=10,
    )

ax.set_ylabel("リスクプレミアム（%）")
ax.set_title("主要国のリスクプレミアム比較（サンプルデータ）")
ax.set_ylim(0, 6.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper left")

output = Path("static/images/finance/visualize/country_risk_premium.svg")
output.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(output)
```

![主要国のリスクプレミアム比較](/images/finance/visualize/country_risk_premium.svg)

### 活用のヒント
- 同じ地域でもリスクプレミアムの差が大きい場合、加重平均や地域インデックスを算出するときに調整が必要です。
- 最新データは NYU Stern の [Country Default Spreads and Risk Premiums](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ctryprem.html) などで定期的に更新されます。自動取得スクリプトを組み合わせると、月次レポートの作成が効率化できます。
- プロジェクト評価（資本コスト）に使う場合は、社債スプレッドや為替ボラティリティと合わせて複数指標の比較チャートを作ると説得力が増します。
