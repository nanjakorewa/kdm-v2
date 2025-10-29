---
title: "イールドカーブのシフトを比較する"
pre: "7.2.6 "
weight: 6
title_suffix: "複数時点の利回り曲線を重ねて政策効果を把握"
---

利回り曲線は金融政策や景気見通しを反映します。複数の時点を重ねて描くと、どの年限がどの程度シフトしたかがよく分かります。サンプルデータを使って、政策発表前後のカーブを比較してみましょう。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("scripts/k_dm.mplstyle")
tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 30])

snapshots = pd.DataFrame(
    {
        "2023-06": [0.012, 0.013, 0.015, 0.018, 0.021, 0.025, 0.026, 0.027, 0.030],
        "2023-09": [0.014, 0.016, 0.018, 0.021, 0.024, 0.027, 0.029, 0.031, 0.033],
        "2023-12": [0.016, 0.018, 0.020, 0.024, 0.027, 0.030, 0.032, 0.034, 0.036],
    },
    index=tenors,
)

fig, ax = plt.subplots(figsize=(9, 4.5))
colors = ["#0ea5e9", "#6366f1", "#f97316"]

for (label, values), color in zip(snapshots.items(), colors):
    ax.plot(
        snapshots.index,
        np.array(values) * 100,
        marker="o",
        linewidth=2,
        label=label,
        color=color,
    )

ax.set_xscale("log")
ax.set_xticks(tenors)
ax.set_xticklabels(["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "30Y"])
ax.set_ylabel("利回り（%）")
ax.set_title("主要金利のカーブシフト比較（サンプルデータ）")
ax.grid(True, which="both", linestyle="--", alpha=0.3)
ax.legend()

output = Path("static/images/finance/visualize/yield_curve_shift.svg")
output.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(output)
```

![イールドカーブの推移比較](/images/finance/visualize/yield_curve_shift.svg)

### 分析のヒント
- 短期金利が大きく上昇し、長期が横ばいの場合は「ベアフラットニング」、逆に長期が上がるなら「ベアスティープニング」と呼ばれます。カーブの形で金融市場のセンチメントを把握できます。
- 期間を対数スケールにしておくと 30 年債までの広いレンジでも視認性が保てます。
- 実際のデータを使う場合は、各年限のリターンを計算してスプレッド（10 年-2 年など）を別グラフで描くと景気後退サインの検知に応用できます。
