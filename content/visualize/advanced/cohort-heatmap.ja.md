---
title: "コホートヒートマップで継続率を一望"
pre: "6.7.19 "
weight: 19
title_suffix: "獲得月 × 継続月を色で把握"
---

プロダクトの継続率分析では、獲得コホートと経過月を表にしたヒートマップが便利です。縦縞や横縞が出ていれば、特定期間の問題を推測できます。

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

cohorts = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
months = [f"{m}ヶ月目" for m in range(1, 7)]
rng = np.random.default_rng(21)
base = np.linspace(0.7, 0.4, num=6)
matrix = np.vstack(
    [
        np.clip(base - idx * 0.03 + rng.normal(0, 0.01, size=base.size), 0.1, 0.9)
        for idx in range(len(cohorts))
    ]
)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)

ax.set_xticks(range(len(months)), labels=months)
ax.set_yticks(range(len(cohorts)), labels=cohorts)
ax.set_title("サブスク継続率のコホートヒートマップ")

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        ax.text(j, i, f"{matrix[i, j]*100:.0f}%", ha="center", va="center", fontsize=9)

cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("継続率")
ax.set_xlabel("経過月")
ax.set_ylabel("獲得コホート")

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/cohort-heatmap.svg")
```

![cohort heatmap](/images/visualize/advanced/cohort-heatmap.svg)

### 読み方のポイント
- 縦方向の濃さが急に落ちているコホートがあれば、その獲得月に問題があったと推測できます。
- 横方向に共通した色変化は、プロダクトのライフサイクル全体の課題を示します。
- 値をパーセント表示にすると、色だけに頼らず具体的な継続率が把握できます。
