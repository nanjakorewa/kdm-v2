---
title: "クアドラントチャートで優先度を分類"
pre: "6.7.13 "
weight: 13
title_suffix: "重要度×実行容易度で施策ポートフォリオ整理"
---

重要度と実行容易度など、指標を2軸で評価する際はクアドラントチャートが有効です。背景を象限ごとに色分けするだけで、優先度の分類が直感的に分かります。

```python
import matplotlib.pyplot as plt
import numpy as np

projects = ["A", "B", "C", "D", "E", "F", "G"]
impact = np.array([8.5, 5.5, 7.2, 3.8, 6.1, 8.9, 4.4])
effort = np.array([3.0, 6.2, 4.1, 5.5, 2.5, 7.0, 3.8])
colors = ["#22c55e", "#f97316", "#38bdf8", "#a855f7", "#14b8a6", "#facc15", "#ef4444"]

fig, ax = plt.subplots(figsize=(6, 6))
ax.axvline(5, color="#475569", linewidth=1.2)
ax.axhline(6, color="#475569", linewidth=1.2)

ax.fill_between([0, 5], 6, 10, color="#bbf7d0", alpha=0.5)
ax.fill_between([5, 10], 6, 10, color="#dbeafe", alpha=0.6)
ax.fill_between([0, 5], 0, 6, color="#fee2e2", alpha=0.6)
ax.fill_between([5, 10], 0, 6, color="#fde68a", alpha=0.6)

ax.scatter(effort, impact, s=110, color=colors, edgecolor="white", linewidth=1)
for label, x_val, y_val in zip(projects, effort, impact):
    ax.text(x_val, y_val + 0.25, label, ha="center", fontsize=10, weight="bold")

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_xlabel("実行容易度（高いほど簡単）")
ax.set_ylabel("ビジネスインパクト（高いほど大きい）")
ax.set_title("プロジェクト優先度のクアドラント分類")
ax.grid(alpha=0.2, linestyle="--")

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/quadrant-chart.svg")
```

![quadrant chart](/images/visualize/advanced/quadrant-chart.svg)

### 読み方のポイント
- 右上象限は高インパクト・高容易度。すぐ着手すべき案件として強調できます。
- 左下象限は負荷が高く効果も小さいため、保留や中止の判断に役立ちます。
- 分割線を中央値に調整すると、データに合わせたバランスの良い分類が可能です。
