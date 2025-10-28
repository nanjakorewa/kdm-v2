---
title: "ガントチャートで短期プロジェクトを整理"
pre: "6.7.11 "
weight: 11
title_suffix: "broken_barh で期間タスクを一枚に"
---

複数チームのタスク期間を俯瞰するにはガントチャートが定番です。`matplotlib.axes.Axes.broken_barh` を使えば簡単に横棒で期間を描画できます。

```python
import matplotlib.pyplot as plt

teams = ["企画", "開発", "QA", "CS"]
timeline = [
    [(1, 3), (5, 2)],
    [(2, 5), (8, 3)],
    [(4, 3), (8, 2)],
    [(3, 2), (7, 4)],
]
colors = ["#38bdf8", "#818cf8", "#f472b6", "#facc15"]

fig, ax = plt.subplots(figsize=(6.2, 3.8))
for idx, (team, segments) in enumerate(zip(teams, timeline)):
    for seg, color in zip(segments, colors):
        ax.broken_barh([seg], (idx - 0.35, 0.7), facecolors=color, alpha=0.85)

ax.set_ylim(-1, len(teams))
ax.set_xlim(0, 12)
ax.set_yticks(range(len(teams)), labels=teams)
ax.set_xticks(range(0, 13))
ax.set_xlabel("週")
ax.set_title("四半期リリース準備のガントチャート")
ax.grid(axis="x", alpha=0.2, linestyle="--", linewidth=0.8)
ax.set_axisbelow(True)

milestones = {"仕様確定": 3, "テスト完了": 9}
for label, week in milestones.items():
    ax.axvline(week, color="#475569", linestyle=":", linewidth=1.2)
    ax.text(week + 0.1, len(teams) - 0.4, label, rotation=90, va="top", fontsize=9)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/gantt-timeline.svg")
```

![gantt chart](/images/visualize/advanced/gantt-timeline.svg)

### 読み方のポイント
- 横棒の長さでタスク期間、縦位置で担当チームが分かります。重なりはリソース衝突のサインです。
- 節目週を縦線で入れておくと、マイルストーンと進捗差が確認しやすくなります。
- 週単位より粒度を細かくしたい場合は、x 軸目盛を日付表示にしても有効です。
