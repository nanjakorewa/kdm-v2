---
title: "ウィンロスチャートで連勝・連敗を瞬時に把握"
pre: "6.7.12 "
weight: 12
title_suffix: "0ライン基準の棒で勝敗シーケンスを可視化"
---

営業やスポーツの勝敗履歴は、連勝・連敗の塊を強調すると流れがつかみやすくなります。ウィンロスチャートなら 0 ラインを挟んだ縦棒で勢いを表現できます。

```python
import matplotlib.pyplot as plt
import numpy as np

results = np.array([1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1])
streaks = np.cumsum(results)
colors = np.where(results > 0, "#22c55e", "#ef4444")

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.bar(
    np.arange(len(results)),
    height=np.where(results > 0, 1, -1),
    width=0.8,
    color=colors,
)
ax.axhline(0, color="#475569", linewidth=1)
ax.set_xticks(range(len(results)), labels=[f"{i+1}戦目" for i in range(len(results))], rotation=60)
ax.set_yticks([-1, 0, 1], labels=["負け", "", "勝ち"])
ax.set_ylim(-1.4, 1.4)
ax.set_title("直近16試合の勝敗シーケンス")
ax.set_ylabel("結果")

for idx, cumulative in enumerate(streaks):
    if (idx + 1) % 4 == 0:
        ax.text(idx, 1.15, f"{cumulative:+}", ha="center", fontsize=9)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/win-loss-chart.svg")
```

![win-loss chart](/images/visualize/advanced/win-loss-chart.svg)

### 読み方のポイント
- 緑が連続していれば連勝モード、赤が続くと連敗期。流れの転換点を説明しやすくなります。
- ストリーク数を補足で載せると、勢いの度合いを定量的に伝えられます。
- 結果が多くなると読みづらくなるので、四半期など一定期間ごとに区切るのがおすすめです。
