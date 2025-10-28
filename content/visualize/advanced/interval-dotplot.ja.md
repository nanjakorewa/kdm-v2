---
title: "インターバルドットプロットで推定値と不確実性を管理"
pre: "6.7.21 "
weight: 21
title_suffix: "平均値と信頼区間をコンパクトに"
---

ABテストや回帰分析の推定結果は、点推定だけでなく信頼区間も合わせて示すと説得力が増します。水平線と点で構成するインターバルドットプロットは読みやすいフォーマットです。

```python
import numpy as np
import matplotlib.pyplot as plt

segments = ["無料会員", "ライトプラン", "スタンダード", "プレミアム"]
effect = np.array([0.12, 0.18, 0.27, 0.35])
low = effect - np.array([0.05, 0.06, 0.07, 0.08])
high = effect + np.array([0.05, 0.06, 0.07, 0.09])

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.hlines(range(len(segments)), low, high, color="#94a3b8", linewidth=3)
ax.scatter(effect, range(len(segments)), color="#0ea5e9", s=90, zorder=3)

ax.axvline(0, color="#475569", linestyle="--", linewidth=1)
ax.set_yticks(range(len(segments)), labels=segments)
ax.set_xlabel("改善率（対照比）")
ax.set_title("施策効果の推定値と90%信頼区間")
ax.set_xlim(-0.05, 0.45)
ax.grid(axis="x", alpha=0.2)

for idx, (eff, lo, hi) in enumerate(zip(effect, low, high)):
    ax.text(hi + 0.01, idx, f"{eff*100:.1f}% (+{(hi - eff)*100:.1f}/-{(eff - lo)*100:.1f})", va="center")

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/interval-dotplot.svg")
```

![interval dotplot](/images/visualize/advanced/interval-dotplot.svg)

### 読み方のポイント
- 横線の長さが不確実性を表します。線が短いほど推定が安定しています。
- 0 を跨ぐかどうかを基準線で確認すれば、効果の有無を即座に判断できます。
- 点を大小で強調すると、重み付けやデータ量も一緒に伝えることができます。
