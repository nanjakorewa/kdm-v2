---
title: "ホライゾンチャートでシーズン変動を圧縮表示"
pre: "6.7.9 "
weight: 9
title_suffix: "積層した色帯で振れ幅とトレンドを同時に伝える"
---

季節変動が大きい系列を限られたスペースで見せたいときは、帯を折りたたむホライゾンチャートが役立ちます。振幅を色と濃さで表現するため、上下の変化が直感的に把握できます。

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 48)
baseline = 120 + 30 * np.sin(2 * np.pi * x / 12)
trend = 0.6 * x
rng = np.random.default_rng(7)
series = baseline + trend + rng.normal(0, 8, size=x.size)
centered = series - series.mean()
band = 20
levels = 3
palette_pos = ["#bae6fd", "#38bdf8", "#0ea5e9"]
palette_neg = ["#fecaca", "#f87171", "#ef4444"]

fig, ax = plt.subplots(figsize=(6.2, 3.6))
for level in range(levels):
    upper = np.clip(centered - level * band, 0, band)
    if np.any(upper > 0):
        ax.fill_between(
            x,
            level * band,
            level * band + upper,
            color=palette_pos[level],
            step="mid",
        )
    lower = np.clip(-centered - level * band, 0, band)
    if np.any(lower > 0):
        ax.fill_between(
            x,
            -(level * band + lower),
            -level * band,
            color=palette_neg[level],
            step="mid",
        )

ax.axhline(0, color="#475569", linewidth=1)
ax.set_xticks(range(0, 49, 6), labels=[f"{m}月" for m in range(1, 9)])
ax.set_yticks([])
ax.set_title("週次セッション数のホライゾンチャート（基準値からの偏差）")
ax.set_xlabel("週")
ax.spines[["top", "right", "left"]].set_visible(False)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/horizon-chart.svg")
```

![horizon chart](/images/visualize/advanced/horizon-chart.svg)

### 読み方のポイント
- 色の濃さが増えるほど偏差が大きい領域です。温度図のようにピーク時期を捉えられます。
- 0ラインより下はマイナス偏差。暖色で塗り分けることで減少期を強調できます。
- 複数系列を横に並べると、限られたスペースでも季節パターンの差を比較しやすくなります。
