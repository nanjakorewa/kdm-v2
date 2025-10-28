---
title: "バーコードタイムラインで頻度を瞬時に把握"
pre: "6.7.16 "
weight: 16
title_suffix: "縦線だけでイベント密度を表す"
---

イベントが発生した日付だけが重要な場合、縦線の並びで密度を見せるバーコードタイムラインが便利です。期間の偏りや集中がひと目で分かります。

```python
import pandas as pd
import matplotlib.pyplot as plt

dates = pd.to_datetime(
    [
        "2024-01-05",
        "2024-01-08",
        "2024-01-12",
        "2024-01-20",
        "2024-02-02",
        "2024-02-07",
        "2024-02-08",
        "2024-02-17",
        "2024-03-01",
        "2024-03-09",
        "2024-03-10",
        "2024-03-24",
        "2024-04-02",
        "2024-04-18",
        "2024-05-01",
    ]
)

fig, ax = plt.subplots(figsize=(6.4, 1.8))
ax.vlines(dates, ymin=0, ymax=1, color="#0f172a", linewidth=2)
ax.set_ylim(0, 1)
ax.set_yticks([])
ax.set_title("重要アラート発生日のバーコードタイムライン")
ax.set_xlabel("日付")
ax.set_xlim(dates.min() - pd.Timedelta(days=3), dates.max() + pd.Timedelta(days=3))

ax.tick_params(axis="x", rotation=45)
ax.spines[["left", "top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/barcode-timeline.svg")
```

![barcode timeline](/images/visualize/advanced/barcode-timeline.svg)

### 読み方のポイント
- 縦線が密集するほどイベント集中期間です。直近の混雑やピークを直感的に伝えられます。
- 線の高さや色を変えると、イベントタイプや重み付けも同時に表現できます。
- 時系列が長い場合は月ごとに区切ったり、スクロール表示にすると読みやすくなります。
