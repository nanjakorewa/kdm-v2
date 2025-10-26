---
title: "注目期間をハイライトする"
pre: "6.4.6 "
weight: 6
title_suffix: "繁忙期やイベント期間を背景で強調"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` 実行で
`static/images/visualize/line/highlight_range.svg` が更新されます。
{{% /notice %}}

通常の折れ線に `ax.axvspan` を組み合わせると、特定期間を背景色で強調できます。大型キャンペーン期間などに便利です。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dates = pd.date_range("2024-04-01", periods=30, freq="D")
sessions = np.random.poisson(lam=500, size=len(dates)) + np.linspace(0, 80, len(dates))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(dates, sessions, color="#6366f1", linewidth=2)

campaign_start = pd.Timestamp("2024-04-10")
campaign_end = pd.Timestamp("2024-04-18")
ax.axvspan(campaign_start, campaign_end, color="#fbbf24", alpha=0.2, label="キャンペーン")

ax.set_ylabel("セッション数")
ax.set_title("日次セッションとキャンペーン期間")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/visualize/line/highlight_range.svg")
```

![highlight range](/images/visualize/line/highlight_range.svg)

### 読み方のポイント

- 背景色は淡い色にし、折れ線よりも目立ち過ぎないようにする。
- 期間を複数指定する場合は色分けやストライプなどで区別する。
- 注釈を加えてイベント名や施策名を記録しておくと分析メモになる。
