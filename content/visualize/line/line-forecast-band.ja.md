---
title: "予測区間をバンドで表現"
pre: "6.4.7 "
weight: 7
title_suffix: "平均予測と上下限を同時に見せる"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` 実行で
`static/images/visualize/line/forecast_band.svg` が生成されます。
{{% /notice %}}

機械学習モデルの予測値と、その上下限を `fill_between` で帯状に描画すると、不確実性も合わせて伝えられます。

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 13)
forecast = 80 + 4 * x + np.random.normal(scale=3, size=len(x))
lower = forecast - np.random.uniform(5, 8, size=len(x))
upper = forecast + np.random.uniform(5, 8, size=len(x))

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x, forecast, color="#2563eb", linewidth=2.5, label="予測値")
ax.fill_between(x, lower, upper, color="#93c5fd", alpha=0.4, label="予測区間 (80%)")

ax.set_xticks(x)
ax.set_xlabel("月")
ax.set_ylabel("売上予測（百万円）")
ax.set_title("売上予測と信頼区間")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/visualize/line/forecast_band.svg")
```

![forecast band](/images/visualize/line/forecast_band.svg)

### 読み方のポイント

- バンドの透明度を上げすぎると折れ線が見づらくなるので 0.3〜0.4 程度が目安。
- 予測区間の幅が広い箇所は不確実性が高いため、追加データの収集やモデル改善を検討できる。
- 予測値と実績を重ねる場合は、色や線種を変えて見間違いを防ぐ。
