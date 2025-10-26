---
title: "Q-Q プロットで正規性をチェック"
pre: "6.2.8 "
weight: 8
title_suffix: "理論分布とのズレを確認する"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` 実行で
`static/images/visualize/distribution/qqplot.svg` が更新されます。
{{% /notice %}}

`scipy.stats.probplot` を用いて、データが正規分布に従うかどうかを視覚的に判断します。直線から大きく外れるほど正規性から逸脱しています。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data = np.random.normal(loc=0, scale=1, size=500)

fig, ax = plt.subplots(figsize=(5, 5))
stats.probplot(data, dist="norm", plot=ax)

ax.set_title("Q-Q プロット（正規分布との比較）")
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/qqplot.svg")
```

![qqplot](/images/visualize/distribution/qqplot.svg)

### 読み方のポイント

- 点が45度線上に並んでいれば正規分布に近い。末端が曲がる場合は裾が重い・軽い。
- 別の理論分布を試したい場合は `dist` 引数を変更する。
- データの平均・分散を合わせて報告すると、分布の解釈がしやすい。
