---
title: "相関ヒートマップ"
pre: "6.4.1 "
weight: 1
title_suffix: "特徴量同士の関係性を色で把握する"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` を実行すると
`static/images/visualize/correlation/corr_heatmap.svg` を再生成します。
{{% /notice %}}

相関係数を色で表現すると、どの指標が一緒に動きやすいかを視覚的に把握できます。`pandas` で相関行列を作り、`seaborn.heatmap` で描画するのが簡単です。

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(
    {
        "売上":    [100, 120, 150, 170, 190, 210, 180, 205],
        "集客":    [230, 260, 300, 320, 350, 370, 330, 360],
        "滞在時間": [18, 20, 23, 25, 27, 29, 26, 28],
        "リピート率": [0.32, 0.34, 0.36, 0.38, 0.4, 0.41, 0.39, 0.4],
    }
)

corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    corr,
    annot=True,
    cmap="Blues",
    vmin=-1,
    vmax=1,
    square=True,
    fmt=".2f",
    ax=ax,
)
ax.set_title("売上指標の相関ヒートマップ")
fig.tight_layout()
fig.savefig("static/images/visualize/correlation/corr_heatmap.svg")
```

![heatmap](/images/visualize/correlation/corr_heatmap.svg)

### 読み方のポイント

- 対角要素は必ず 1 になるので、オフダイアゴナルを重視する。
- 正の相関（青が濃い）が強いペアは、どちらかの指標を伸ばす施策がもう一方にも効く可能性が高い。
- 逆に相関が弱い指標は独立に改善できる領域だと考えられる。
