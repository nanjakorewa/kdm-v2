---
title: "2次元 KDE で密度を等高線表示"
pre: "6.2.6 "
weight: 6
title_suffix: "散布図では見えにくい密度をなめらかに描く"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` 実行で
`static/images/visualize/distribution/kde2d.svg` が作成されます。
{{% /notice %}}

`seaborn.kdeplot` を使うと 2 変数の密度を等高線または塗りつぶしで描けます。散布図が密集しているときに有効です。

```python
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins").dropna(subset=["bill_length_mm", "bill_depth_mm"])

fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.kdeplot(
    data=penguins,
    x="bill_length_mm",
    y="bill_depth_mm",
    hue="species",
    fill=True,
    thresh=0.05,
    levels=6,
    alpha=0.6,
    ax=ax,
)

ax.set_xlabel("くちばし長 (mm)")
ax.set_ylabel("くちばし深さ (mm)")
ax.set_title("ペンギン種別の 2 次元 KDE")
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/kde2d.svg")
```

![kde2d](/images/visualize/distribution/kde2d.svg)

### 読み方のポイント

- 等高線が密な箇所ほどデータが集まっている。色の濃淡で頻度を直感的に把握できる。
- `thresh` を調整すると僅かな密度の輪郭を省略できる。
- 大量データでは計算が重い場合があるため、サンプリングや`bw_adjust`で帯域を調整する。
