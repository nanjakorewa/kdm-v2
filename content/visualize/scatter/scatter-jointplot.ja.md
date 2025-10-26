---
title: "Jointplotで相関と分布を同時に表示"
pre: "6.5.4 "
weight: 4
title_suffix: "散布図＋周辺分布の組み合わせ"
---

`seaborn.jointplot` は中央に散布図、上下左右にヒストグラムを描画してくれる便利な関数です。

```python
import seaborn as sns

fmri = sns.load_dataset("fmri")
subset = fmri[fmri["region"] == "frontal"].copy()

g = sns.jointplot(
    data=subset,
    x="timepoint",
    y="signal",
    kind="kde",
    fill=True,
    cmap="Blues",
    height=4.5,
)
g.fig.suptitle("fMRI 信号の Jointplot", y=1.02)
g.fig.savefig("static/images/visualize/scatter/jointplot.svg")
```

![jointplot](/images/visualize/scatter/jointplot.svg)

### 読み方のポイント

- 中央の散布図が濃い領域は出現頻度が高い。周辺の KDE で各軸の分布も把握できる。
- `kind="hex"` や `kind="hist"` に切り替えると別タイプのチャートになる。
- データ量が多いと等高線の描画が重くなるため、サンプリングするか `levels` を減らすと良い。
