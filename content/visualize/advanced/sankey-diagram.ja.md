---
title: "サンキーダイアグラムでフロー分岐を把握"
pre: "6.7.25 "
weight: 25
title_suffix: "matplotlib.sankey で顧客流れを可視化"
---

顧客がチャネルを経由して別のアクションへ流れる状況は、サンキーダイアグラムで枝分かれを示すと分かりやすくなります。`matplotlib.sankey` を使えばシンプルなフローが描画できます。

```python
from matplotlib.sankey import Sankey
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.4, 4))
sankey = Sankey(ax=ax, unit=None, format="%+d", gap=0.45)

sankey.add(
    flows=[1000, -420, -280, -200, -100],
    labels=["流入", "無料登録", "カート離脱", "FAQ閲覧", "直帰"],
    orientations=[0, 1, -1, -1, -1],
    pathlengths=[0.5, 0.7, 0.6, 0.5, 0.5],
    trunklength=1.0,
)
sankey.add(
    flows=[420, -260, -100, -60],
    labels=["無料登録", "有料化", "再来訪", "離脱"],
    orientations=[0, 1, -1, -1],
    prior=0,
    connect=(1, 0),
    pathlengths=[0.6, 0.6, 0.4, 0.4],
)
diagrams = sankey.finish()
for text in diagrams[0].texts + diagrams[1].texts:
    text.set_fontsize(10)

ax.set_title("来訪後の顧客フロー（サンキーダイアグラム）")

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/sankey-diagram.svg")
```

![sankey diagram](/images/visualize/advanced/sankey-diagram.svg)

### 読み方のポイント
- 幹の太さがフロー量を表すので、どの分岐で大きく減っているかが直感的に把握できます。
- 連結している枝をたどることで、特定の出口（例: 有料化）までのボトルネックを追跡できます。
- 分岐が多すぎる場合は、主要経路に絞って描くと読みやすさが保てます。
