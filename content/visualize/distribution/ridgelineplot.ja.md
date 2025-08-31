---
title: "リッジラインプロット"
pre: "6.2.3 "
weight: 3
title_replace: "pythonでリッジラインプロットを作成する"
---

複数のグループの分布とその差異を視覚化するために使用されるチャート。分布のグラフを重ねて表示する（ことが多い）ので、グループごとのわずかな分布の違いや頂点の位置の違い・変化を可視化しやすい。

{{% notice document %}}
[ridgeplot: beautiful ridgeline plots in Python](https://github.com/tpvasconcelos/ridgeplot)
{{% /notice %}}

```python
import numpy as np
import seaborn as sns
from ridgeplot import ridgeplot


# 可視化したいカラムのリスト
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# サンプルデータ
df = sns.load_dataset("iris")
df = df[columns]

# リッジラインプロット
fig = ridgeplot(
samples=df.values.T, labels=columns, colorscale="viridis", coloralpha=0.6
)
fig.update_layout(height=500, width=800)
fig.show()
```

![png](/images/visualize/distribution/ridgeline.png)!
