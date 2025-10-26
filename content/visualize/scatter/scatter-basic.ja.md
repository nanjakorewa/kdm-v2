---
title: "散布図の基本形"
pre: "6.5.1 "
weight: 1
title_suffix: "相関の有無をざっくり確認する"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` 実行時に
`static/images/visualize/scatter/basic.svg` が生成されます。
{{% /notice %}}

身長と体重の関係をランダムデータで描いた基本的な散布図です。`seaborn.scatterplot` を使うとスタイルを統一できます。

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
height = rng.normal(167, 8, size=120)
weight = 0.6 * height + rng.normal(0, 5, size=120) + 30

fig, ax = plt.subplots(figsize=(5.5, 4))
sns.scatterplot(x=height, y=weight, ax=ax, color="#2563eb", edgecolor="white")

ax.set_xlabel("身長（cm）")
ax.set_ylabel("体重（kg）")
ax.set_title("身長と体重の関係")
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/scatter/basic.svg")
```

![scatter basic](/images/visualize/scatter/basic.svg)

### 読み方のポイント

- 相関が強いほど点が細い帯状にまとまる。相関が弱いと円形に散らばる。
- 点が重なりすぎる場合はアルファ値（`alpha`）を下げる、もしくは Hexbin など別手法を検討する。
- 外れ値を見つけたら、別途明示して原因分析につなげる。
