---
title: "Hexbin プロットで密度を捉える"
pre: "6.7.1 "
weight: 1
title_suffix: "散布図よりも高密度領域を見せやすい"
---

散布図に点が重なりすぎる場合は、六角形のビンでカウントする Hexbin プロットが便利です。`matplotlib.hexbin` で簡単に描けます。

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
session = rng.gamma(shape=3, scale=12, size=1000)   # 滞在時間（分）
amount = rng.normal(loc=2500, scale=700, size=1000) # 購入金額（円）

fig, ax = plt.subplots(figsize=(6, 4))
hb = ax.hexbin(
    amount,
    session,
    gridsize=18,
    cmap="Blues",
    mincnt=1,
)
ax.set_xlabel("購入金額（円）")
ax.set_ylabel("滞在時間（分）")
ax.set_title("滞在時間 × 購入金額の Hexbin")
cb = fig.colorbar(hb, ax=ax, shrink=0.85)
cb.set_label("件数")

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/hexbin.svg")
```

![hexbin](/images/visualize/advanced/hexbin.svg)

### 読み方のポイント

- 色が濃い六角形ほどデータが密集している領域。値の偏りを把握しやすい。
- `mincnt` を指定すると、一定数以上データが入ったセルだけを描画できる。
- カラーバーを付ければ件数を定量的に説明でき、ヒートマップとしても機能する。
