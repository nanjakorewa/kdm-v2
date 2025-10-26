---
title: "Rugplotで個票を補助線として表示"
pre: "6.2.10 "
weight: 10
title_suffix: "ヒストグラムや KDE の補足として使う"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` を実行すると
`static/images/visualize/distribution/rugplot.svg` が生成されます。
{{% /notice %}}

ヒストグラムや KDE に `rugplot` を重ねると、データの一つひとつがどこにあるかが分かりやすくなります。

```python
import seaborn as sns
import matplotlib.pyplot as plt

diamonds = sns.load_dataset("diamonds").sample(300, random_state=0)

fig, ax = plt.subplots(figsize=(6, 3.5))
sns.kdeplot(data=diamonds, x="price", ax=ax, color="#0ea5e9")
sns.rugplot(data=diamonds, x="price", ax=ax, color="#1d4ed8", alpha=0.4)

ax.set_xlabel("価格 ($)")
ax.set_ylabel("密度")
ax.set_title("ダイヤ価格の KDE + Rugplot")
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/rugplot.svg")
```

![rugplot](/images/visualize/distribution/rugplot.svg)

### 読み方のポイント

- Rugplot の短い線が密集している箇所はデータが多い。
- 色を薄くしておけば KDE の主役を奪わずに情報を補足できる。
- 大量データで Rugplot を使うと描画負荷が高いので、サンプリングや `height` の調整を検討する。
