---
title: "密度プロット"
pre: "6.2.2 "
weight: 2
title_replace: "pythonで密度プロットを作成する"
---

数値データがどのように分布しているかを滑らかな曲線で可視化します。

{{% notice document %}}
[seaborn.kdeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 4))
df = sns.load_dataset("iris")

for column, color in zip(["sepal_length", "sepal_width", "petal_length", "petal_width"], ["#2563eb", "#0ea5e9", "#22c55e", "#f97316"]):
    sns.kdeplot(data=df, x=column, ax=ax, color=color, label=column)

ax.set_xlabel("測定値")
ax.set_ylabel("密度")
ax.set_title("Iris データの密度プロット")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/densityplot.svg")
```

![density plot](/images/visualize/distribution/densityplot.svg)

### 読み方のポイント

- 曲線が高い部分はデータが集中している。裾の形状で分布の幅が分かる。
- 複数系列を重ねる場合は色と凡例で識別し、透明度を下げて重なりを見やすくすると効果的。
- 平滑化帯域 `bw_adjust` を調整すると曲線の滑らかさをコントロールできる。
