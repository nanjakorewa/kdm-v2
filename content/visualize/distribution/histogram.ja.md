---
title: "ヒストグラム"
pre: "6.2.1 "
weight: 1
title_replace: "pythonでヒストグラムを作成する"
---

数値データがどのように散らばっているかを可視化します。

{{% notice document %}}
[sns.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn.histplot)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("iris")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df, x="sepal_length", binwidth=0.5, color="#2563eb", ax=ax)
ax.set_xlabel("がく片の長さ (cm)")
ax.set_ylabel("頻度")
ax.set_title("Iris データのヒストグラム")
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/histogram.svg")
```

![histogram](/images/visualize/distribution/histogram.svg)

### 読み方のポイント

- ビン幅を変えると見え方が大きく変わるため、分析目的に合わせて調整する。
- データが偏っている場合は対数軸を使うと細部が見やすくなる。
- KDE や rugplot を重ねると連続的な傾向が分かりやすい。
