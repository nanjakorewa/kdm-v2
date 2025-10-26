---
title: "箱ひげ図で分布の要約を示す"
pre: "6.2.5 "
weight: 5
title_suffix: "四分位数と外れ値を一目で把握"
---

箱ひげ図は中央値・四分位数・外れ値を 1 本で表せる定番チャートです。カテゴリごとに比較すると、分散の違いが見えます。

```python
import seaborn as sns
import matplotlib.pyplot as plt

mpg = sns.load_dataset("mpg").dropna(subset=["mpg", "origin"])

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=mpg, x="origin", y="mpg", palette="Set2", ax=ax)

ax.set_xlabel("生産地域")
ax.set_ylabel("燃費 (MPG)")
ax.set_title("地域別燃費の箱ひげ図")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/boxplot.svg")
```

![boxplot](/images/visualize/distribution/boxplot.svg)

### 読み方のポイント

- 箱は四分位範囲（IQR）、中央線は中央値。ひげは通常 1.5×IQR の範囲を示す。
- 外れ値が多すぎる場合は、別図で詳細を確認するか上限を調整する。
- 箱ひげ図を横にするとラベルが長い場合でも読みやすくなる。
