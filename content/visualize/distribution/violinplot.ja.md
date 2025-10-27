---
title: "バイオリンプロット"
pre: "6.2.4 "
weight: 4
title_replace: "pythonでバイオリンプロットを作成する"
---

バイオリンプロットは箱ひげ図に KDE を重ねたようなチャートで、分布の形状と外れ値を同時に把握できます。

```python
import seaborn as sns
import matplotlib.pyplot as plt

penguins = sns.load_dataset("penguins").dropna(subset=["bill_length_mm", "species"])

fig, ax = plt.subplots(figsize=(6, 4))
sns.violinplot(data=penguins, x="species", y="bill_length_mm", palette="Set3", ax=ax)
ax.set_xlabel("ペンギンの種類")
ax.set_ylabel("くちばしの長さ (mm)")
ax.set_title("種別のくちばし長 バイオリンプロット")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/violin.svg")
```

![violin plot](/images/visualize/distribution/violin.svg)

### 読み方のポイント

- バイオリンの幅が広い部分はデータが密集している。箱ひげ図より滑らかに分布を表現できる。
- 中央の白い点線は中央値と四分位範囲を示す。箱ひげ図と同時に表示すると解釈しやすい。
- カテゴリが多い場合は横向きにするか、色数を絞って視認性を保つ。
