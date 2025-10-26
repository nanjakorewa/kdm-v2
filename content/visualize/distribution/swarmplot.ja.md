---
title: "Swarmplot で個々のデータを重ならず配置"
pre: "6.2.9 "
weight: 9
title_suffix: "カテゴリごとの分布と個票を同時に表示"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` 実行で
`static/images/visualize/distribution/swarmplot.svg` が生成されます。
{{% /notice %}}

Swarmplot は各データ点を重ならないようにずらして配置することで、個々の値を保ちながら分布の形を表現できます。

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

fig, ax = plt.subplots(figsize=(6, 4))
sns.swarmplot(data=tips, x="day", y="total_bill", hue="sex", dodge=True, ax=ax)

ax.set_xlabel("来店曜日")
ax.set_ylabel("会計金額 ($)")
ax.set_title("曜日別の会計金額 Swarmplot")
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/swarmplot.svg")
```

![swarmplot](/images/visualize/distribution/swarmplot.svg)

### 読み方のポイント

- 点が一列に並ぶ高さが密度の高さを示す。外れ値も点として残るので見逃しにくい。
- データ数が非常に多い場合は計算が重いので、サンプリングや `size` の調整が必要。
- `dodge=True` を使うと hue のカテゴリごとに列を分けて比較できる。
