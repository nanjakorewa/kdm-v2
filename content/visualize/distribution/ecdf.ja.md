---
title: "ECDF で累積分布を可視化"
pre: "6.2.7 "
weight: 7
title_suffix: "値がどこまでで何％たまるかを見る"
---

経験的累積分布関数（ECDF）は、ある値以下のサンプル割合を表現するシンプルなチャートです。閾値判断に役立ちます。

```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

fig, ax = plt.subplots(figsize=(6, 4))
sns.ecdfplot(data=tips, x="total_bill", hue="time", ax=ax)

ax.set_xlabel("会計金額 ($)")
ax.set_ylabel("累積割合")
ax.set_title("会計金額の ECDF")
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/ecdf.svg")
```

![ecdf](/images/visualize/distribution/ecdf.svg)

### 読み方のポイント

- 曲線の勾配が急な部分はデータが集中している。緩い部分は散らばりが大きい。
- 例えば「80% の顧客は 30 ドル以下」といった閾値判断が容易になる。
- 比較する系列が多い場合は色数を抑え、凡例と線種で区別すると読みやすい。
