---
title: "人口ピラミッド型でユーザー層の偏りを確認"
pre: "6.7.18 "
weight: 18
title_suffix: "男女やカテゴリを左右対称で比較"
---

男女比やA/Bテストでの属性差を伝えたいとき、左右に振り分ける人口ピラミッド型チャートが効果的です。年齢レンジ別の偏りが一目瞭然になります。

```python
import numpy as np
import matplotlib.pyplot as plt

age_groups = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54"]
male = np.array([4200, 3800, 3300, 2700, 2200, 1600, 1200])
female = np.array([4000, 3600, 3500, 2900, 2400, 1700, 1300])

fig, ax = plt.subplots(figsize=(6, 4.8))
ax.barh(age_groups, male, color="#38bdf8", label="男性")
ax.barh(age_groups, -female, color="#f472b6", label="女性")

ax.set_xlabel("ユーザー数")
ax.set_title("年齢帯別の会員数（人口ピラミッド）")
ax.set_xlim(-4500, 4500)
ax.set_xticks([-4000, -2000, 0, 2000, 4000], labels=["4000", "2000", "0", "2000", "4000"])
ax.legend(loc="upper right")

for idx, (m_val, f_val) in enumerate(zip(male, female)):
    ax.text(m_val + 80, idx, f"{m_val}", va="center")
    ax.text(-f_val - 280, idx, f"{f_val}", va="center")

ax.axvline(0, color="#0f172a", linewidth=1)
ax.spines[["right", "top"]].set_visible(False)
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/population-pyramid.svg")
```

![population pyramid](/images/visualize/advanced/population-pyramid.svg)

### 読み方のポイント
- 左右の長さで属性別の量を比較。左右差が大きい帯ほど偏りが強いです。
- 軸を人数スケールで合わせると視覚的な比較が正確になります。
- ラベルを添えると棒の長さを読み取らなくても人数を把握できます。
