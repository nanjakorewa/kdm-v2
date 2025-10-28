---
title: "ラグプロットで自己相関の癖を掘る"
pre: "6.7.24 "
weight: 24
title_suffix: "直近値と過去値の関係を散布図に"
---

時系列の自己相関を見つけるには、1期前の値と現在値を散布図にしたラグプロットが有効です。右上がりであれば自己相関が強く、パターンが読めます。

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)
series = np.cumsum(rng.normal(0, 1.2, size=120)) + 50
lag = 1

x_prev = series[:-lag]
x_curr = series[lag:]

fig, ax = plt.subplots(figsize=(4.4, 4.4))
ax.scatter(x_prev, x_curr, color="#38bdf8", alpha=0.7)

coef = np.corrcoef(x_prev, x_curr)[0, 1]
ax.set_xlabel("1期前の値")
ax.set_ylabel("現在の値")
ax.set_title(f"ラグ{lag}の散布図（相関係数 {coef:.2f}）")
ax.grid(alpha=0.2)

lims = [min(series) - 2, max(series) + 2]
ax.plot(lims, lims, color="#475569", linestyle="--", linewidth=1)
ax.set_xlim(lims)
ax.set_ylim(lims)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/lag-scatter.svg")
```

![lag scatter](/images/visualize/advanced/lag-scatter.svg)

### 読み方のポイント
- 点が右上がりに並ぶほど自己相関が強いサインです。トレンドが続く傾向にあると判断できます。
- 円状に広がるなら自己相関が弱く、ランダムウォークに近い挙動だと推測できます。
- 複数のラグを小さな multiples で並べると、どの遅れを特徴量に使うべきか判断しやすくなります。
