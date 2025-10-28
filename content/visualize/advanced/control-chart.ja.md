---
title: "管理図でプロセスの異常値を見逃さない"
pre: "6.7.23 "
weight: 23
title_suffix: "平均±3σで工程のばらつきを監視"
---

問い合わせ件数や製造歩留まりなど、ばらつきを監視したいときは管理図が有効です。統計的な管理限界を描くことで、異常値を即座に検知できます。

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(5)
values = 45 + rng.normal(0, 3, size=28)
values[[6, 18]] += np.array([12, -10])  # 異常値を混ぜる

mean = values.mean()
std = values.std(ddof=1)
ucl = mean + 3 * std
lcl = mean - 3 * std

fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(values, marker="o", color="#0ea5e9")
ax.axhline(mean, color="#334155", linewidth=1.3, label="平均")
ax.axhline(ucl, color="#ef4444", linestyle="--", label="UCL")
ax.axhline(lcl, color="#ef4444", linestyle="--", label="LCL")

ax.set_xticks(range(0, len(values), 4), labels=[f"W{i+1}" for i in range(0, len(values), 4)])
ax.set_title("呼び出し処理時間の管理図")
ax.set_ylabel("平均処理秒数")
ax.grid(alpha=0.2)

for idx, val in enumerate(values):
    if val > ucl or val < lcl:
        ax.annotate(
            "異常",
            (idx, val),
            xytext=(idx + 0.5, val + 4),
            arrowprops=dict(arrowstyle="->", color="#ef4444"),
            color="#ef4444",
        )

ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("static/images/visualize/advanced/control-chart.svg")
```

![control chart](/images/visualize/advanced/control-chart.svg)

### 読み方のポイント
- 平均線の上下に管理限界 (±3σ) を引くことで、統計的に異常な点を即座に把握できます。
- 異常点が続く場合は、工程の状態が変わったサインと捉え原因を深掘りしましょう。
- 折れ線の接続方法やマーカーの形を調整すると、レポートのトーンに合わせやすくなります。
