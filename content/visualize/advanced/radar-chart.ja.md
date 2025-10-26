---
title: "レーダーチャートで多項目を比較"
pre: "6.7.2 "
weight: 2
title_suffix: "複数指標を一度にレポートする"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` を実行すると
`static/images/visualize/advanced/radar.svg` が生成されます。
{{% /notice %}}

プロダクト A/B の KPI をレーダーチャートで比較する例です。`matplotlib` の極座標を使用します。

```python
import numpy as np
import matplotlib.pyplot as plt

metrics = ["UX", "機能", "安定性", "速度", "サポート"]
values_a = np.array([4.2, 4.5, 4.0, 3.9, 4.4])
values_b = np.array([3.8, 4.1, 4.3, 4.5, 3.7])

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
values_a = np.concatenate((values_a, [values_a[0]]))
values_b = np.concatenate((values_b, [values_b[0]]))
angles = np.concatenate((angles, [angles[0]]))

fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax.plot(angles, values_a, color="#2563eb", linewidth=2, label="プロダクトA")
ax.fill(angles, values_a, color="#2563eb", alpha=0.25)
ax.plot(angles, values_b, color="#f97316", linewidth=2, label="プロダクトB")
ax.fill(angles, values_b, color="#f97316", alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
ax.set_yticks([3, 4, 5])
ax.set_ylim(0, 5)
ax.set_title("KPI 比較レーダーチャート")
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/radar.svg")
```

![radar](/images/visualize/advanced/radar.svg)

### 読み方のポイント

- 指標ごとの面積が広いほど性能が高い。重ねることで優劣が一目で分かる。
- 角度間隔は等間隔にすると比較しやすい。項目が多すぎると読みづらくなる。
- 主要指標は別途棒グラフなどで補足し、数値も併記すると説得力が増す。
