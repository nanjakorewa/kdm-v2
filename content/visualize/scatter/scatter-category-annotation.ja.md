---
title: "カテゴリ別の散布図に注釈を追加"
pre: "6.5.6 "
weight: 6
title_suffix: "平均値ラインや注釈で洞察を補足"
---

クラスタごとに色分けし、平均値ラインと注釈を加えた散布図です。注目してほしい領域をテキストで示すと伝わりやすくなります。

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
cluster = np.concatenate([np.full(80, "A"), np.full(70, "B"), np.full(60, "C")])
x = np.concatenate([rng.normal(40, 6, 80), rng.normal(55, 5, 70), rng.normal(65, 4, 60)])
y = np.concatenate([rng.normal(70, 5, 80), rng.normal(60, 6, 70), rng.normal(80, 5, 60)])
palette = {"A": "#2563eb", "B": "#10b981", "C": "#f97316"}

fig, ax = plt.subplots(figsize=(6, 4))
for cat in np.unique(cluster):
    mask = cluster == cat
    ax.scatter(x[mask], y[mask], label=f"グループ {cat}", color=palette[cat], alpha=0.75, edgecolors="white", s=60)

ax.axvline(np.mean(x), color="#9ca3af", linestyle="--", linewidth=1)
ax.axhline(np.mean(y), color="#9ca3af", linestyle="--", linewidth=1)

ax.annotate(
    "高スコア帯",
    xy=(66, 84),
    xytext=(72, 90),
    arrowprops=dict(arrowstyle="->", color="#1f2937"),
    fontsize=11,
)

ax.set_xlabel("評価指標 X")
ax.set_ylabel("評価指標 Y")
ax.set_title("カテゴリ別散布図と注釈")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/visualize/scatter/category_annotation.svg")
```

![scatter annotation](/images/visualize/scatter/category_annotation.svg)

### 読み方のポイント

- 平均線を描くと全体の基準が分かり、各クラスターの位置付けを説明しやすくなる。
- 注釈は矢印やハイライトと組み合わせると視線が誘導できる。
- 凡例の順序を重視順・規模順に並べ替えると、意思決定者への説明がスムーズ。
