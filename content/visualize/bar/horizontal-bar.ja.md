---
title: "横向き棒グラフでランキング表示"
pre: "6.3.2 "
weight: 2
title_suffix: "ラベルが長い場合に有効"
---

{{% notice tip %}}
`python scripts/generate_visualize_assets.py` 実行時に
`static/images/visualize/bar/horizontal_bar.svg` が作成されます。
{{% /notice %}}

アンケート結果のようにラベルが長いときは横向きの棒グラフが適しています。

```python
import matplotlib.pyplot as plt

features = [
    "検索のしやすさ",
    "ページ速度",
    "モバイル対応",
    "デザイン",
    "問い合わせ対応",
]
score = [4.6, 4.2, 4.8, 4.1, 4.5]

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.barh(features, score, color="#0ea5e9")

ax.invert_yaxis()
ax.set_xlabel("満足度 (5 点満点)")
ax.set_title("サイト改善アンケート 結果")
ax.set_xlim(0, 5)
ax.bar_label(bars, fmt="%.1f", padding=6)
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/bar/horizontal_bar.svg")
```

![horizontal bar](/images/visualize/bar/horizontal_bar.svg)

### 読み方のポイント

- 項目数が多いときは横棒が視線移動しやすい。
- 値を降順に並べるとランキングが明確になる。
- 棒の色を統一し、強調が必要な場合のみ別色を使う。
