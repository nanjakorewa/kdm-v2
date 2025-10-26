---
title: "ステップラインで区切りのある変化を表す"
pre: "6.4.5 "
weight: 5
title_suffix: "料金改定など段階的な変化に向く"
---

価格改定のように段階的に値が変わる場合はステップチャートが便利です。`plt.step` を使うと階段状の線を描けます。

```python
import matplotlib.pyplot as plt

months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]
price = [980, 980, 1100, 1100, 1250, 1250, 1350]

fig, ax = plt.subplots(figsize=(6, 4))
ax.step(months, price, where="post", color="#0ea5e9", linewidth=2.5)

ax.set_ylabel("料金（円）")
ax.set_title("プラン料金の改定推移")
ax.set_ylim(900, 1400)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/visualize/line/step_line.svg")
```

![step line](/images/visualize/line/step_line.svg)

### 読み方のポイント

- `where="post"` で区間後半に値が変わる形式を表現できる。改定が前倒しなら `"pre"` を使う。
- 価格など水平区間が多い場合は注釈で改定理由を添えると伝わりやすい。
- 突然の変化を強調したいときは線の色を切り替える演出も有効。
