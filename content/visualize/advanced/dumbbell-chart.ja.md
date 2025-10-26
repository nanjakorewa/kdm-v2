---
title: "ダンベルチャートでビフォー・アフター比較"
pre: "6.7.4 "
weight: 4
title_suffix: "開始と終了の差を1本で強調"
---

ダンベルチャートは開始値と終了値を 1 本の線で結び、差分を視覚的に示します。

```python
import numpy as np
import matplotlib.pyplot as plt

departments = ["営業", "開発", "サポート", "マーケ", "人事"]
before = np.array([68, 72, 65, 70, 60])
after = np.array([78, 80, 72, 74, 68])

fig, ax = plt.subplots(figsize=(6, 4))
ax.hlines(departments, before, after, color="#94a3b8", linewidth=3)
ax.scatter(before, departments, color="#ef4444", s=80, label="施策前")
ax.scatter(after, departments, color="#22c55e", s=80, label="施策後")

ax.set_xlabel("エンゲージメントスコア")
ax.set_title("施策前後のエンゲージメント比較（ダンベルチャート）")
ax.legend(loc="lower right")
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/advanced/dumbbell.svg")
```

![dumbbell chart](/images/visualize/advanced/dumbbell.svg)

### 読み方のポイント

- 線の向きが右に伸びているほど改善、左に伸びているほど悪化を表す。
- 点の色や凡例で時点を区別し、どちらが開始か一目で分かるようにする。
- 差分を数値で表示したい場合はラベルや注釈を追加すると理解が深まる。
