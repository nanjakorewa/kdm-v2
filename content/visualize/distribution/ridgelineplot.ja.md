---
title: "リッジラインプロット"
pre: "6.2.3 "
weight: 3
title_replace: "pythonでリッジラインプロットを作成する"
---

カテゴリごとの分布を重ねて表示するリッジラインプロットは、季節性やクラス間の違いを把握するのに便利です。

```python
import seaborn as sns
import matplotlib.pyplot as plt

mpg = sns.load_dataset("mpg").dropna(subset=["mpg", "origin"])

sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=(6, 5))
sns.violinplot(
    data=mpg,
    x="mpg",
    y="origin",
    scale="width",
    inner=None,
    palette="Set2",
    ax=ax,
)
ax.set_xlabel("燃費 (MPG)")
ax.set_ylabel("生産地域")
ax.set_title("地域別燃費のリッジライン風プロット")
ax.grid(axis="x", alpha=0.2)

fig.tight_layout()
fig.savefig("static/images/visualize/distribution/ridgeline.svg")
```

![ridgeline](/images/visualize/distribution/ridgeline.svg)

### 読み方のポイント

- 各カテゴリの分布形状と中央値の位置を同時に把握できる。
- 重なり具合からカテゴリ間の類似性が分かる。必要に応じて透明度を調整する。
- 実際には `joypy` など専用ライブラリを使うと純粋なリッジラインを描ける。ここでは `violinplot` を活用して近似している。
