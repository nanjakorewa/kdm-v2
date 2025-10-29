---
title: "ローリングβをヒートマップで可視化する"
pre: "7.2.5 "
weight: 5
title_suffix: "市場指数との連動性が高まるタイミングを把握"
---

株式の市場感応度（β）は期間によって大きく変わります。60 日ローリングで算出した β をヒートマップにすると、いつ市場と連動しやすくなったか、逆にディフェンシブに振る舞ったかが一目瞭然です。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)
plt.style.use("scripts/k_dm.mplstyle")

dates = pd.date_range("2023-01-01", periods=260, freq="B")
market = np.random.normal(0.0005, 0.01, size=len(dates)).cumsum()
returns = pd.DataFrame(index=dates)
returns["market"] = np.diff(np.insert(market, 0, 0))

for ticker, beta, vol in [
    ("AAA", 1.1, 0.012),
    ("BBB", 0.8, 0.009),
    ("CCC", 1.4, 0.015),
    ("DDD", 0.6, 0.008),
]:
    noise = np.random.normal(0, vol, size=len(dates))
    returns[ticker] = beta * returns["market"] + noise

window = 60
rolling_beta = (
    returns[["AAA", "BBB", "CCC", "DDD"]]
    .rolling(window)
    .apply(
        lambda col: np.cov(col, returns.loc[col.index, "market"])[0, 1]
        / np.var(returns.loc[col.index, "market"]),
        raw=False,
    )
)

fig, ax = plt.subplots(figsize=(10, 4.5))
im = ax.imshow(
    rolling_beta.T,
    aspect="auto",
    cmap="RdYlGn",
    vmin=0.4,
    vmax=1.6,
)

ax.set_yticks(range(len(rolling_beta.columns)))
ax.set_yticklabels(rolling_beta.columns)
ax.set_xticks(np.linspace(0, len(dates) - 1, 6))
ax.set_xticklabels(pd.Series(dates).dt.strftime("%Y-%m-%d").iloc[::len(dates)//6])
ax.set_title("60 日ローリング β（市場指数との回帰）")
ax.set_xlabel("日付")
ax.set_ylabel("銘柄")

cbar = fig.colorbar(im, ax=ax, pad=0.01)
cbar.ax.set_ylabel("β", rotation=90)

output = Path("static/images/finance/visualize/rolling_beta_heatmap.svg")
output.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(output)
```

![ローリングβヒートマップ](/images/finance/visualize/rolling_beta_heatmap.svg)

### 読み方のポイント
- β が 1.0 を超えると市場より値動きが大きくなり、0.8 など 1.0 を下回るとディフェンシブに変化したと解釈できます。
- ノイズが多い銘柄は β の変動が激しくなるため、ローリングウィンドウを長めに設定するか、指数平滑平均で安定化すると読みやすくなります。
- MSCI などのセクター指数で同じヒートマップを作ると、セクターごとのリスクオン / リスクオフ局面を定性的に把握できます。
