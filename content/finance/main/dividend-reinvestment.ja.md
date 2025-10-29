---
title: "配当再投資で総リターンをシミュレーションする"
pre: "7.1.14 "
weight: 14
not_use_colab: true
---

配当を現金のまま受け取る場合と、すぐに再投資する場合では長期の総リターンに大きな差が生まれます。擬似データを使って、配当再投資シナリオと現金保持シナリオを `pandas` で比較します。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(21)
plt.style.use("scripts/k_dm.mplstyle")

dates = pd.date_range("2015-01-01", periods=9 * 12, freq="M")
price = 100 * np.exp(np.cumsum(np.random.normal(0.005, 0.04, len(dates))))
dividend_yield = np.random.uniform(0.015, 0.025, len(dates))  # 年率 1.5〜2.5%

frame = pd.DataFrame({"price": price, "dividend_yield": dividend_yield}, index=dates)
frame["dividend_cash"] = frame["price"] * frame["dividend_yield"] / 12
```

### シナリオ 1: 配当を現金のまま保持

```python
cash_hold = frame["dividend_cash"].cumsum()
total_value_cash = frame["price"] + cash_hold
```

### シナリオ 2: 配当を即座に再投資

```python
shares = 1.0
cash = 0.0
portfolio_value = []

for price, dividend in zip(frame["price"], frame["dividend_cash"]):
    cash += dividend
    additional_shares = cash / price
    shares += additional_shares
    cash = 0.0
    portfolio_value.append(shares * price)

total_value_reinvest = pd.Series(portfolio_value, index=frame.index)
```

### 結果を比較

```python
comparison = pd.DataFrame(
    {
        "現金保持": total_value_cash,
        "再投資": total_value_reinvest,
    }
)

fig, ax = plt.subplots(figsize=(9, 4.2))
comparison.plot(ax=ax)
ax.set_title("配当再投資シミュレーション（擬似データ）")
ax.set_ylabel("評価額（基準=100）")
ax.grid(alpha=0.3)

output = Path("static/images/finance/main/dividend_reinvestment.svg")
output.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(output)
```

![配当再投資と現金保持の比較](/images/finance/main/dividend_reinvestment.svg)

### 実務に向けて
- 実際のデータを利用する際は、配当落ち日・支払日を考慮して再投資のタイミングを合わせます。
- 海外株の場合は源泉徴収税や為替影響を考慮する必要があります。税引後配当を用意し、合計ネットリターンを比較するとより現実的です。
- ポートフォリオ全体では、銘柄ごとの配当再投資をまとめて日次で行うか、一定金額が貯まった段階でまとめて再投資するなど、運用ルールを明示して検証しましょう。
