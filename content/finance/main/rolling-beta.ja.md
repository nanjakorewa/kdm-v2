---
title: "ローリングβで市場感応度を測る"
pre: "7.1.13 "
weight: 13
not_use_colab: true
---

β（ベータ）は銘柄の市場感応度を示す代表的な指標です。固定期間で推計するだけでなく、ローリングで追跡すると市場環境の変化を捉えやすくなります。ここでは擬似データを用いて、`pandas` で β を計算する関数と、結果をグラフ化する方法をまとめます。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(7)
plt.style.use("scripts/k_dm.mplstyle")

dates = pd.date_range("2022-01-01", periods=520, freq="B")
market_returns = np.random.normal(0.0004, 0.01, size=len(dates))

signals = {}
for ticker, beta, vol in [
    ("AAA", 1.2, 0.012),
    ("BBB", 0.7, 0.009),
]:
    residual = np.random.normal(0, vol, size=len(dates))
    signals[ticker] = beta * market_returns + residual

returns = pd.DataFrame({"market": market_returns, **signals}, index=dates)
```

### ローリングβを計算する関数

```python
def rolling_beta(asset: pd.Series, benchmark: pd.Series, window: int = 120) -> pd.Series:
    def _calc(x: pd.Series) -> float:
        ref = benchmark.loc[x.index]
        cov = np.cov(x, ref)[0, 1]
        var = np.var(ref)
        return cov / var if var > 0 else np.nan

    return asset.rolling(window).apply(_calc, raw=False)


betas = pd.DataFrame(
    {
        ticker: rolling_beta(returns[ticker], returns["market"])
        for ticker in ["AAA", "BBB"]
    }
)
```

### β の推移を可視化

```python
fig, ax = plt.subplots(figsize=(9, 4.2))
betas.plot(ax=ax, linewidth=1.6)
ax.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1, label="β=1")
ax.set_title("120 営業日ローリングβ（サンプルデータ）")
ax.set_ylabel("β")
ax.legend()
ax.grid(alpha=0.3)

output = Path("static/images/finance/main/rolling_beta_trend.svg")
output.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(output)
```

![ローリングβの推移例](/images/finance/main/rolling_beta_trend.svg)

### 実務でのポイント
- パラメータ（ウィンドウ長やリターンの頻度）は資産ごとに調整します。ボラティリティが高い資産は長めのウィンドウで平滑化すると安定します。
- ベンチマークには株価指数だけでなく、金利やコモディティなど複数のファクターを組み合わせることで、マルチファクターモデルに拡張できます。
- ローリングβをアラート化し、閾値を超えたらリバランスを検討する、といったルールベースの運用に組み込むのも有効です。
