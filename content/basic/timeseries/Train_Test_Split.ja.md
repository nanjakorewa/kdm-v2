---
title: "学習データとテストデータの区切り"
pre: "2.8.19 "
weight: 19
title_suffix: "時間順に分割した様子を可視化する"
---

{{< lead >}}
時系列ではシャッフルせず、過去から未来へ時間順に分割するのが基本です。区切りを図示して合意を得ましょう。
{{< /lead >}}

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(12)
dates = pd.date_range("2020-01-01", periods=240, freq="D")
series = pd.Series(500 + np.cumsum(rng.normal(0, 4, len(dates))), index=dates)

split_point = len(series) - 60
train = series.iloc[:split_point]
test = series.iloc[split_point:]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train.index, train, color="#2563eb", label="学習データ")
ax.plot(test.index, test, color="#ef4444", label="テストデータ")
ax.axvline(test.index[0], color="#475569", linestyle="--", linewidth=1.2)
ax.text(
    test.index[0],
    series.min(),
    "  ← テスト開始",
    color="#475569",
    va="bottom",
)
ax.set_title("時間順の学習・テスト分割")
ax.set_xlabel("日付")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/train_test_split.svg")
```

![plot](/images/timeseries/train_test_split.svg)

### 読み方のポイント

- 未来のデータを学習に含めないことが重要。可視化で境界を示すと関係者と議論しやすい。
- テスト区間を十分確保できない場合はクロスバリデーションやウォークフォワードを検討する。
- バリデーションを追加する場合も同様に時間順で並べ、リークが起きないようにする。

