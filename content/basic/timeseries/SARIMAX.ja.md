---
title: "SARIMAX モデル"
pre: "2.8.3 "
weight: 3
title_suffix: "季節性と外部要因を組み込む ARIMA"
---

{{< lead >}}
SARIMAX は ARIMA に季節項（Seasonal）と外生変数（Exogenous）を拡張したモデルで、実務の複雑な時系列に対応できます。
{{< /lead >}}

## モデルの構成

- **非季節部分**：ARIMA(p, d, q) の階差と AR/MA 成分。
- **季節部分**：季節階差と (P, D, Q, s) による季節 AR/MA。
- **外生変数**：実測済みの説明変数を回帰として追加できる。

## Python コード例

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

rng = np.random.default_rng(32)
dates = pd.date_range("2018-01-01", periods=6 * 12, freq="M")
trend = 1.2 * np.arange(len(dates))
seasonal = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
noise = rng.normal(0, 3, len(dates))
series = pd.Series(200 + trend + seasonal + noise, index=dates)

train = series.iloc[:-12]
test_index = series.index[-12:]

model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
result = model.fit(disp=False)
forecast = result.get_forecast(steps=12)
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int()

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.plot(series.index, series, color="#cbd5f5", label="実測値（全体）")
ax.plot(train.index, train, color="#2563eb", linewidth=1.2, label="学習区間")
ax.plot(test_index, pred_mean, color="#f97316", linewidth=1.6, label="SARIMAX 予測")
ax.fill_between(
    test_index,
    pred_ci.iloc[:, 0],
    pred_ci.iloc[:, 1],
    color="#f97316",
    alpha=0.2,
    label="95% 信頼区間",
)
ax.set_title("SARIMAX モデルの予測")
ax.set_xlabel("年月")
ax.set_ylabel("値")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("static/images/timeseries/sarimax.svg")
```

![plot](/images/timeseries/sarimax.svg)

## モデル選択のコツ

- 季節周期 \(s\) を先に決める（例：年次季節なら 12、週次季節なら 52）。
- 非季節と季節の階差を入れすぎると過差分になる。まずは D=1（季節差）から試す。
- AIC/BIC や残差診断を使い、残差が白色雑音に近いかを確認する。

## 活用シーン

- 小売の需要予測：プロモーションや休日を説明変数として追加。
- エネルギー需要：気温・湿度など気象データを外生変数に。
- Web トラフィック：イベントカレンダーや広告指標を取り込む。

---

実装時は `statsmodels` の `SARIMAX` クラスを使うと手軽です。より大規模なデータや高速化が必要な場合は `pmdarima` や Prophet、機械学習モデルと合わせて比較検討しましょう。

