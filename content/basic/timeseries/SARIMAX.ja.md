---
title: "SARIMAX モデル"
pre: "2.8.3 "
weight: 3
title_suffix: "季節性と外部要因を組み込む ARIMA"
---

{{< lead >}}
SARIMAX（Seasonal ARIMA with eXogenous regressors）は、ARIMA に季節成分と外生変数を加えて現実的な時系列を表現するモデルです。売上データのように周期とプロモーション効果が混在するケースで威力を発揮します。
{{< /lead >}}

---

## 1. モデル構造

- **通常項 (p, d, q)**：標準の ARIMA と同じく、自己回帰・差分・移動平均。
- **季節項 (P, D, Q, s)**：`s` 期間ごとの周期性を `P`（自己回帰）、`D`（季節差分）、`Q`（移動平均）で表現。
- **外生変数 (exog)**：プロモーションや祝日、気温など、系列に影響する説明変数を一緒に学習。

季節差分 `D` を 1 に設定すると、前年同月との違いをモデリングできます。季節自己回帰 `P` や季節移動平均 `Q` を増やすと周期パターンの追従が細かくなりますが、過学習しやすいので `s` に対して小さめに設定するのがコツです。

---

## 2. Python で実装

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 週次季節性とトレンドを持つサンプル
date_index = pd.date_range("2021-01-01", periods=300, freq="D")
base = 100 + 0.2 * np.arange(len(date_index))
seasonal = 10 * np.sin(2 * np.pi * date_index.dayofweek / 7)
promo = (date_index.dayofweek >= 5).astype(int)  # 週末フラグ
noise = np.random.normal(scale=3.0, size=len(date_index))
sales = base + seasonal + 8 * promo + noise

df = pd.DataFrame({"y": sales, "promo": promo}, index=date_index)
train = df.iloc[:260]
test = df.iloc[260:]

model = SARIMAX(
    train["y"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7),
    exog=train[["promo"]],
)
fitted = model.fit(disp=False)

forecast = fitted.get_forecast(
    steps=len(test),
    exog=test[["promo"]],
)
mean_forecast = forecast.predicted_mean
conf_int = forecast.conf_int()
```

`seasonal_order=(1, 1, 1, 7)` は 7 日周期の季節自己回帰・季節差分・季節移動平均すべて 1 の設定。週次の繰り返しを捉えたい場合に適しています。

---

## 3. ハイパーパラメータの直感

| パラメータ | 意味 | 増やすとどうなるか |
| --- | --- | --- |
| `order(p,d,q)` | 短期の自己相関・トレンド・ショックを捉える | `p` / `q` を増やすと細かな揺れに追従しやすいが、データ量が少ないと暴走しやすい |
| `seasonal_order(P,D,Q,s)` | 周期性の大きさと形 | `P`/`Q` を増やすと季節波形を柔軟に表現。ただしパラメータ爆発に注意。`s` は周期（例：12=月次） |
| `exog` | 外部要因 | 影響を直接モデル化できるが、将来予測時に exog の予測値が必要になる |

グリッドサーチを行う場合は `p` `q` を 0〜2 程度、`P` `Q` も同程度で始めると現実的。`D` を 1 にすると季節性が強くても安定しやすいですが、季節差分し過ぎると系列ががさつくので注意。

---

## 4. 結果の可視化

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(train.index, train["y"], label="train")
plt.plot(test.index, test["y"], label="actual")
plt.plot(mean_forecast.index, mean_forecast, label="forecast")
plt.fill_between(
    mean_forecast.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color="gray",
    alpha=0.2,
    label="confidence interval",
)
plt.legend()
plt.title("SARIMAX 予測 vs 実績")
plt.show()
```

外生変数を入れると、プロモーション期間に合わせて予測が押し上がる様子が確認できます。

---

## 5. 実務でのポイント

- **季節周期の決め方**：日次なら 7（週）、365（年）、月次なら 12 を試す。ACF の季節ラグが強いところが目安。
- **外生変数のズレ**：予測対象期間で外生変数を先に準備しておく。将来が未知なら別モデルで予測する必要がある。
- **安定性の確認**：残差の自己相関・正規性・定常性をチェックし、必要なら `exog` を見直す。
- **計算コスト**：季節パラメータが増えるとフィッティングに時間が掛かる。最小限の組み合わせからスタート。

---

## まとめ

- SARIMAX は ARIMA を拡張し、季節性と外部要因を同時に扱える柔軟なモデル。
- `seasonal_order` と `exog` を上手に設定すると、週末効果やキャンペーンなどビジネス要因を自然に組み込める。
- AIC/BIC と残差診断を併用しつつ、必要なパラメータだけを増やすことで安定した時系列予測につながる。

---
