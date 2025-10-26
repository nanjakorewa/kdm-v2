---
title: "ARIMA モデル"
pre: "2.8.2 "
weight: 2
title_suffix: "自己回帰と移動平均の組み合わせ"
---

{{< lead >}}
ARIMA（AutoRegressive Integrated Moving Average）は、時系列を差分で平滑化しつつ自己回帰と移動平均を組み合わせて将来を予測する古典的モデルです。トレンドが緩やかで季節性が弱い系列に特に効果を発揮します。
{{< /lead >}}

---

## 1. モデルの考え方

- **自己回帰 (AR)**：直近の値が次の値に影響する。`p` を増やすと「どれだけ前の情報まで見るか」を広げる。
- **差分 (I)**：`d` 回の差分で非定常成分（トレンド）を取り除く。`d=0` なら差分なし、`d=1` で一階差分。
- **移動平均 (MA)**：過去の予測誤差を補正に使う。`q` を増やすとショックの持続を表現しやすくなる。

`p,d,q` を組み合わせ、ARIMA(p, d, q) と表記します。差分を取り過ぎるとノイズが増えて予測が不安定になるので、トレンドが消える最小限の `d` を選ぶのがコツです。

---

## 2. Python で学習と予測

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# サンプル時系列（トレンド + ノイズ）
index = pd.date_range("2020-01-01", periods=200, freq="D")
signal = 0.3 * np.arange(200) + 10 * np.sin(np.arange(200) / 12)
noise = np.random.normal(scale=2.0, size=200)
series = pd.Series(signal + noise, index=index)

train = series[:180]
test = series[180:]

model = ARIMA(train, order=(2, 1, 1))
fitted = model.fit()

forecast = fitted.forecast(steps=len(test))
print(forecast.head())
```

`order=(2, 1, 1)` は AR(2) + 1階差分 + MA(1) の構成を意味します。`summary()` を表示すると係数や情報量基準 (AIC) を確認できます。

---

## 3. ハイパーパラメータと直感

| パラメータ | 直感的な意味 | 大きくすると |
| --- | --- | --- |
| `p` | どこまで過去の値を参照するか | 過去の揺らぎが影響し周期性を捉えやすいが、過剰にすると過学習へ |
| `d` | トレンド除去の差分回数 | 非定常を抑え安定するが、取りすぎはノイズを増やし予測が暴れがち |
| `q` | 誤差を何ステップ補正するか | 突発的な外乱の余韻を吸収できるが、過剰にすると遅れが発生 |

`seasonal_order` を指定できる `SARIMAX` モデルを使うと季節成分も扱えます。まずは ACF/PACF プロットを眺め、`p` `q` の候補を決めてから AIC/BIC で絞るのが一般的です。

---

## 4. 残差診断

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

resid = fitted.resid.dropna()
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(resid)
axes[0].set_title("残差時系列")
plot_acf(resid, lags=30, ax=axes[1])
axes[1].set_title("残差の自己相関")
plt.show()
```

- 残差が白色雑音に近い → モデルが系列構造をうまく捉えているサイン。
- 明確な自己相関が残る → `p` や `q` を増やす、季節要素を追加するなど調整が必要。

---

## 5. 実務のヒント

- **差分の回数を減らす努力**：`d=1` で十分なケースがほとんど。`d=2` 以上はデータが強いトレンドを持つ場合のみ。
- **季節性が強い場合**：`statsmodels` の `SARIMAX` や Prophet、季節分解 + ARIMA などを検討する。
- **外部変数を加えたい**：`exog` 引数でプロモーションや気象など説明変数を組み込める。
- **スケールの安定化**：対数変換や Box-Cox で分散を均すと RMSE が改善することが多い。

---

## まとめ

- ARIMA は差分で定常化した系列に AR と MA を組み合わせるシンプルな予測モデル。
- `p` `d` `q` の調整で過去の記憶・トレンド除去・外乱吸収のバランスを取る。
- 残差診断と情報量基準を併用し、必要なら季節項や外部変数を追加していくと安定した予測を得やすい。

---
