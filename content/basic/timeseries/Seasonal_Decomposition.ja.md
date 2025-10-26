---
title: "季節分解とトレンド分析"
pre: "2.8.4 "
weight: 4
title_suffix: "STL で系列を分けて理解する"
---

{{< lead >}}
季節分解は時系列をトレンド・季節性・残差に分割し、系列の構造を直感的に捉える手法です。STL（Seasonal-Trend decomposition using Loess）は柔軟な平滑化を行い、季節パターンの変化にも追従しやすくなっています。
{{< /lead >}}

---

## 1. なぜ分解するのか

- トレンドと季節性を分離すると、どの成分がモデル化の邪魔をしているか把握できる。
- 異常検知において、残差だけを監視することで外れ値を発見しやすくなる。
- ARIMA や Prophet を適用する前の前処理として、季節成分を除いた系列を作るなど応用が効く。

---

## 2. Python で STL 分解

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

index = pd.date_range("2020-01-01", periods=365, freq="D")
trend = 0.2 * np.arange(365)
seasonal = 5 * np.sin(2 * np.pi * index.dayofyear / 30)
noise = np.random.normal(scale=1.5, size=365)
series = trend + seasonal + noise

stl = STL(series, period=30, robust=True)
result = stl.fit()

result.plot()
plt.show()
```

`period` は季節周期。30 日周期の変動を想定した例です。`robust=True` にすると外れ値の影響を受けにくい平滑化になります。

---

## 3. ハイパーパラメータの直感

| パラメータ | 役割 | 変化させると |
| --- | --- | --- |
| `period` | 季節性の周期 | 大きくすると長期の繰り返しを抜き出す。周期を誤るとトレンドに季節成分が混ざる |
| `seasonal` | 季節成分を平滑化する Loess の窓幅 | 小さいと急な季節変化に敏感、大きいと滑らかだが細部を見落とす |
| `trend` | トレンド成分の窓幅 | 大きくすると長い傾向を抽出、小さいと短期変動までトレンド扱いになる |
| `low_pass` | 残差からノイズを除く平滑化 | ノイズをさらに滑らかにしたいときに調整 |

STL では `seasonal`, `trend`, `low_pass` を奇数にする必要があります。既定値でうまくいくことが多いですが、週次季節性（7）や月次季節性（12）などデータの周期に合わせた `period` を設定することが重要です。

---

## 4. 応用例

- **季節調整済み系列の作成**：`series - result.seasonal` で季節性を除いた値を得て、ARIMA や回帰の入力に使う。
- **残差で異常検知**：`result.resid` の絶対値が大きい区間をアラートとして扱う。
- **トレンド比較**：複数のシリーズでトレンド成分を比較することで、構造的な変化を発見。
- **季節性の変化をモニタリング**：時間とともに季節成分の振幅が増減していないかウォッチ。

---

## 5. 実務でのヒント

- 期間が短いデータでは季節分解の信頼度が下がる。最低でも数周期分のデータを用意。
- 変則的なカレンダー（祝日・移動祝日）が重要な場合は、ダミー変数を外生変数として別モデルに組み込む。
- STL に加えて、`seasonal_decompose`（古典的分解）や Facebook Prophet の `add_regressor` なども検討すると良い。

---

## まとめ

- STL 分解はトレンド・季節性・残差を柔軟に分離し、時系列の理解と前処理を助ける。
- `period` と平滑化パラメータの調整で、どの周期や変動に注目するかを制御できる。
- 予測モデルを構築する前に成分を可視化することで、適切なモデル選択や異常検知の精度向上につながる。

---
