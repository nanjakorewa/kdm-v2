---
title: "Prophetを使ってみる"
pre: "2.8.1 "
weight: 1
---

{{% youtube "uUMDo8HOcrI" %}}

<div class="pagetop-box">
  <p><b>Prophet</b>は、Facebook（現Meta）が開発した時系列予測のライブラリです。  
  季節性やトレンドを簡単にモデル化でき、統計の専門知識がなくても直感的に利用できます。  
  ビジネス現場でも「売上予測」「アクセス数予測」などに広く使われています。</p>
</div>

---

## 1. Prophetとは？
- 時系列データを「トレンド + 季節性 + 休日効果」に分解して予測するモデル。  
- 欠損値や外れ値にも比較的強い。  
- scikit-learn風のインターフェースで、学習・予測がシンプル。  

公式ドキュメント：  
- [Installation](https://facebook.github.io/prophet/docs/installation.html)  
- [Quick Start](https://facebook.github.io/prophet/docs/quick_start.html)  

{{% notice seealso %}}
[K_DM - 時系列 > 予測 > Prophet](https://k-dm.work/ja/timeseries/forecast/001-prophet/) でもProphetを扱っています。  
本章の続きは [K_DM - 時系列](https://k-dm.work/ja/timeseries/) に掲載しています。
{{% /notice %}}

---

## 2. ダミーの時系列データを作成

```python
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet

sns.set(rc={"figure.figsize": (15, 8)})

# 365日のデータ
date = pd.date_range("2020-01-01", periods=365, freq="D")
y = [np.cos(di.weekday()) + di.month % 2 + np.log(i + 1) for i, di in enumerate(date)]

df = pd.DataFrame({"ds": date, "y": y})
df.index = date

sns.lineplot(data=df)
```

![png](/images/basic/timeseries/prophet_files/prophet_3_1.png)

---

## 3. Prophetでモデル学習

```python
m = Prophet(yearly_seasonality=False, daily_seasonality=True)
m.fit(df)
```

出力例（一部省略）：
```
Optimization terminated normally: 
  Convergence detected: relative gradient magnitude is below tolerance
```

---

## 4. 未来データを作成して予測

```python
# 90日先まで予測
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)

fig1 = m.plot(forecast)
```

![png](/images/basic/timeseries/prophet_files/prophet_8_0.png)

---

## 5. Prophetのポイント
- **季節性の自動モデリング**  
  （例：日ごと・週ごと・年ごとの周期性）  
- **トレンド変化点の自動検出**  
  （急な上昇や下降に対応可能）  
- **拡張性**：休日効果などカスタム要素を追加できる。  

---

## まとめ
- Prophetは「簡単に」「柔軟に」使える時系列予測ツール。  
- データフレームを渡すだけで未来予測が可能。  
- 季節性・トレンドのあるデータに強い。  
- 実務でも「売上・需要・アクセス数」などの予測にすぐ応用できる。  

---
