---
title: "Prophetのモデルの中身"
pre: "5.6.2 "
weight: 2
title_suffix: "を仕組みを理解する"
---

{{% youtube "JIjETRPKpT4" %}}

Prophetがどのようにしてモデルを作成しているのか、もう少し詳細に見てみたいと思います。

{{% notice ref %}}
Taylor, Sean J., and Benjamin Letham. "Forecasting at scale." The American Statistician 72.1 (2018): 37-45.
{{% /notice %}}



```python
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
from prophet import Prophet
```

## 実験に使用するデータ
１年間のデータを用意します。このデータには以下の特徴があります。

- 2020/1/1 ~ 2020/12/31 の期間のデータ
- 週ごとに周期的な数値を取る
- 土日は大きな値をとる
- 不定期にイベントが発生し、その時は非常に大きな値を取る(`horidays_index`に該当する日付を指しています)


```python
date = pd.date_range("2020-01-01", periods=365, freq="D")
horidays_index = random.sample([i for i in range(365)], 10)
y = [
    np.log(10 + i + 10 * np.cos(i))  # トレンド
    + np.cos(di.weekday() * np.pi / 28) * 3  # 週ごとに周期性あり
    + (di.weekday() in {5, 6}) * 0.5  #       〃
    + (i in horidays_index) * 2  # 祝日だけ数値を増やす
    + np.random.rand() / 10  # ノイズ
    for i, di in enumerate(date)
]

df = pd.DataFrame({"ds": date, "y": y})
df.index = date

plt.title("サンプルデータ")
sns.lineplot(data=df)
plt.show()
```


    
![png](/images/timeseries/forecast/002-Prophet-alg_files/002-Prophet-alg_5_0.png)
    


## トレンドの指定
### growth="linear"


```python
# モデルを作成
m = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    growth="linear",
)
m.fit(df)

# 将来を予測
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
fig = m.plot(forecast)
plt.axvspan(18627, 18627 + 90, color="coral", alpha=0.4, label="予測期間")
plt.legend()
plt.show()
```

    Initial log joint probability = -2.99674
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       1040.17   0.000845008       95.4398      0.3546      0.3546      128   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         166       1045.08   8.28451e-05       66.0092   1.087e-06       0.001      252  LS failed, Hessian reset 
         184       1045.09   1.58284e-05       61.1258   3.535e-07       0.001      307  LS failed, Hessian reset 
         199       1045.09   1.20349e-07        45.161      0.0243           1      330   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         264       1045.71   0.000316932       140.641   3.665e-06       0.001      456  LS failed, Hessian reset 
         292       1045.82   5.70302e-05       48.9256   1.155e-06       0.001      533  LS failed, Hessian reset 
         299       1045.82   7.71897e-06       40.8043      0.3543           1      544   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         305       1045.82   5.91182e-07       46.3158   2.336e-08       0.001      594  LS failed, Hessian reset 
         317       1045.82   2.63213e-06       51.4268   5.486e-08       0.001      649  LS failed, Hessian reset 
         343       1045.82   1.59952e-05        25.621   2.535e-07       0.001      728  LS failed, Hessian reset 
         353       1045.82   6.70942e-07       31.5639   1.825e-08       0.001      779  LS failed, Hessian reset 
         355       1045.82   3.11455e-08       40.1124     0.08424           1      783   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance



    
![png](/images/timeseries/forecast/002-Prophet-alg_files/002-Prophet-alg_7_1.png)
    


### growth="flat"


```python
# モデルを作成
m = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    growth="flat",
)
m.fit(df)

# 将来を予測
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
fig = m.plot(forecast)
plt.axvspan(18627, 18627 + 90, color="coral", alpha=0.4, label="予測期間")
plt.legend()
plt.show()
```

    Initial log joint probability = -3.30635
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
           8       725.322    0.00030891     0.0789655       0.955       0.955       14   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance



    
![png](/images/timeseries/forecast/002-Prophet-alg_files/002-Prophet-alg_9_1.png)
    


## 季節変化の指定
### seasonalityが全てFalse


```python
# モデルを作成
m = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    growth="linear",
)
m.fit(df)

# 将来を予測
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
fig = m.plot(forecast)
plt.axvspan(18627, 18627 + 90, color="coral", alpha=0.4, label="予測期間")
plt.legend()
plt.show()
```

    Initial log joint probability = -2.99674
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       1014.78   0.000250992       97.5427       1.468     0.01468      134   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         199          1020    0.00045602       60.8104           1           1      268   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         208        1020.1   0.000167489       76.8606   1.772e-06       0.001      314  LS failed, Hessian reset 
         299       1020.33   0.000353008       41.1849        2.43       0.243      446   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         352       1020.86   8.23232e-05       43.9782   1.438e-06       0.001      586  LS failed, Hessian reset 
         399       1020.99   0.000244078        56.987           1           1      657   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         488       1021.04   3.90795e-05       57.3821   6.974e-07       0.001      856  LS failed, Hessian reset 
         499       1021.04   1.23592e-05       41.4144           1           1      876   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         510       1021.04   4.89524e-06       59.8544   1.187e-07       0.001      929  LS failed, Hessian reset 
         571       1021.06   1.29427e-06        39.655   2.478e-08       0.001     1071  LS failed, Hessian reset 
         598       1021.07   1.94887e-05       41.0613   3.113e-07       0.001     1157  LS failed, Hessian reset 
         599       1021.07   4.80514e-06       27.1307       0.643       0.643     1158   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         645       1021.07   8.81062e-06       74.3981   1.365e-07       0.001     1264  LS failed, Hessian reset 
         660       1021.07   2.61181e-07       41.0723   5.687e-09       0.001     1321  LS failed, Hessian reset 
         661       1021.07    9.2387e-08       26.7092      0.8838      0.8838     1322   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance



    
![png](/images/timeseries/forecast/002-Prophet-alg_files/002-Prophet-alg_11_1.png)
    


### weekly_seasonality=True


```python
# モデルを作成
m = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    growth="linear",
)
m.fit(df)

# 将来を予測
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
fig = m.plot(forecast)
plt.axvspan(18627, 18627 + 90, color="coral", alpha=0.4, label="予測期間")
plt.legend()
plt.show()
```

    Initial log joint probability = -2.99674
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       1040.17   0.000845008       95.4398      0.3546      0.3546      128   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         166       1045.08   8.28451e-05       66.0092   1.087e-06       0.001      252  LS failed, Hessian reset 
         184       1045.09   1.58284e-05       61.1258   3.535e-07       0.001      307  LS failed, Hessian reset 
         199       1045.09   1.20349e-07        45.161      0.0243           1      330   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         264       1045.71   0.000316932       140.641   3.665e-06       0.001      456  LS failed, Hessian reset 
         292       1045.82   5.70302e-05       48.9256   1.155e-06       0.001      533  LS failed, Hessian reset 
         299       1045.82   7.71897e-06       40.8043      0.3543           1      544   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         305       1045.82   5.91182e-07       46.3158   2.336e-08       0.001      594  LS failed, Hessian reset 
         317       1045.82   2.63213e-06       51.4268   5.486e-08       0.001      649  LS failed, Hessian reset 
         343       1045.82   1.59952e-05        25.621   2.535e-07       0.001      728  LS failed, Hessian reset 
         353       1045.82   6.70942e-07       31.5639   1.825e-08       0.001      779  LS failed, Hessian reset 
         355       1045.82   3.11455e-08       40.1124     0.08424           1      783   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance



    
![png](/images/timeseries/forecast/002-Prophet-alg_files/002-Prophet-alg_13_1.png)
    


### yearly_seasonality=True


```python
# モデルを作成
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    growth="linear",
)
m.fit(df)

# 将来を予測
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
fig = m.plot(forecast)
plt.axvspan(18627, 18627 + 90, color="coral", alpha=0.4, label="予測期間")
plt.legend()
plt.show()
```

    Initial log joint probability = -2.99674
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       1076.96   1.35095e-06        94.027      0.3775      0.3775      136   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         125       1076.96   2.65686e-06        89.152   2.895e-08       0.001      203  LS failed, Hessian reset 
         164       1076.96   2.28191e-08       91.6592      0.4114      0.4114      258   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance



    
![png](/images/timeseries/forecast/002-Prophet-alg_files/002-Prophet-alg_15_1.png)
    


## 休日・イベント効果
### 休日・イベントの指定なし(`holidays=None`)


```python
# モデルを作成
m = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    growth="linear",
    holidays=None,  # 休日・イベントの指定なし
)
m.fit(df)

# 将来を予測
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
fig = m.plot(forecast)
plt.axvspan(18627, 18627 + 90, color="coral", alpha=0.4, label="予測期間")
plt.legend()
plt.show()
```

    Initial log joint probability = -2.99674
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       1040.17   0.000845008       95.4398      0.3546      0.3546      128   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         166       1045.08   8.28451e-05       66.0092   1.087e-06       0.001      252  LS failed, Hessian reset 
         184       1045.09   1.58284e-05       61.1258   3.535e-07       0.001      307  LS failed, Hessian reset 
         199       1045.09   1.20349e-07        45.161      0.0243           1      330   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         264       1045.71   0.000316932       140.641   3.665e-06       0.001      456  LS failed, Hessian reset 
         292       1045.82   5.70302e-05       48.9256   1.155e-06       0.001      533  LS failed, Hessian reset 
         299       1045.82   7.71897e-06       40.8043      0.3543           1      544   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         305       1045.82   5.91182e-07       46.3158   2.336e-08       0.001      594  LS failed, Hessian reset 
         317       1045.82   2.63213e-06       51.4268   5.486e-08       0.001      649  LS failed, Hessian reset 
         343       1045.82   1.59952e-05        25.621   2.535e-07       0.001      728  LS failed, Hessian reset 
         353       1045.82   6.70942e-07       31.5639   1.825e-08       0.001      779  LS failed, Hessian reset 
         355       1045.82   3.11455e-08       40.1124     0.08424           1      783   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance



    
![png](/images/timeseries/forecast/002-Prophet-alg_files/002-Prophet-alg_17_1.png)
    


### 休日・イベントの指定あり


```python
# モデルを作成
df_holidays = pd.DataFrame(
    {"holiday": "event", "ds": [di for i, di in enumerate(date) if i in horidays_index]}
)

m = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=False,
    growth="linear",
    holidays=df_holidays,  # 休日・イベントの指定あり
)
m.fit(df)

# 将来を予測
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
fig = m.plot(forecast)
plt.axvspan(18627, 18627 + 90, color="coral", alpha=0.4, label="予測期間")
plt.legend()
plt.show()
```

    Initial log joint probability = -2.99674
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       1343.59    0.00181622       336.302      0.9914      0.9914      120   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         199       1366.61    0.00129331       915.041      0.2388      0.9805      232   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         265       1372.99   0.000111116       289.963   4.926e-07       0.001      351  LS failed, Hessian reset 
         299       1374.36    0.00236219       89.8611           1           1      393   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         399       1375.61    0.00707432       82.6526           1           1      509   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         431       1375.93   8.82293e-05       240.658    4.05e-07       0.001      581  LS failed, Hessian reset 
         453       1376.02   6.12661e-05       95.4391    1.44e-06       0.001      644  LS failed, Hessian reset 
         471       1376.06    1.2062e-05       14.2904   2.239e-07       0.001      709  LS failed, Hessian reset 
         485        1376.1    1.9227e-05       69.5783   2.446e-07       0.001      772  LS failed, Hessian reset 
         493       1376.12   3.43778e-05       51.3583   1.873e-06       0.001      837  LS failed, Hessian reset 
         499       1376.13   2.95479e-05       34.8029      0.5524      0.5524      847   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         509       1376.13   8.88412e-06       44.6026   5.452e-07       0.001      897  LS failed, Hessian reset 
         521       1376.14    1.6737e-05       27.0944   1.441e-07       0.001      947  LS failed, Hessian reset 
         531       1376.14   7.82708e-06       47.5548   1.997e-07       0.001     1000  LS failed, Hessian reset 
         567       1376.28   3.47594e-05       113.402   3.767e-07       0.001     1079  LS failed, Hessian reset 
         599       1376.35   0.000359311       95.2959           1           1     1118   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         604       1376.35   4.84989e-06        13.344   1.606e-07       0.001     1168  LS failed, Hessian reset 
         612       1376.35   2.91053e-08       12.4075     0.04535           1     1191   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance



    
![png](/images/timeseries/forecast/002-Prophet-alg_files/002-Prophet-alg_19_1.png)
    

