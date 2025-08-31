---
title: "mplfinance"
pre: "7.1.2 "
weight: 2
not_use_colab: true
title_suffix: "をpythonで実行してみる"
---


{{% youtube "Z_3CmggMqCY" %}}

<div class="pagetop-box">
<a href="https://github.com/matplotlib/mplfinance">mplfinance</a>とはファイナンスデータを可視化するためのmatplotlibの拡張です。今回はこれを使って、金融データからグラフを作ってみようと思います。基本的な使い方は<a href="https://github.com/matplotlib/mplfinance#tutorials">mplfinanceのリポジトリの『Tutorial』</a>にて説明されており、これをみながらグラフを作ってみます。
</div>



## データの読み込み
{{% notice document %}}
[Pandas-datareader | Remote Data Access](https://pandas-datareader.readthedocs.io/en/latest/remote_data.html)
{{% /notice %}}


```python
import os
import time
import pandas as pd
import pandas_datareader.data as web
from pandas_datareader._utils import RemoteDataError
```

{{% notice info %}}
[Yahoo Terms of Service](https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html)
{{% /notice %}}

```python
def get_finance_data(
    ticker_symbol: str, start="2021-01-01", end="2021-06-30", savedir="data"
) -> pd.DataFrame:
    """株価を記録したデータを取得します

    Args:
        ticker_symbol (str): Description of param1
        start (str): 期間はじめの日付, optional.
        end (str): 期間終わりの日付, optional.

    Returns:
        res: 株価データ

    """
    res = None
    filepath = os.path.join(savedir, f"{ticker_symbol}_{start}_{end}_historical.csv")
    os.makedirs(savedir, exist_ok=True)

    if not os.path.exists(filepath):
        try:
            time.sleep(5.0)
            res = web.DataReader(ticker_symbol, "yahoo", start=start, end=end)
            res.to_csv(filepath, encoding="utf-8-sig")
        except (RemoteDataError, KeyError):
            print(f"ticker_symbol ${ticker_symbol} が正しいか確認してください。")
    else:
        res = pd.read_csv(filepath, index_col="Date")
        res.index = pd.to_datetime(res.index)

    assert res is not None, "データ取得に失敗しました"
    return res
```


```python
# 銘柄名、期間、保存先ファイル
ticker_symbol = "NVDA"
start = "2021-01-01"
end = "2021-06-30"

# データを取得する
df = get_finance_data(ticker_symbol, start=start, end=end, savedir="../data")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Adj Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-12-31</th>
      <td>131.509995</td>
      <td>129.149994</td>
      <td>131.365005</td>
      <td>130.550003</td>
      <td>19242400.0</td>
      <td>130.413864</td>
    </tr>
    <tr>
      <th>2021-01-04</th>
      <td>136.524994</td>
      <td>129.625000</td>
      <td>131.042496</td>
      <td>131.134995</td>
      <td>56064000.0</td>
      <td>130.998245</td>
    </tr>
    <tr>
      <th>2021-01-05</th>
      <td>134.434998</td>
      <td>130.869995</td>
      <td>130.997498</td>
      <td>134.047501</td>
      <td>32276000.0</td>
      <td>133.907700</td>
    </tr>
    <tr>
      <th>2021-01-06</th>
      <td>132.449997</td>
      <td>125.860001</td>
      <td>132.225006</td>
      <td>126.144997</td>
      <td>58042400.0</td>
      <td>126.013443</td>
    </tr>
    <tr>
      <th>2021-01-07</th>
      <td>133.777496</td>
      <td>128.865005</td>
      <td>129.675003</td>
      <td>133.440002</td>
      <td>46148000.0</td>
      <td>133.300842</td>
    </tr>
  </tbody>
</table>
</div>



## OHLCをプロット
OHLCとは始値・高値・安値・終値のことで、普段最もよくみるグラフの一つです。
ローソク足と、線でのプロットをしてみます。

{{% notice document %}}
[mplfinance | matplotlib utilities for the visualization, and visual analysis, of financial data](https://github.com/matplotlib/mplfinance)
{{% /notice %}}


```python
import mplfinance as mpf

mpf.plot(df, type="candle", style="starsandstripes", figsize=(12, 4))
mpf.plot(df, type="line", style="starsandstripes", figsize=(12, 4))
```


    
![png](/images/finance/main/001_mplfinance_files/001_mplfinance_6_0.png)
    



    
![png](/images/finance/main/001_mplfinance_files/001_mplfinance_6_1.png)
    


## 移動平均線
株価や外国為替のテクニカル分析において使用される指標の一つに、移動平均線というものがあります。

> 現在のテクニカル分析では、短期、中期、長期の3本の移動平均線を同時に表示させる例が多く、日足チャートでは、5日移動平均線、25日移動平均線、75日移動平均線がよく使われる。（引用元：[Wikipedia 移動平均線](https://ja.wikipedia.org/wiki/%E7%A7%BB%E5%8B%95%E5%B9%B3%E5%9D%87%E7%B7%9A)）

5/25/75日移動平均線をプロットするために `mav` オプションを指定します。


```python
mpf.plot(df, type="candle", style="starsandstripes", figsize=(12, 4), mav=[5, 25, 75])
```


    
![png](/images/finance/main/001_mplfinance_files/001_mplfinance_8_0.png)
    


## 凡例の表示
どの線が何日移動平均線かわかりにくいので、凡例を追加します。
{{% notice document %}}
[matplotlib.pyplot.legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html)
{{% /notice %}}


```python
import japanize_matplotlib
import matplotlib.patches as mpatches

fig, axes = mpf.plot(
    df,
    type="candle",
    style="starsandstripes",
    figsize=(12, 4),
    mav=[5, 25, 75],
    returnfig=True,
)
fig.legend(
    [f"{days} days" for days in [5, 25, 75]], bbox_to_anchor=(0.0, 0.78, 0.28, 0.102)
)
```




    <matplotlib.legend.Legend at 0x7f8be96e5b80>




    
![png](/images/finance/main/001_mplfinance_files/001_mplfinance_10_1.png)
    


## ボリュームの表示


```python
fig, axes = mpf.plot(
    df,
    title="NVDA 2021/1/B~2021/6/E",
    type="candle",
    style="starsandstripes",
    figsize=(12, 4),
    mav=[5, 25, 75],
    volume=True,
    datetime_format="%Y/%m/%d",
    returnfig=True,
)
fig.legend(
    [f"{days} days" for days in [5, 25, 75]], bbox_to_anchor=(0.0, 0.78, 0.28, 0.102)
)
```




    <matplotlib.legend.Legend at 0x7f8beac867f0>




    
![png](/images/finance/main/001_mplfinance_files/001_mplfinance_12_1.png)
    



```python
fig, axes = mpf.plot(
    df,
    title="NVDA 2021/1/B~2021/6/E",
    type="candle",
    style="yahoo",
    figsize=(12, 4),
    mav=[5, 25, 75],
    volume=True,
    datetime_format="%Y/%m/%d",
    returnfig=True,
)
fig.legend(
    [f"{days} days" for days in [5, 25, 75]], bbox_to_anchor=(0.0, 0.78, 0.28, 0.102)
)
```



    
![png](/images/finance/main/001_mplfinance_files/001_mplfinance_13_1.png)
    

