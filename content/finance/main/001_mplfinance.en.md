---
title: "mplfinance"
pre: "7.1.2 "
weight: 2
not_use_colab: true
searchtitle: "Using mplfinance to load stock price data into the python environment"
---


<div class="pagetop-box">
<a href="https://github.com/matplotlib/mplfinance">mplfinance</a>とはファイナンスデータを可視化するためのmatplotlibの拡張です。今回はこれを使って、金融データからグラフを作ってみようと思います。基本的な使い方は<a href="https://github.com/matplotlib/mplfinance#tutorials">mplfinanceのリポジトリの『Tutorial』</a>にて説明されており、これをみながらグラフを作ってみます。

<a href="https://github.com/matplotlib/mplfinance">mplfinance</a> is an extension to matplotlib for visualizing finance data. In this article, we will try to create a graph from financial data using mplfinance. The basic usage is explained in the "Tutorial" in the mplfinance repository.
</div>



## Loading Data

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
    """Retrieves data recording stock prices

    Args:
        ticker_symbol (str): Description of param1
        start (str): Date of beginning of period, optional.
        end (str): Date of end of period, optional.

    Returns:
        res: Stock Price Data

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
# ticker symbol, period, and destination file
ticker_symbol = "NVDA"
start = "2021-01-01"
end = "2021-06-30"

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



## Plotting OHLC

OHLC is the opening price, high, low, and close, and is one of the graphs we usually look at most often.
Let's plot it as a candlestick and as a line.

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
    


## Moving Average

One of the indicators used in technical analysis of stock prices and foreign exchange is the moving average.
In current technical analysis, there are many examples where three moving averages (short, medium, and long term) are displayed simultaneously. In daily charts, the 5-day, 25-day, and 75-day moving averages are often used.

Specify the `mav` option to plot the 5/25/75 day moving average.


```python
mpf.plot(df, type="candle", style="starsandstripes", figsize=(12, 4), mav=[5, 25, 75])
```


    
![png](/images/finance/main/001_mplfinance_files/001_mplfinance_8_0.png)
    


## Show legend

Add a legend because it is difficult to tell which line represents a moving average over what number of days.

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
    


## Displaying Volume


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
    

