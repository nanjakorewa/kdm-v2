---
title: "ジグザグ"
pre: "7.2.4 "
weight: 4
searchtitle: "pythonでZigZagをプロットする"
---


## ZigZagをプロットする

Zigzagを使用してジグザグをプロットしてみます。
株価のデータとして[TORM plc (TRMD)](https://finance.yahoo.com/quote/TRMD/history?period1=1676641453&period2=1708177453&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true)のデータを使用しています。

{{% notice document %}}
[ジグザグの見方・使い方](https://www.sevendata.co.jp/shihyou/technical/zig.html)
{{% /notice %}}


{{% notice document %}}
[https://github.com/jbn/ZigZag](https://github.com/jbn/ZigZag)
{{% /notice %}}


```python
import mplfinance as mpf
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
from zigzag import peak_valley_pivots

# TORM plc (TRMD)
trmd = pd.read_csv("TRMD.csv", index_col="Date")
trmd.index = pd.to_datetime(trmd.index)
trmd.head(5)
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
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
      <th>2023-02-17</th>
      <td>32.080002</td>
      <td>32.669998</td>
      <td>31.760000</td>
      <td>32.060001</td>
      <td>25.389601</td>
      <td>206600</td>
    </tr>
    <tr>
      <th>2023-02-21</th>
      <td>32.490002</td>
      <td>33.250000</td>
      <td>31.840000</td>
      <td>32.119999</td>
      <td>25.437117</td>
      <td>240900</td>
    </tr>
    <tr>
      <th>2023-02-22</th>
      <td>32.709999</td>
      <td>32.759998</td>
      <td>31.110001</td>
      <td>31.450001</td>
      <td>24.906519</td>
      <td>340400</td>
    </tr>
    <tr>
      <th>2023-02-23</th>
      <td>33.380001</td>
      <td>34.639999</td>
      <td>33.330002</td>
      <td>34.299999</td>
      <td>27.163548</td>
      <td>496600</td>
    </tr>
    <tr>
      <th>2023-02-24</th>
      <td>34.889999</td>
      <td>34.965000</td>
      <td>34.290001</td>
      <td>34.689999</td>
      <td>27.472404</td>
      <td>338000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def plot_zigzag(ax, X, p):
    """zigzagをプロット
    args:
        ax: axis
        X: pandas.Series
        p: pivots index
    """
    ax.plot(np.arange(len(X))[p != 0], X[p != 0], "g-", linewidth=1)
    return ax


fig, axes = mpf.plot(
    trmd, type="candle", style="starsandstripes", figsize=(12, 4), returnfig=True
)

pivots = peak_valley_pivots(trmd["Close"], 0.03, -0.03)
axes[0] = plot_zigzag(axes[0], trmd["Close"], pivots)
```


    
![png](/images/finance/visualize/ZigZag_files/ZigZag_2_0.png)
    

