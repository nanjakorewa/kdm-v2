---
title: 調整
weight: 6
pre: "<b>5.1.5 </b>"
searchtitle: "CPIを用いて時系列データをインフレ調整する"
---

貨幣価値の影響を受けるデータは、分析前に調整する必要がある時もあります。
ここでは消費者物価指数 (CPI)を用いて時系列データに対してインフレ調整をしてみます。具体的には

- FREDから消費者物価指数 (CPI)のデータを、pythonのAPIを用いて取得
- アメリカの世帯収入の中央値データを取得
- アメリカの世帯収入の中央値データをCPIを用いて調整する
- アメリカの世帯収入の中央値データ(調整済み)のデータと、CPIを用いて調整したデータを比較

してみます。

{{% notice document %}}
[A Python library that quickly adjusts U.S. dollars for inflation using the Consumer Price Index (CPI).](https://palewi.re/docs/cpi/)
{{% /notice %}}



```python
import os
import cpi
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from full_fred.fred import Fred
from datetime import date


cpi.update()  # 必ず実行する
```

## FREDからデータを取得する

[FRED](https://fred.stlouisfed.org/)から必要なデータを取得します。データの取得方法については別動画で説明していますが、pythonからアクセスするためにはAPIキーの発行が必要です。

{{% youtube "OjlMTN4Uq0k" %}}


```python
# FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred()
print(f"FRED APIキーが環境変数に設定されている：{fred.env_api_key_found()}")


def get_fred_data(name, start="2013-01-01", end=""):
    df = fred.get_series_df(name)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")

    if end == "":
        df = df.loc[f"{start}":]
    else:
        df = df.loc[f"{start}":f"{end}"]

    return df
```

    FRED APIキーが環境変数に設定されている：True


## Household Income in the United States（世帯収入データ）
[Median Household Income in the United States (MEHOINUSA646N)](https://fred.stlouisfed.org/series/MEHOINUSA646N)のデータを確認してみます。このデータは、データの単位が

> Units:  Current Dollars, Not Seasonally Adjusted

となっており、このデータはインフレ調整・季節調整がされていません。
この家計データをCPIを用いて調整してみます。


```python
data = get_fred_data("MEHOINUSA646N", start="2013-01-01", end="")
data.head()
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
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>53585</td>
    </tr>
    <tr>
      <th>2014-01-01</th>
      <td>53657</td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>56516</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>59039</td>
    </tr>
    <tr>
      <th>2017-01-01</th>
      <td>61136</td>
    </tr>
  </tbody>
</table>
</div>



### CPIを用いて値を調整する


```python
data["adjusted"] = [
    cpi.inflate(dollers.value, date.year) for date, dollers in data.iterrows()
]
```

### 調整前と調整後の比較


```python
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=data,
    x="date",
    y="value",
    label="Current Dollars, Not Seasonally Adjusted(MEHOINUSA646N)",
)
sns.lineplot(data=data, x="date", y="adjusted", label="MEHOINUSA646N adjusted")
plt.legend()
```




    <matplotlib.legend.Legend at 0x28d341a5e80>




    
![png](/images/timeseries/preprocess/005-inflation-adjustment_files/005-inflation-adjustment_9_1.png)
    


## Real Median Household Income in the United States
調整済みのデータは[Real Median Household Income in the United States (MEHOINUSA672N)](https://fred.stlouisfed.org/series/MEHOINUSA672N)にて提供されています。先ほどインフレ調整したデータ（`data`の中にある`adjusted`列の値）とMEHOINUSA672Nの値を比較してみます。調整前の値を消費者物価指数 (CPI)を用いて調整したものは、調整済みの値（MEHOINUSA672N）とほとんど一致するのが期待値です。


```python
data_real = get_fred_data("MEHOINUSA672N", start="2013-01-01", end="")
data_real.head()
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
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>62425</td>
    </tr>
    <tr>
      <th>2014-01-01</th>
      <td>61468</td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>64631</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>66657</td>
    </tr>
    <tr>
      <th>2017-01-01</th>
      <td>67571</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_real, x="date", y="value", label="MEHOINUSA672N")
sns.lineplot(data=data, x="date", y="adjusted", label="MEHOINUSA646N adjusted")

# 確認のため、プロットした値をテキストで表示しています
for t, v in data_real.iterrows():
    plt.text(t, v[0] - 500, f"{v[0]:.2f}")

for t, v in data.iterrows():
    plt.text(t, v[1] + 500, f"{v[1]:.2f}")

plt.legend()
```




    <matplotlib.legend.Legend at 0x28d347968b0>




    
![png](/images/timeseries/preprocess/005-inflation-adjustment_files/005-inflation-adjustment_12_1.png)
    

