---
title: データの確認
weight: 1
pre: "<b>5.1.1 </b>"
not_use_colab: true
searchtitle: "pythonで時系列データに対してトレンドの線を引いてみる"
---

{{% youtube "4swXGBqlbP8" %}}

## データの中身を見る

```python
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats
from statsmodels.tsa import stattools
```

## データセットの読み込み


```python
data = pd.read_csv("sample.csv")
data.head(10)
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
      <th>Date</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1981-01-01</td>
      <td>20.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1981-01-02</td>
      <td>17.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1981-01-03</td>
      <td>18.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1981-01-04</td>
      <td>14.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1981-01-05</td>
      <td>15.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1981-01-06</td>
      <td>15.8</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1981-01-07</td>
      <td>15.8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1981-01-08</td>
      <td>17.4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1981-01-09</td>
      <td>21.8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1981-01-10</td>
      <td>20.0</td>
    </tr>
  </tbody>
</table>
</div>



## タイムスタンプをdatetimeにする

Date列は現在Object型、つまり文字列として読み込まれています。これをタイムスタンプとして扱うため、
[datetime --- 基本的な日付型および時間型](https://docs.python.org/ja/3/library/datetime.html)を確認しつつ、datetime型に変換します。


```python
data["Date"] = data["Date"].apply(
    lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d")
)

print(f"Date列のdtype: {data['Date'].dtype}")
```

    Date列のdtype: datetime64[ns]


## 時系列の概要を確認する

### pandas.DataFrame.describe
はじめに、データがどのようなものであるか簡単に確認します。
[pandas.DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html)を使って、Temp列の簡単な統計量を確認します。


```python
data.describe()
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
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3650.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>11.177753</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.071837</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>14.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>26.300000</td>
    </tr>
  </tbody>
</table>
</div>



### 折れ線グラフ
[seaborn.lineplot](https://seaborn.pydata.org/generated/seaborn.lineplot.html)を使って、どのような周期になっているか確認します。


```python
plt.figure(figsize=(12, 6))
sns.lineplot(x=data["Date"], y=data["Temp"])
plt.ylabel("Temp")
plt.grid(axis="x")
plt.grid(axis="y", color="r", alpha=0.3)
plt.show()
```


    
![png](/images/timeseries/preprocess/001-check-data_files/001-check-data_11_0.png)
    


### ヒストグラム


```python
plt.figure(figsize=(12, 6))
plt.hist(x=data["Temp"], rwidth=0.8)
plt.xlabel("Temp")
plt.ylabel("日数")
plt.grid(axis="y")
plt.show()
```


    
![png](/images/timeseries/preprocess/001-check-data_files/001-check-data_13_0.png)
    


### 自己相関とコレログラム
[pandas.plotting.autocorrelation_plot](https://pandas.pydata.org/docs/reference/api/pandas.plotting.autocorrelation_plot.html)を用いて自己相関を確認し、時系列データの周期性をチェックします。

> 大雑把に言うと、自己相関とは、信号がそれ自身を時間シフトした信号とどれくらい一致するかを測る尺度であり、時間シフトの大きさの関数として表される。　引用元：[wikipedia - 自己相関](https://ja.wikipedia.org/wiki/%E8%87%AA%E5%B7%B1%E7%9B%B8%E9%96%A2#:~:text=%E8%87%AA%E5%B7%B1%E7%9B%B8%E9%96%A2%EF%BC%88%E3%81%98%E3%81%93%E3%81%9D%E3%81%86%E3%81%8B,%E9%96%A2%E6%95%B0%E3%81%A8%E3%81%97%E3%81%A6%E8%A1%A8%E3%81%95%E3%82%8C%E3%82%8B%E3%80%82)


```python
plt.figure(figsize=(12, 6))
pd.plotting.autocorrelation_plot(data["Temp"])
plt.grid()
plt.axvline(x=365)
plt.xlabel("ラグ")
plt.ylabel("自己相関")
plt.show()
```


    
![png](/images/timeseries/preprocess/001-check-data_files/001-check-data_15_0.png)
    


### 単位根検定
データが単位根過程であるかどうかを確認します。
単位根過程であることを帰無仮説とする検定（Augmented Dickey-Fuller test）を行います。

[statsmodels.tsa.stattools.adfuller](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html)


```python
stattools.adfuller(data["Temp"], autolag="AIC")
```




    (-4.444804924611697,
     0.00024708263003610177,
     20,
     3629,
     {'1%': -3.4321532327220154,
      '5%': -2.862336767636517,
      '10%': -2.56719413172842},
     16642.822304301197)



### トレンドの確認
1次元多項式を時系列にフィットさせてトレンドの線を引きます。今回のデータはほとんどトレンド定常に近いので、トレンドはほとんどありません。

[numpy.poly1d — NumPy v1.22 Manual](https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html)


```python
def get_trend(timeseries, deg=3):
    """時系列データのトレンドの線を作成する

    Args:
        timeseries(pd.Series) : 時系列データ。

    Returns:
        pd.Series: トレンドに相当する時系列データ。
    """
    x = list(range(len(timeseries)))
    y = timeseries.values
    coef = np.polyfit(x, y, deg)
    trend = np.poly1d(coef)(x)
    return pd.Series(data=trend, index=timeseries.index)

data["Trend"] = get_trend(data["Temp"])

# グラフをプロット
plt.figure(figsize=(12, 6))
sns.lineplot(x=data["Date"], y=data["Temp"], alpha=0.5, label="Temp")
sns.lineplot(x=data["Date"], y=data["Trend"], label="トレンド")
plt.grid(axis="x")
plt.legend()
plt.show()
```


    
![png](/images/timeseries/preprocess/001-check-data_files/001-check-data_19_0.png)
    


#### 補足：トレンドがはっきりとある場合

緑色がトレンドとなる線です。

```python
data_sub = data.copy()
data_sub["Temp"] = (
    data_sub["Temp"] + np.log(data_sub["Date"].dt.year - 1980) * 10
)  # ダミーのトレンド
data_sub["Trend"] = get_trend(data_sub["Temp"])

# グラフをプロット
plt.figure(figsize=(12, 6))
sns.lineplot(x=data_sub["Date"], y=data_sub["Temp"], alpha=0.5, label="Temp")
sns.lineplot(x=data_sub["Date"], y=data_sub["Trend"], label="トレンド")
plt.grid(axis="x")
plt.legend()
plt.show()
```


    
![png](/images/timeseries/preprocess/001-check-data_files/001-check-data_21_0.png)
    