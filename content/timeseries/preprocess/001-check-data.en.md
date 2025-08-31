---
title: Check Dataset
weight: 1
pre: "<b>5.1.1 </b>"
not_use_colab: true
searchtitle: "Drawing a trend line for time series data in python"
---

## See what's in the data

```python
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy import stats
from statsmodels.tsa import stattools
```

## Reading a dataset from a csv file


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



## Set timestamp to datetime

The Date column is currently read as an Object type, i.e., a string. To treat it as a timestamp, use the following
[datetime --- Basic Date and Time Types](https://docs.python.org/3/library/datetime.html) to convert it to a datetime type.


```python
data["Date"] = data["Date"].apply(
    lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%d")
)

print(f"Date column dtype: {data['Date'].dtype}")
```

   Date column dtype: datetime64[ns]


## Get an overview of a time series

### pandas.DataFrame.describe

To begin, we briefly review what the data looks like.
We will use [pandas.DataFrame.describe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) to check some simple statistics for the Temp column.


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



### Line graph

Use [seaborn.lineplot](https://seaborn.pydata.org/generated/seaborn.lineplot.html) to see what the cycle looks like.


```python
plt.figure(figsize=(12, 6))
sns.lineplot(x=data["Date"], y=data["Temp"])
plt.ylabel("Temp")
plt.grid(axis="x")
plt.grid(axis="y", color="r", alpha=0.3)
plt.show()
```


    
![png](/images/timeseries/preprocess/001-check-data_files/001-check-data_11_0.png)
    


### Histogram


```python
plt.figure(figsize=(12, 6))
plt.hist(x=data["Temp"], rwidth=0.8)
plt.xlabel("Temp")
plt.ylabel("日数")
plt.grid(axis="y")
plt.show()
```


    
![png](/images/timeseries/preprocess/001-check-data_files/001-check-data_13_0.png)
    


### Autocorrelation and Cholerograms

Using [pandas.plotting.autocorrelation_plot](https://pandas.pydata.org/docs/reference/api/pandas.plotting.autocorrelation_plot.html) Check autocorrelation to check the periodicity of time series data.
Roughly speaking, autocorrelation is a measure of how well a signal matches a time-shifted signal of itself, expressed as a function of the magnitude of the time shift.


```python
plt.figure(figsize=(12, 6))
pd.plotting.autocorrelation_plot(data["Temp"])
plt.grid()
plt.axvline(x=365)
plt.xlabel("lag")
plt.ylabel("autocorrelation")
plt.show()
```


    
![png](/images/timeseries/preprocess/001-check-data_files/001-check-data_15_0.png)
    


### Unit Root Test

We check to see if the data are a unit root process.
The Augmented Dickey-Fuller test is used to test the null hypothesis of a unit root process.

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



### Checking the trend

The trend line is drawn by fitting a one-dimensional polynomial to the time series. Since the data in this case is almost trend-stationary, there is almost no trend.

[numpy.poly1d — NumPy v1.22 Manual](https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html)


```python
def get_trend(timeseries, deg=3):
    """Create a trend line for time-series data

    Args:
        timeseries(pd.Series) : time-series data

    Returns:
        pd.Series: trend line
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
    


#### Supplement: If there is a clear trend

The green line is the trend line.

```python
data_sub = data.copy()
data_sub["Temp"] = (
    data_sub["Temp"] + np.log(data_sub["Date"].dt.year - 1980) * 10
)  # Dummy Trends
data_sub["Trend"] = get_trend(data_sub["Temp"])

plt.figure(figsize=(12, 6))
sns.lineplot(x=data_sub["Date"], y=data_sub["Temp"], alpha=0.5, label="Temp")
sns.lineplot(x=data_sub["Date"], y=data_sub["Trend"], label="トレンド")
plt.grid(axis="x")
plt.legend()
plt.show()
```


    
![png](/images/timeseries/preprocess/001-check-data_files/001-check-data_21_0.png)
    