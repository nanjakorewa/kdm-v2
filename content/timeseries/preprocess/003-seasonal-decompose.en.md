---
title: Trend & Periodicity
weight: 3
pre: "<b>5.1.3 </b>"
searchtitle: "Decompose time series data into trend and seasonality by STL decomposition in python"
---

<div class="pagetop-box">
    <p>Time-series data are sometimes considered to be composed of seasonality, trend components, and residuals, and there are times when we want to decompose time-series data into these three components.
    STL decomposition decomposes data into seasonality, trend components, and residuals based on the specified period length. This method is effective when the length of the period of the seasonal component (waves that occur repeatedly) is constant and the length of the period is known in advance.</p>
</div>

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
```

## Sample data created

The data is a combination of several periodic values, and the trend changes in a segmented manner. Noise is also added by `np.random.rand()`.


```python
date_list = pd.date_range("2021-01-01", periods=365, freq="D")
value_list = [
    10
    + i % 14
    + np.log(1 + i % 28 + np.random.rand())
    + np.sqrt(1 + i % 7 + np.random.rand()) * 2
    + (((i - 100) / 10)) * (i > 100)
    - ((i - 200) / 7) * (i > 200)
    + np.random.rand()
    for i, di in enumerate(date_list)
]

df = pd.DataFrame(
    {
        "日付": date_list,
        "観測値": value_list,
    }
)

df.head(10)
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
      <th>日付</th>
      <th>観測値</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-01</td>
      <td>13.874478</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-02</td>
      <td>15.337944</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-03</td>
      <td>17.166133</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-01-04</td>
      <td>19.561274</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-01-05</td>
      <td>20.426502</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2021-01-06</td>
      <td>22.127725</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2021-01-07</td>
      <td>24.484329</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2021-01-08</td>
      <td>22.313675</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2021-01-09</td>
      <td>23.716303</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2021-01-10</td>
      <td>25.861587</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 5))
sns.lineplot(x=df["日付"], y=df["観測値"])
plt.grid(axis="x")
plt.show()
```


    
![png](/images/timeseries/preprocess/003-seasonal-decompose_files/003-seasonal-decompose_6_0.png)
    


## Decompose into trend, periodicity, and residuals

Time series data using [statsmodels.tsa.seasonal.STL](https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.STL.html)

- Trend (.trend)
- Seasonal/periodic(.seasonal)
- Residual(.resid)

(In the following code, `dr` means [DecomposeResult](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.DecomposeResult.html#statsmodels.tsa.seasonal.DecomposeResult).

{{% notice info %}}
Cleveland, Robert B., et al. "STL: A seasonal-trend decomposition." J. Off. Stat 6.1 (1990): 3-73. ([pdf](http://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf))
{{% /notice %}}



```python
period = 28
stl = STL(df["観測値"], period=period)
dr = stl.fit()
```


```python
_, axes = plt.subplots(figsize=(12, 8), ncols=1, nrows=4, sharex=True)

axes[0].set_title("観測値")
axes[0].plot(dr.observed)
axes[0].grid()

axes[1].set_title("トレンド")
axes[1].plot(dr.trend)
axes[1].grid()

axes[2].set_title("季節性")
axes[2].plot(dr.seasonal)
axes[2].grid()

axes[3].set_title("その他の要因・残差")
axes[3].plot(dr.resid)
axes[3].grid()

plt.tight_layout()
plt.show()
```


    
![png](/images/timeseries/preprocess/003-seasonal-decompose_files/003-seasonal-decompose_9_0.png)
    


### Check for resistance to outliers


```python
def check_outlier():
    period = 28
    df_outlier = df["観測値"].copy()

    for i in range(1, 6):
        df_outlier[i * 50] = df_outlier[i * 50] * 1.4

    stl = STL(df_outlier, period=period, trend=31)
    dr_outlier = stl.fit()

    _, axes = plt.subplots(figsize=(12, 8), ncols=1, nrows=4, sharex=True)

    axes[0].set_title("観測値")
    axes[0].plot(dr_outlier.observed)
    axes[0].grid()

    axes[1].set_title("トレンド")
    axes[1].plot(dr_outlier.trend)
    axes[1].grid()

    axes[2].set_title("季節性")
    axes[2].plot(dr_outlier.seasonal)
    axes[2].grid()

    axes[3].set_title("その他の要因・残差")
    axes[3].plot(dr_outlier.resid)
    axes[3].grid()


check_outlier()
```


    
![png](/images/timeseries/preprocess/003-seasonal-decompose_files/003-seasonal-decompose_11_0.png)
    


## Check to see if the trend is extracted

If the trend is correctly extracted, the trend should not contain a cyclical component and the PACF (Partial Autocorrelation) should be close to zero.

{{% notice document %}}
- [statsmodels.tsa.stattools.acf](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.acf.html)
- [statsmodels.tsa.stattools.pacf](https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.pacf.html)
{{% /notice %}}


```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
plt.suptitle("トレンドの自己相関・偏自己相関")
plot_acf(dr.trend.dropna(), ax=axes[0])
plot_pacf(dr.trend.dropna(), method="ywm", ax=axes[1])
plt.tight_layout()
plt.show()

_, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
plt.suptitle("残差の自己相関・偏自己相関")
plot_acf(dr.resid.dropna(), ax=axes[0])
plot_pacf(dr.resid.dropna(), method="ywm", ax=axes[1])
plt.tight_layout()
plt.show()
```


    
![png](/images/timeseries/preprocess/003-seasonal-decompose_files/003-seasonal-decompose_13_0.png)
    



    
![png](/images/timeseries/preprocess/003-seasonal-decompose_files/003-seasonal-decompose_13_1.png)
    


## Portmanteau test


```python
from statsmodels.stats.diagnostic import acorr_ljungbox

acorr_ljungbox(dr.resid.dropna(), lags=int(np.log(df.shape[0])))
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
      <th>lb_stat</th>
      <th>lb_pvalue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>3.962297</td>
      <td>0.046530</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.823514</td>
      <td>0.089658</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.456753</td>
      <td>0.037457</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.092893</td>
      <td>0.016674</td>
    </tr>
    <tr>
      <th>5</th>
      <td>23.134467</td>
      <td>0.000318</td>
    </tr>
  </tbody>
</table>
</div>

