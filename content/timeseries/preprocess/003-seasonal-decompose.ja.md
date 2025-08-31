---
title: STL分解
weight: 3
pre: "<b>5.1.3 </b>"
searchtitle: "pythonにて時系列データをSTL分解でトレンド・季節性に分解する"
---

{{% youtube "1Lr4k6QrKXw" %}}

<div class="pagetop-box">
    <p>時系列データは季節性・トレンド成分・残差で構成されるというとらえ方があり、時系列データをこの３つの要素に分解したい時があります。
    STL分解は指定された周期の長さに基づいてデータを季節性・トレンド成分・残差に分解します。季節成分（繰り返し発生する波）の周期の長さが一定で、かつ事前に周期の長さがわかっている場合に有効な手法です。</p>
</div>


```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
```

## サンプルデータを作成

周期的な数値を複数組合せ、さらに区分的にトレンドが変化しています。また `np.random.rand()` でノイズも乗せています。


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
    


## トレンド・周期性・残差に分解する

[statsmodels.tsa.seasonal.STL](https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.STL.html)を用いて時系列データを

- トレンド(.trend)
- 季節/周期性(.seasonal)
- 残差(.resid)

に分解してみます。（以下のコード中のdrは[DecomposeResult](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.DecomposeResult.html#statsmodels.tsa.seasonal.DecomposeResult)を指しています。）

{{% notice info %}}
参考文献：Cleveland, Robert B., et al. "STL: A seasonal-trend decomposition." J. Off. Stat 6.1 (1990): 3-73. ([pdf](http://www.nniiem.ru/file/news/2016/stl-statistical-model.pdf))
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
    


### 外れ値に強いか確認


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
    


## トレンドを捉えることができているか確認する

もしもトレンドが正しく抽出できている場合は、トレンドに周期的な要素が含まれていないはずであり、PACF（偏自己相関）が０に近くなるはずです。


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
    


## 残差に対するかばん検定


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



## トレンドを近似する

区分的な線形関数で近似してみます。
コードは[ruoyu0088/segments_fit.ipynb](https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a)のものを使用しています。


```python
plt.figure(figsize=(12, 4))
trend_data = dr.trend.dropna()
plt.plot(trend_data)
plt.show()
```


    
![png](/images/timeseries/preprocess/003-seasonal-decompose_files/003-seasonal-decompose_17_0.png)
    


### 区分線形関数
[ruoyu0088/segments_fit.ipynb](https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a)のコードを元にしています。

{{% notice document %}}
- [numpy.percentile](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html)
- [scipy.special.huber](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.huber.html)
- [ruoyu0088/segments_fit.ipynb](https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a)
{{% /notice %}}

```python
from scipy import optimize
from scipy.special import huber


def segments_fit(X, Y, count, is_use_huber=True, delta=0.5):
    """
    original: https://gist.github.com/ruoyu0088/70effade57483355bbd18b31dc370f2a
    ruoyu0088氏作成の ruoyu0088/segments_fit.ipynb　より引用しています
    """
    xmin = X.min()
    xmax = X.max()
    seg = np.full(count - 1, (xmax - xmin) / count)
    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array(
        [Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init]
    )

    def func(p):
        seg = p[: count - 1]
        py = p[count - 1 :]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        if is_use_huber:
            return np.mean(huber(delta, Y - Y2))
        else:
            return np.mean((Y - Y2) ** 2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method="Nelder-Mead")
    return func(r.x)
```


```python
x = np.arange(0, len(trend_data))
y = trend_data

plt.figure(figsize=(12, 4))
px, py = segments_fit(x, y, 4)
plt.plot(x, y, "-x", label="トレンド")
plt.plot(px, py, "-o", label="近似した線形関数")
plt.ylim(20, 40)
plt.legend()
```



    
![png](/images/timeseries/preprocess/003-seasonal-decompose_files/003-seasonal-decompose_20_1.png)
    


## 元データからトレンドを除去する
### トレンドのみのデータを用意する


```python
trend_data_first, trend_data_last = trend_data.iloc[0], trend_data.iloc[-1]

for i in range(int(period / 2)):
    trend_data[trend_data.index.min() - 1] = trend_data_first
    trend_data[trend_data.index.max() + 1] = trend_data_last

trend_data.sort_index()
```




    -14     23.728925
    -13     23.728925
    -12     23.728925
    -11     23.728925
    -10     23.728925
              ...    
     374    26.206443
     375    26.206443
     376    26.206443
     377    26.206443
     378    26.206443
    Name: trend, Length: 393, dtype: float64



### トレンドを取り除いた波形を確認する


```python
plt.figure(figsize=(12, 4))
plt.title("観測値")
plt.plot(dr.observed, label="トレンドあり")
plt.plot(dr.observed - trend_data, label="トレンド除去済み")
plt.grid()
```


    
![png](/images/timeseries/preprocess/003-seasonal-decompose_files/003-seasonal-decompose_24_0.png)
    

