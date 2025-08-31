---
title: MSTL分解
weight: 4
pre: "<b>5.1.3.2 </b>"
searchtitle: "pythonにて時系列データをMSTL分解でトレンド・季節性に分解する"
---


## 複数の周期性が重なったデータでをトレンド・季節/周期性・残差に分解する

MSTLはSTL分解(LOESSによる季節・トレンド分解)手法を拡張したもので、複数の季節パターンを持つ時系列の分解が可能です。MSTLはstatsmodelの `version==0.14.0` 以降でのみ使用可能です。詳細は[ドキュメント](https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.MSTL.html)をご確認ください。


{{% notice ref %}}
K. Bandura, R.J. Hyndman, and C. Bergmeir (2021) MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns. arXiv preprint arXiv:2107.13462.
{{% /notice %}}


{{% notice tip %}}
Anacondaを仮想環境として使用している場合、`conda install -c conda-forge statsmodels`でインストールされるものは`0.13.X`となっています(2022/11/1時点)。その場合、作業中の仮想環境の中で以下のコマンドを使用して最近のバージョンをインストールしてください。

```
pip install git+https://github.com/statsmodels/statsmodels
```
{{% /notice %}}


```python
import japanize_matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from statsmodels.tsa.seasonal import MSTL
```

## サンプルデータを作成
周期的な数値を複数組合せ、さらに区分的にトレンドが変化しています。また `np.random.rand()` でノイズも乗せています。


```python
date_list = pd.date_range("2018-01-01", periods=1000, freq="D")
value_list = [
    10
    + i % 14
    + 2 * np.sin(10 * np.pi * i / 24)
    + 5 * np.cos(2 * np.pi * i / (24 * 7)) * 2
    + np.log(i**3 + 1)
    + np.sqrt(i)
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
<table border="1" class="dataframe">
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
      <td>2018-01-01</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-02</td>
      <td>24.618006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-03</td>
      <td>26.583476</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-04</td>
      <td>26.587164</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-05</td>
      <td>28.330645</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2018-01-06</td>
      <td>32.415653</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2018-01-07</td>
      <td>35.578666</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2018-01-08</td>
      <td>35.663289</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2018-01-09</td>
      <td>34.892380</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2018-01-10</td>
      <td>36.617664</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(15, 5))
sns.lineplot(x=df["日付"], y=df["観測値"])
plt.grid(axis="x")
plt.show()
```


    
![png](/images/timeseries/preprocess/003-seasonal-decompose-mstl_files/003-seasonal-decompose-mstl_5_0.png)
    


# トレンド・季節/周期性・残差に分解する


```python
periods = (24, 24 * 7)
mstl = MSTL(df["観測値"], periods=periods).fit()
```

- トレンド(.trend)
- 季節/周期性(.seasonal)
- 残差(.resid)

をそれぞれプロットしてみます。今回は二つの周期の異なる三角関数を足しているので `.seasonal` には二つの列が含まれています。
残差のプロットにところどころ山があるものの、ほとんどの領域で残差が0に近い（＝きれいに分解できている）ことが確認できます。


```python
_, axes = plt.subplots(figsize=(12, 8), ncols=1, nrows=4, sharex=True)

axes[0].set_title("観測値")
axes[0].plot(mstl.observed)
axes[0].grid()

axes[1].set_title("トレンド")
axes[1].plot(mstl.trend)
axes[1].grid()

axes[2].set_title("季節性")
axes[2].plot(mstl.seasonal.iloc[:, 0], label="周期的な要素１")
axes[2].plot(mstl.seasonal.iloc[:, 1], label="周期的な要素２")
axes[2].legend()
axes[2].grid()

axes[3].set_title("その他の要因・残差")
axes[3].plot(mstl.resid)
axes[3].grid()

plt.tight_layout()
plt.show()
```


    
![png](/images/timeseries/preprocess/003-seasonal-decompose-mstl_files/003-seasonal-decompose-mstl_9_0.png)
    

