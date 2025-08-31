---
title: Isolation Forest
weight: 2
chapter: true
not_use_colab: false
not_use_twitter: false
pre: "<b>5.8.2 </b>"
---


{{% youtube "9ZYtTL4tFHY" %}}

## Isolation Forest

isolation Forestを使って異常値を検出してみます。
このページでは時系列データの中にいくつか異常値を混ぜ、それが検出できるかどうか検証します。
また、最後に異常値と判定したルールを可視化します。


```python
import japanize_matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.tree import plot_tree
```

### 人工データの作成
`[11, 49, 149, 240, 300, 310]`の日付にて異常値を混ぜておきます。
カテゴリ変数や整数の特徴も含まれるデータです。数値のみプロットしてみます。


```python
rs = np.random.RandomState(365)
dates = pd.date_range("1 1 2016", periods=365, freq="D")

data = {}
data["月"] = [d.strftime("%m") for d in dates]
data["曜日"] = [d.strftime("%A") for d in dates]
data["特徴1"] = [np.sin(d.day / 50) + np.random.rand() for d in dates]
data["特徴2"] = [np.cos(d.day / 50) + np.random.rand() for d in dates]
data["特徴3"] = [3 * np.random.rand() + np.log(d.dayofyear) * 0.03 for d in dates]
data["特徴4"] = [np.random.choice(["☀", "☂", "☁"]) for d in dates]
column_names = list(data.keys())


anomaly_index = [11, 49, 149, 240, 300, 310]
anomaly_dates = [dates[i] for i in anomaly_index]
for i in anomaly_index:
    data["特徴1"][i] = 2.5
    data["特徴2"][i] = 0.3


data = pd.DataFrame(data, index=dates)

plt.figure(figsize=(10, 4))
sns.lineplot(data=data, palette="tab10", linewidth=2.5)
```




    <Axes: >




    
![png](/images/timeseries/unsupervised/isolation-forest_files/isolation-forest_3_1.png)
    



```python
data
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
      <th>月</th>
      <th>曜日</th>
      <th>特徴1</th>
      <th>特徴2</th>
      <th>特徴3</th>
      <th>特徴4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01</th>
      <td>01</td>
      <td>Friday</td>
      <td>0.432678</td>
      <td>1.645027</td>
      <td>1.118289</td>
      <td>☀</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>01</td>
      <td>Saturday</td>
      <td>0.099463</td>
      <td>1.645734</td>
      <td>0.289383</td>
      <td>☀</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>01</td>
      <td>Sunday</td>
      <td>0.297490</td>
      <td>1.613958</td>
      <td>2.499115</td>
      <td>☀</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>01</td>
      <td>Monday</td>
      <td>1.043077</td>
      <td>1.029947</td>
      <td>0.484240</td>
      <td>☀</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>01</td>
      <td>Tuesday</td>
      <td>0.184797</td>
      <td>1.355265</td>
      <td>2.795718</td>
      <td>☁</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-12-26</th>
      <td>12</td>
      <td>Monday</td>
      <td>0.914207</td>
      <td>1.580330</td>
      <td>0.617349</td>
      <td>☂</td>
    </tr>
    <tr>
      <th>2016-12-27</th>
      <td>12</td>
      <td>Tuesday</td>
      <td>1.229753</td>
      <td>1.599599</td>
      <td>2.605319</td>
      <td>☀</td>
    </tr>
    <tr>
      <th>2016-12-28</th>
      <td>12</td>
      <td>Wednesday</td>
      <td>0.936100</td>
      <td>1.408378</td>
      <td>2.540507</td>
      <td>☀</td>
    </tr>
    <tr>
      <th>2016-12-29</th>
      <td>12</td>
      <td>Thursday</td>
      <td>1.156575</td>
      <td>1.428601</td>
      <td>0.831889</td>
      <td>☀</td>
    </tr>
    <tr>
      <th>2016-12-30</th>
      <td>12</td>
      <td>Friday</td>
      <td>1.383185</td>
      <td>1.544629</td>
      <td>1.416885</td>
      <td>☁</td>
    </tr>
  </tbody>
</table>
<p>365 rows × 6 columns</p>
</div>



### カテゴリ変数の変換
「曜日」のような特徴をIsolation Forestで扱うためにダミー変数に変換します。


```python
X = pd.get_dummies(data)
X
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
<table border="1" class="dataframe" style="font-size:0.5em">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>特徴1</th>
      <th>特徴2</th>
      <th>特徴3</th>
      <th>月_01</th>
      <th>月_02</th>
      <th>...</th>
      <th>曜日_Thursday</th>
      <th>曜日_Tuesday</th>
      <th>曜日_Wednesday</th>
      <th>特徴4_☀</th>
      <th>特徴4_☁</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-01</th>
      <td>0.432678</td>
      <td>1.645027</td>
      <td>1.118289</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-01-02</th>
      <td>0.099463</td>
      <td>1.645734</td>
      <td>0.289383</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-01-03</th>
      <td>0.297490</td>
      <td>1.613958</td>
      <td>2.499115</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-01-04</th>
      <td>1.043077</td>
      <td>1.029947</td>
      <td>0.484240</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>0.184797</td>
      <td>1.355265</td>
      <td>2.795718</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-12-26</th>
      <td>0.914207</td>
      <td>1.580330</td>
      <td>0.617349</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-12-27</th>
      <td>1.229753</td>
      <td>1.599599</td>
      <td>2.605319</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-12-28</th>
      <td>0.936100</td>
      <td>1.408378</td>
      <td>2.540507</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-12-29</th>
      <td>1.156575</td>
      <td>1.428601</td>
      <td>0.831889</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2016-12-30</th>
      <td>1.383185</td>
      <td>1.544629</td>
      <td>1.416885</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
    </tr>
  </tbody>
</table>
<p>365 rows × 25 columns</p>
</div>



### Isolation Forestの作成


```python
ilf = IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination=0.01,
    max_features=5,
    bootstrap=False,
    random_state=np.random.RandomState(365),
)

ilf.fit(X)

data["is_anomaly"] = ilf.predict(X) < 0
data["anomaly_score"] = ilf.decision_function(X)
```

### 検出した日付と正解の比較
異常値として検出したタイミングと正解を比較します。


```python
plt.figure(figsize=(10, 4))
plt.title("検出箇所")
sns.lineplot(data=data[column_names], palette="tab10", linewidth=2.5)
for d in data[data["is_anomaly"]].index:
    plt.axvline(x=d, color="red", linewidth=4)


plt.figure(figsize=(10, 4))
plt.title("正解")
sns.lineplot(data=data[column_names], palette="tab10", linewidth=2.5)
for d in anomaly_dates:
    plt.axvline(x=d, color="black", linewidth=4)
```


    
![png](/images/timeseries/unsupervised/isolation-forest_files/isolation-forest_10_0.png)
    



    
![png](/images/timeseries/unsupervised/isolation-forest_files/isolation-forest_10_1.png)
    


### 異常値のルール

サンプル数（samples）が１の分岐が一番右にあり、それは特徴１による分岐だと分かります。
実際、今回の異常値は特徴1が大きすぎる値のときに異常値になりやすいです。


```python
plt.figure(figsize=(24, 8))
plot_tree(
    ilf.estimators_[0],
    feature_names=column_names,
    filled=True,
    fontsize=13,
    max_depth=3,
    precision=2,
    rounded=True,
)

plt.show()
```


    
![png](/images/timeseries/unsupervised/isolation-forest_files/isolation-forest_12_0.png)
    

