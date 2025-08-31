---
title: "数値に対する演算"
pre: "3.2.3 "
weight: 3
title_replace: "pandas.DataFrameで平均・中央値などを一括で求める"
---

<div class="pagetop-box">
    <p>数値として読み取れる列を選択に対しては pandas.DataFrame.agg を使うことで平均・中央値・自分で定義した演算を一括で実行することができます。使用しているデータセットについては『<a href="../sample-data/">総務省のデータ</a>』の項を参照してください。</p>
</div>


## データを読み込む


```python
import pandas as pd

df = pd.read_csv("./z-001.csv")
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
      <th>Unnamed: 0</th>
      <th>Unnamed: 1</th>
      <th>平成21年度</th>
      <th>平成22年度</th>
      <th>平成23年度</th>
      <th>平成24年度</th>
      <th>平成25年度</th>
      <th>平成26年度</th>
      <th>平成27年度</th>
      <th>平成28年度</th>
      <th>平成29年度</th>
      <th>平成30年度</th>
      <th>令和元年度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>社会保障関係費</td>
      <td>29.8</td>
      <td>31.2</td>
      <td>32.7</td>
      <td>32.7</td>
      <td>31.8</td>
      <td>32.8</td>
      <td>33.7</td>
      <td>34.4</td>
      <td>34.6</td>
      <td>34.4</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>公債費</td>
      <td>18.9</td>
      <td>20.3</td>
      <td>19.8</td>
      <td>20.9</td>
      <td>20.9</td>
      <td>21.4</td>
      <td>21.3</td>
      <td>20.6</td>
      <td>20.9</td>
      <td>20.6</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>機関費</td>
      <td>11.9</td>
      <td>12.5</td>
      <td>11.7</td>
      <td>10.6</td>
      <td>11.8</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.4</td>
      <td>11.5</td>
      <td>11.9</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>教育費</td>
      <td>11.7</td>
      <td>11.9</td>
      <td>11.7</td>
      <td>12.1</td>
      <td>11.6</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.8</td>
      <td>11.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>国土保全及び開発費</td>
      <td>12.0</td>
      <td>11.3</td>
      <td>10.6</td>
      <td>10.2</td>
      <td>10.9</td>
      <td>10.5</td>
      <td>9.9</td>
      <td>10.4</td>
      <td>10.5</td>
      <td>10.5</td>
      <td>10.8</td>
    </tr>
  </tbody>
</table>
</div>



## カラム名を変更する
{{% notice document %}}
[pandas.Series.str.replace](https://pandas.pydata.org/docs/reference/api/pandas.Series.str.replace.html)
{{% /notice %}}

`Unnamed:XX` がわかりにくいので、これらの文字列を置換します。


```python
repl = lambda x: x.group(0)[::-1]
df.columns = df.columns.str.replace("Unnamed:.", "列名", regex=True)
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
      <th>列名0</th>
      <th>列名1</th>
      <th>平成21年度</th>
      <th>平成22年度</th>
      <th>平成23年度</th>
      <th>平成24年度</th>
      <th>平成25年度</th>
      <th>平成26年度</th>
      <th>平成27年度</th>
      <th>平成28年度</th>
      <th>平成29年度</th>
      <th>平成30年度</th>
      <th>令和元年度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>社会保障関係費</td>
      <td>29.8</td>
      <td>31.2</td>
      <td>32.7</td>
      <td>32.7</td>
      <td>31.8</td>
      <td>32.8</td>
      <td>33.7</td>
      <td>34.4</td>
      <td>34.6</td>
      <td>34.4</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>公債費</td>
      <td>18.9</td>
      <td>20.3</td>
      <td>19.8</td>
      <td>20.9</td>
      <td>20.9</td>
      <td>21.4</td>
      <td>21.3</td>
      <td>20.6</td>
      <td>20.9</td>
      <td>20.6</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>機関費</td>
      <td>11.9</td>
      <td>12.5</td>
      <td>11.7</td>
      <td>10.6</td>
      <td>11.8</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.4</td>
      <td>11.5</td>
      <td>11.9</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>教育費</td>
      <td>11.7</td>
      <td>11.9</td>
      <td>11.7</td>
      <td>12.1</td>
      <td>11.6</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.8</td>
      <td>11.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>国土保全及び開発費</td>
      <td>12.0</td>
      <td>11.3</td>
      <td>10.6</td>
      <td>10.2</td>
      <td>10.9</td>
      <td>10.5</td>
      <td>9.9</td>
      <td>10.4</td>
      <td>10.5</td>
      <td>10.5</td>
      <td>10.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 列名1の列はインデックスにして削除する
df.index = df["列名1"]
df = df.drop(["列名1"], axis=1)
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
      <th>列名0</th>
      <th>平成21年度</th>
      <th>平成22年度</th>
      <th>平成23年度</th>
      <th>平成24年度</th>
      <th>平成25年度</th>
      <th>平成26年度</th>
      <th>平成27年度</th>
      <th>平成28年度</th>
      <th>平成29年度</th>
      <th>平成30年度</th>
      <th>令和元年度</th>
    </tr>
    <tr>
      <th>列名1</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>社会保障関係費</th>
      <td>2</td>
      <td>29.8</td>
      <td>31.2</td>
      <td>32.7</td>
      <td>32.7</td>
      <td>31.8</td>
      <td>32.8</td>
      <td>33.7</td>
      <td>34.4</td>
      <td>34.6</td>
      <td>34.4</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>公債費</th>
      <td>3</td>
      <td>18.9</td>
      <td>20.3</td>
      <td>19.8</td>
      <td>20.9</td>
      <td>20.9</td>
      <td>21.4</td>
      <td>21.3</td>
      <td>20.6</td>
      <td>20.9</td>
      <td>20.6</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>機関費</th>
      <td>4</td>
      <td>11.9</td>
      <td>12.5</td>
      <td>11.7</td>
      <td>10.6</td>
      <td>11.8</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.4</td>
      <td>11.5</td>
      <td>11.9</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>教育費</th>
      <td>5</td>
      <td>11.7</td>
      <td>11.9</td>
      <td>11.7</td>
      <td>12.1</td>
      <td>11.6</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.8</td>
      <td>11.9</td>
    </tr>
    <tr>
      <th>国土保全及び開発費</th>
      <td>6</td>
      <td>12.0</td>
      <td>11.3</td>
      <td>10.6</td>
      <td>10.2</td>
      <td>10.9</td>
      <td>10.5</td>
      <td>9.9</td>
      <td>10.4</td>
      <td>10.5</td>
      <td>10.5</td>
      <td>10.8</td>
    </tr>
  </tbody>
</table>
</div>



## 列名を確認する
想定したデータ型になっているかを確認します。


```python
df.dtypes
```




    列名0         int64
    平成21年度    float64
    平成22年度    float64
    平成23年度    float64
    平成24年度    float64
    平成25年度    float64
    平成26年度    float64
    平成27年度    float64
    平成28年度    float64
    平成29年度    float64
    平成30年度    float64
    令和元年度     float64
    dtype: object



## 年度ごとに平均・中央値などを求めてみる
{{% notice document %}}
- [pandas.DataFrame.select_dtypes](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html)
- [pandas.DataFrame.agg](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html)
{{% /notice %}}

数値として読み取れる列を選択して、その平均や中央値を求めてみます。"list of functions and/or function names, e.g. [np.sum, 'mean']"ということで関数名や関数を指定すると、列ごとにそれを実行します。



```python
df.select_dtypes("number").agg(["mean", "median", "min", "max", "sum", "var", "std"])
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
      <th>列名0</th>
      <th>平成21年度</th>
      <th>平成22年度</th>
      <th>平成23年度</th>
      <th>平成24年度</th>
      <th>平成25年度</th>
      <th>平成26年度</th>
      <th>平成27年度</th>
      <th>平成28年度</th>
      <th>平成29年度</th>
      <th>平成30年度</th>
      <th>令和元年度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>4.500000</td>
      <td>15.700000</td>
      <td>15.90000</td>
      <td>15.983333</td>
      <td>15.783333</td>
      <td>15.683333</td>
      <td>15.816667</td>
      <td>15.883333</td>
      <td>15.866667</td>
      <td>15.933333</td>
      <td>15.900000</td>
      <td>15.933333</td>
    </tr>
    <tr>
      <th>median</th>
      <td>4.500000</td>
      <td>11.950000</td>
      <td>12.20000</td>
      <td>11.700000</td>
      <td>11.350000</td>
      <td>11.700000</td>
      <td>11.700000</td>
      <td>11.750000</td>
      <td>11.550000</td>
      <td>11.650000</td>
      <td>11.850000</td>
      <td>11.850000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.000000</td>
      <td>9.900000</td>
      <td>8.20000</td>
      <td>9.400000</td>
      <td>8.200000</td>
      <td>7.100000</td>
      <td>6.800000</td>
      <td>6.900000</td>
      <td>6.700000</td>
      <td>6.300000</td>
      <td>6.200000</td>
      <td>6.400000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.000000</td>
      <td>29.800000</td>
      <td>31.20000</td>
      <td>32.700000</td>
      <td>32.700000</td>
      <td>31.800000</td>
      <td>32.800000</td>
      <td>33.700000</td>
      <td>34.400000</td>
      <td>34.600000</td>
      <td>34.400000</td>
      <td>34.700000</td>
    </tr>
    <tr>
      <th>sum</th>
      <td>27.000000</td>
      <td>94.200000</td>
      <td>95.40000</td>
      <td>95.900000</td>
      <td>94.700000</td>
      <td>94.100000</td>
      <td>94.900000</td>
      <td>95.300000</td>
      <td>95.200000</td>
      <td>95.600000</td>
      <td>95.400000</td>
      <td>95.600000</td>
    </tr>
    <tr>
      <th>var</th>
      <td>3.500000</td>
      <td>57.364000</td>
      <td>72.29200</td>
      <td>80.605667</td>
      <td>88.293667</td>
      <td>83.053667</td>
      <td>92.613667</td>
      <td>99.489667</td>
      <td>103.422667</td>
      <td>106.434667</td>
      <td>104.080000</td>
      <td>103.862667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.870829</td>
      <td>7.573903</td>
      <td>8.50247</td>
      <td>8.978066</td>
      <td>9.396471</td>
      <td>9.113378</td>
      <td>9.623599</td>
      <td>9.974451</td>
      <td>10.169694</td>
      <td>10.316718</td>
      <td>10.201961</td>
      <td>10.191303</td>
    </tr>
  </tbody>
</table>
</div>



## 年度ごとの変化を可視化する
seabornを用いてデータをプロットしてみます。平成と令和が含まれる列のみを対象に抽出してプロットします。japanize_matplotlibにより日本語を表示できるようにしています。列名は `pandas.DataFrame.filter`でフィルタリングしています。

{{% notice document %}}
[seaborn](https://seaborn.pydata.org/)
{{% /notice %}}


```python
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

data = df.filter(regex="(平成|令和).+", axis=1).T
plt.figure(figsize=(12, 4))
sns.lineplot(data=data, palette="tab10", linewidth=2.5)
plt.grid()
plt.legend()

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
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>列名1</th>
      <th>社会保障関係費</th>
      <th>公債費</th>
      <th>機関費</th>
      <th>教育費</th>
      <th>国土保全及び開発費</th>
      <th>産業経済費</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>平成21年度</th>
      <td>29.8</td>
      <td>18.9</td>
      <td>11.9</td>
      <td>11.7</td>
      <td>12.0</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>平成22年度</th>
      <td>31.2</td>
      <td>20.3</td>
      <td>12.5</td>
      <td>11.9</td>
      <td>11.3</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>平成23年度</th>
      <td>32.7</td>
      <td>19.8</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>10.6</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>平成24年度</th>
      <td>32.7</td>
      <td>20.9</td>
      <td>10.6</td>
      <td>12.1</td>
      <td>10.2</td>
      <td>8.2</td>
    </tr>
    <tr>
      <th>平成25年度</th>
      <td>31.8</td>
      <td>20.9</td>
      <td>11.8</td>
      <td>11.6</td>
      <td>10.9</td>
      <td>7.1</td>
    </tr>
  </tbody>
</table>
</div>




    
![png](/images/prep/data/read-number-from-data_files/read-number-from-data_10_1.png)
    

