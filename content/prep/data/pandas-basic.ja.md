---
title: "pandasの使用"
pre: "3.2.2 "
weight: 2
---


## データを読み込む
- テーブル全体 ＝ DataFrameと呼ぶ
- テーブルの各行 ＝ rowと呼ぶ
- テーブルの各列 ＝ columnと呼び、Seriesというデータに収まっている


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



## データを整形する
{{% notice document %}}
[pandas.DataFrame](https://pandas.pydata.org/docs/reference/frame.html)
{{% /notice %}}
列名の変更・重複の除去・型の変更・欠損の置き換えなどを実行する。データ全体の欠損を置き換えたり、データの各列・行ごとに一括で統計量を求めるメソッドが用意されている。

```
pandas.DataFrame.mad
pandas.DataFrame.max
pandas.DataFrame.mean
pandas.DataFrame.median
pandas.DataFrame.min
pandas.DataFrame.mode
```

### 欠損や余分なデータへの対応

{{% notice document %}}
- [pandas.DataFrame.fillna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html?highlight=fillna)
- [pandas.DataFrame.dropna](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)
- [pandas.DataFrame.applymap](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.applymap.html?highlight=applymap)
{{% /notice %}}


```python
df = df.fillna(0)  # 欠損を0に置き換え
df = df.drop(["Unnamed: 0"], axis=1)  #  'Unnamed: 0'列を削除
df.columns = df.columns.str.replace("Unnamed: ", "列")  # 列名の Unnamed を別の文字列に置換
df = df.applymap(
    lambda x: int(x) if type(x) == float else x
)  # 数値データならば全て整数にするような関数を全体に適用する

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
      <th>列1</th>
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
      <td>社会保障関係費</td>
      <td>29</td>
      <td>31</td>
      <td>32</td>
      <td>32</td>
      <td>31</td>
      <td>32</td>
      <td>33</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>公債費</td>
      <td>18</td>
      <td>20</td>
      <td>19</td>
      <td>20</td>
      <td>20</td>
      <td>21</td>
      <td>21</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>機関費</td>
      <td>11</td>
      <td>12</td>
      <td>11</td>
      <td>10</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>教育費</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>12</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>国土保全及び開発費</td>
      <td>12</td>
      <td>11</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>9</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



### データを絞り込む

{{% notice document %}}
[pandas.DataFrame.query](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)
{{% /notice %}}

```python
# 平成21年度 > 15となるような行のみを抽出します
df.query("平成21年度 > 15")
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
      <th>列1</th>
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
      <td>社会保障関係費</td>
      <td>29</td>
      <td>31</td>
      <td>32</td>
      <td>32</td>
      <td>31</td>
      <td>32</td>
      <td>33</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>1</th>
      <td>公債費</td>
      <td>18</td>
      <td>20</td>
      <td>19</td>
      <td>20</td>
      <td>20</td>
      <td>21</td>
      <td>21</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 平成21年度 > 10 & 平成22年度 < 15 となるような行のみを抽出します
df.query("平成21年度 > 10 & 平成22年度 < 15")
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
      <th>列1</th>
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
      <th>2</th>
      <td>機関費</td>
      <td>11</td>
      <td>12</td>
      <td>11</td>
      <td>10</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>教育費</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>12</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>4</th>
      <td>国土保全及び開発費</td>
      <td>12</td>
      <td>11</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>9</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



### データの列ごとの統計量を算出する
{{% notice document %}}
[pandas.Series — pandas 1.4.1 documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.html)
{{% /notice %}}

```python
# 数値の列のみに絞り込む
df_num = df.select_dtypes("number")

# 列をリストにする関数
def concat_str(series: pd.Series) -> list:
    """列に含まれるデータをリストにする"""
    return [s for s in series]


# 列ごとの統計量を求める、指定した関数にデータを渡して実行することもできる
df_num.agg(["min", "max", "median", "mean", concat_str])
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
      <th>min</th>
      <td>9</td>
      <td>8</td>
      <td>9</td>
      <td>8</td>
      <td>7</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>max</th>
      <td>29</td>
      <td>31</td>
      <td>32</td>
      <td>32</td>
      <td>31</td>
      <td>32</td>
      <td>33</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
    <tr>
      <th>median</th>
      <td>11.5</td>
      <td>11.5</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>11.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>15.0</td>
      <td>15.5</td>
      <td>15.333333</td>
      <td>15.333333</td>
      <td>15.0</td>
      <td>15.166667</td>
      <td>15.166667</td>
      <td>15.333333</td>
      <td>15.333333</td>
      <td>15.333333</td>
      <td>15.333333</td>
    </tr>
    <tr>
      <th>concat_str</th>
      <td>[29, 18, 11, 11, 12, 9]</td>
      <td>[31, 20, 12, 11, 11, 8]</td>
      <td>[32, 19, 11, 11, 10, 9]</td>
      <td>[32, 20, 10, 12, 10, 8]</td>
      <td>[31, 20, 11, 11, 10, 7]</td>
      <td>[32, 21, 11, 11, 10, 6]</td>
      <td>[33, 21, 11, 11, 9, 6]</td>
      <td>[34, 20, 11, 11, 10, 6]</td>
      <td>[34, 20, 11, 11, 10, 6]</td>
      <td>[34, 20, 11, 11, 10, 6]</td>
      <td>[34, 20, 11, 11, 10, 6]</td>
    </tr>
  </tbody>
</table>
</div>



## 列ごとの統計量を求める
{{% notice document %}}
[pandas.Series](https://pandas.pydata.org/docs/reference/series.html)
{{% /notice %}}
ドキュメントにメソッドとプロパティの一覧が記載されている、`min`や`max`などを始めとして様々な計算を簡単に実行することができる。欠損が含まれているかどうかを示すプロパティも定義されている。

### 数値データ


```python
平成21年の列 = df["平成21年度"]
print(平成21年の列)
```

    0    29
    1    18
    2    11
    3    11
    4    12
    5     9
    Name: 平成21年度, dtype: int64



```python
print(f" min: {平成21年の列.min()}")
print(f" max: {平成21年の列.max()}")
print(f" median: {平成21年の列.median()}")
print(f" dtypes(データ型): {平成21年の列.dtypes}")
print(f" hasnans(欠損の有無): {平成21年の列.hasnans}")
```

     min: 9
     max: 29
     median: 11.5
     dtypes(データ型): int64
     hasnans(欠損の有無): False


### 文字列データ
文字列の出現回数を求めてみます。出現回数を知るには `value_counts` を使います。

もしも文字列に対すしてどのような操作が用意されているか忘れてしまった場合やまったく見当がつかない場合、ドキュメントを調べることで見つけることができます。たとえば文字列の出現回数を知りたい時はドキュメントで「count value」などと検索すれば、

- Categorical data
- How to calculate summary statistics?
- pandas.DataFrame.count
- pandas.Series.count

などのページが出てきます。そして、[pandas.DataFrame.count](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.count.html?highlight=count%20value%20column)などを見れば
『**See also**(こちらもご覧ください)』の欄に`DataFrame.value_counts`へのリンクがあります。`value_counts`でドキュメントを検索すれば [pandas.Series.value_counts](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html?highlight=value_counts#pandas.Series.value_counts)が見つかり、これを使えば文字列の出現回数を求められることが分かります。


```python
df["平成21年度"].value_counts()
```




    11    2
    29    1
    18    1
    12    1
    9     1
    Name: 平成21年度, dtype: int64


