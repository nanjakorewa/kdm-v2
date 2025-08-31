---
title: "SQLで操作"
pre: "3.2.4 "
weight: 4
title_replace: "pandas.DataFrameでSQLを実行する"
---

<div class="pagetop-box">
<p>使用しているデータセットについては『<a href="../sample-data/">総務省のデータ</a>』の項を参照してください。
Pandasを使用すればデータベースをSQLで操作することもできます。
以下の例では、読み込んだｃｓｖをSQLデータベースに変換し、それに対して SELECT文を実行します。</p>
</div>

## データを読み込む


```python
import pandas as pd

df = pd.read_csv("./z-001.csv")
df.columns = df.columns.str.replace(" ", "")  # カラム名に含まれるスペースは消す
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
      <th>Unnamed:0</th>
      <th>Unnamed:1</th>
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



## SQLで操作できる形式にする
{{% notice document %}}
[pandas.DataFrame.to_sql](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_sql.html)
{{% /notice %}}


```python
from sqlite3 import connect

conn = connect(":memory:")
df.to_sql("Table1", conn)
```

## SQLを実行する
{{% notice document %}}
[pandas.read_sql](https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html)
{{% /notice %}}


```python
pd.read_sql("SELECT 平成21年度 FROM Table1", conn)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>29.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.9</td>
    </tr>
  </tbody>
</table>
</div>


