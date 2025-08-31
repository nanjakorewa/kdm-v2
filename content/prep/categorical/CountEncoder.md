---
title: "Count Encoder"
pre: "3.4.3 "
weight: 3
title_replace: "pythonでCountエンコーダを使う"
---

## サンプルデータ
「人口総数」を予測したいとして、「元号」をエンコードしたいとします。


```python
import pandas as pd

X = pd.read_csv("../data/sample.csv")
TARGET_NAME = "人口総数"
FEATURE_NAME = "元号"
X.head()
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
      <th>元号</th>
      <th>和暦</th>
      <th>西暦</th>
      <th>人口総数</th>
      <th>町名</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>大正</td>
      <td>9.0</td>
      <td>1920.0</td>
      <td>394748</td>
      <td>A町</td>
    </tr>
    <tr>
      <th>1</th>
      <td>大正</td>
      <td>9.0</td>
      <td>1920.0</td>
      <td>31421</td>
      <td>B町</td>
    </tr>
    <tr>
      <th>2</th>
      <td>大正</td>
      <td>9.0</td>
      <td>1920.0</td>
      <td>226993</td>
      <td>C町</td>
    </tr>
    <tr>
      <th>3</th>
      <td>大正</td>
      <td>9.0</td>
      <td>1920.0</td>
      <td>253689</td>
      <td>D町</td>
    </tr>
    <tr>
      <th>4</th>
      <td>大正</td>
      <td>9.0</td>
      <td>1920.0</td>
      <td>288602</td>
      <td>E町</td>
    </tr>
  </tbody>
</table>
</div>



## CountEncoder

{{% notice document %}}
[category_encoders.count.CountEncoder](http://contrib.scikit-learn.org/category_encoders/count.html)
{{% /notice %}}


```python
from category_encoders.count import CountEncoder
from sklearn.compose import make_column_transformer

c_ce = CountEncoder()

y = X[TARGET_NAME]
X[f"{FEATURE_NAME}_ce"] = c_ce.fit_transform(X[FEATURE_NAME], y)
```

## 結果を確認する
カテゴリ変数の列が `CountEncoder` でエンコードされていることを確認します。


```python
X[[FEATURE_NAME, f"{FEATURE_NAME}_ce"]]
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
      <th>元号</th>
      <th>元号_ce</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>大正</td>
      <td>96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>大正</td>
      <td>96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>大正</td>
      <td>96</td>
    </tr>
    <tr>
      <th>3</th>
      <td>大正</td>
      <td>96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>大正</td>
      <td>96</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>975</th>
      <td>平成</td>
      <td>300</td>
    </tr>
    <tr>
      <th>976</th>
      <td>平成</td>
      <td>300</td>
    </tr>
    <tr>
      <th>977</th>
      <td>平成</td>
      <td>300</td>
    </tr>
    <tr>
      <th>978</th>
      <td>平成</td>
      <td>300</td>
    </tr>
    <tr>
      <th>979</th>
      <td>平成</td>
      <td>300</td>
    </tr>
  </tbody>
</table>
<p>980 rows × 2 columns</p>
</div>



## 元号の出現回数を確認する


```python
X[FEATURE_NAME].value_counts()
```




    昭和    584
    平成    300
    大正     96
    Name: 元号, dtype: int64


