---
title: "パターンマッチで列を選択"
pre: "3.5.3 "
weight: 3
---

## サンプルデータ


```python
import pandas as pd

X = pd.read_csv("../data/sample.csv")
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



## sklearn.compose.make_column_selector
[sklearn.compose.make_column_selector](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_selector.html)
を使用します。`pattern="暦"`で暦が含まれる列を選択し、StandardScalerを適用します。


```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

n_ss = StandardScaler()

# 暦が含まれる列のみスケーリング
ct = make_column_transformer(
    (n_ss, make_column_selector(pattern="暦")), sparse_threshold=0
)
X_transform = ct.fit_transform(X)

# 変換後のテーブル
pd.DataFrame(X_transform).head()
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
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.946623</td>
      <td>-1.665466</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.946623</td>
      <td>-1.665466</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.946623</td>
      <td>-1.665466</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.946623</td>
      <td>-1.665466</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.946623</td>
      <td>-1.665466</td>
    </tr>
  </tbody>
</table>
</div>


