---
title: "One-hot"
pre: "3.4.1 "
weight: 1
title_replace: "pythonでOne-hotエンコーダを使う"
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



## OneHotEncoder

{{% notice document %}}
- [sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [sklearn.compose.make_column_transformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html)
{{% /notice %}}


```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

c_oh = OneHotEncoder()

ct = make_column_transformer(
    (c_oh, ["元号"]),  # カテゴリ変数にonehotを適用するように指定
    sparse_threshold=0,  # 結果の行列をdenseな行列にする
)
```

## 結果を確認する
元号（大正・昭和・平成）が三次元のベクトルになっていることが確認できます。


```python
# 変換方法を求めて、変換した結果を返す
X_transform = ct.fit_transform(X)

# 変換後のテーブルを表示
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
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


