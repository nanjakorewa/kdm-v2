---
title: "Target Encoder"
pre: "3.4.2 "
weight: 2
title_replace: "pythonでTarget Encoderを使う"
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



## TargetEncoder

{{% notice document %}}
- [category_encoders.target_encoder.TargetEncoder](http://contrib.scikit-learn.org/category_encoders/targetencoder.html)
- [sklearn.compose.make_column_transformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html)
{{% /notice %}}


```python
from category_encoders.target_encoder import TargetEncoder

c_te = TargetEncoder()

y = X[TARGET_NAME]
X[f"{FEATURE_NAME}_te"] = c_te.fit_transform(X[FEATURE_NAME], y)
```

## 結果を確認する
カテゴリ変数の列が `TargetEncoder` でエンコードされていることを確認します。
この方法は、目的変数の平均値をそのままエンコードに使用します。つまり、あるデータをエンコードするために**そのデータの目的変数の情報**を使用しています(leakage[1]と呼びます)。そのため、データ数が少ない場合は特に、実際に将来のデータに対して予測した場合とCVで評価した場合を比較すると、CV時に誤差が少なく見積もられる可能性がある点に注意して下さい。

{{% notice ref %}}
[1] Kaufman, Shachar, et al. "Leakage in data mining: Formulation, detection, and avoidance." ACM Transactions on Knowledge Discovery from Data (TKDD) 6.4 (2012): 1-21.
{{% /notice %}}

{{% youtube "0l3g7pcx5FM" %}}


```python
X[[FEATURE_NAME, f"{FEATURE_NAME}_te"]]
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
      <th>元号_te</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>大正</td>
      <td>144791.552083</td>
    </tr>
    <tr>
      <th>1</th>
      <td>大正</td>
      <td>144791.552083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>大正</td>
      <td>144791.552083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>大正</td>
      <td>144791.552083</td>
    </tr>
    <tr>
      <th>4</th>
      <td>大正</td>
      <td>144791.552083</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>975</th>
      <td>平成</td>
      <td>100294.140000</td>
    </tr>
    <tr>
      <th>976</th>
      <td>平成</td>
      <td>100294.140000</td>
    </tr>
    <tr>
      <th>977</th>
      <td>平成</td>
      <td>100294.140000</td>
    </tr>
    <tr>
      <th>978</th>
      <td>平成</td>
      <td>100294.140000</td>
    </tr>
    <tr>
      <th>979</th>
      <td>平成</td>
      <td>100294.140000</td>
    </tr>
  </tbody>
</table>
<p>980 rows × 2 columns</p>
</div>



## 元号ごとの平均値
元号ごとのターゲットの平均値を用いてエンコードされていることを確認します


```python
X.groupby(FEATURE_NAME).agg("mean")[TARGET_NAME]
```




    元号
    大正    144791.552083
    平成    100294.140000
    昭和    108003.279110
    Name: 人口総数, dtype: float64


