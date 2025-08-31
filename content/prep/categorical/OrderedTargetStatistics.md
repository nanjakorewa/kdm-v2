---
title: "Ordered Target Statistics"
pre: "3.4.4 "
weight: 4
title_replace: "pythonでOrdered Target Statisticsを使う"
---

{{% notice ref %}}
Prokhorenkova, Liudmila, et al. "CatBoost: unbiased boosting with categorical features." arXiv preprint arXiv:1706.09516 (2017).
{{% /notice %}}
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



## Ordered Target Statistics

{{% notice document %}}
- [category_encoders.cat_boost.CatBoostEncoder](http://contrib.scikit-learn.org/category_encoders/catboost.html)
- [sklearn.compose.make_column_transformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html)
{{% /notice %}}


```python
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.compose import make_column_transformer

c_ots = CatBoostEncoder()

y = X[TARGET_NAME]
X[f"{FEATURE_NAME}_ots"] = c_ots.fit_transform(X[FEATURE_NAME], y)
```

## 結果を確認する
カテゴリ変数の列が `CatBoostEncoder` でエンコードされていることを確認します。


```python
X[[FEATURE_NAME, f"{FEATURE_NAME}_ots"]]
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
      <th>元号_ots</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>大正</td>
      <td>109247.087755</td>
    </tr>
    <tr>
      <th>1</th>
      <td>大正</td>
      <td>251997.543878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>大正</td>
      <td>178472.029252</td>
    </tr>
    <tr>
      <th>3</th>
      <td>大正</td>
      <td>190602.271939</td>
    </tr>
    <tr>
      <th>4</th>
      <td>大正</td>
      <td>203219.617551</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>975</th>
      <td>平成</td>
      <td>101664.388810</td>
    </tr>
    <tr>
      <th>976</th>
      <td>平成</td>
      <td>101406.646760</td>
    </tr>
    <tr>
      <th>977</th>
      <td>平成</td>
      <td>101126.513717</td>
    </tr>
    <tr>
      <th>978</th>
      <td>平成</td>
      <td>100845.716013</td>
    </tr>
    <tr>
      <th>979</th>
      <td>平成</td>
      <td>100588.073626</td>
    </tr>
  </tbody>
</table>
<p>980 rows × 2 columns</p>
</div>



## エンコード結果の分布を確認する


```python
import matplotlib.pyplot as plt
import japanize_matplotlib

plt.figure(figsize=(8, 4))
for i, ci in enumerate(X[FEATURE_NAME].unique()):
    plt.hist(
        X.query(f"{FEATURE_NAME}=='{ci}'")[f"{FEATURE_NAME}_ots"], label=ci, alpha=0.5
    )

plt.title("エンコードされた結果の分布")
plt.legend(title=FEATURE_NAME)
```



    
![png](/images/prep/categorical/OrderedTargetStatistics_files/OrderedTargetStatistics_7_1.png)
    

