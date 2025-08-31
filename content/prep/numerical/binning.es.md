---
title: "Binning"
pre: "3.3.1 "
weight: 1
title_replace: "División de un número en bins"
---


```python
import numpy as np
import pandas as pd

df = pd.read_csv("../data/sample.csv")
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



## Binning de datos en cada cuantil

{{% notice document %}}
[pandas.qcut](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html)
{{% /notice %}}

Binning based on how much of the data is X% of the total when sorted.
```
0.0        46.0
0.1     18002.9 <- 10%の値
0.2     20476.8
0.3     22755.0
0.4     26204.8
0.5     30824.0
0.6     45622.6
0.7     89873.9
0.8    245544.0
0.9    290714.1 <- 90%の値
1.0    765403.0
```


```python
df["人口総数_ビン化"] = pd.qcut(df["人口総数"], q=11)
df[["人口総数", "人口総数_ビン化"]].head()
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
      <th>人口総数</th>
      <th>人口総数_ビン化</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>394748</td>
      <td>(294187.0, 765403.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>31421</td>
      <td>(28169.0, 34470.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>226993</td>
      <td>(214984.0, 249929.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>253689</td>
      <td>(249929.0, 294187.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>288602</td>
      <td>(249929.0, 294187.0]</td>
    </tr>
  </tbody>
</table>
</div>


