---
title: "処理の進捗を表示"
pre: "3.5.2 "
weight: 2
title_replace: "tqdmで処理の進捗を表示する"
---

pandasでデータ加工をする時、時間がかかる場合があります。
この時にtqdmを用いて進捗を表示することができます。

{{% notice document %}}
[tqdm Pandas Integration](https://github.com/tqdm/tqdm#table-of-contents)
{{% /notice %}}

> Due to popular demand we've added support for pandas -- here's an example for DataFrame.progress_apply and DataFrameGroupBy.progress_apply:


```python
import pandas as pd

df = pd.read_csv("../data/sample.csv")[:100]
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



# 処理の進捗を表示する
`rand_with_sleep(x: int)`はランダムな時間だけ待機して待機した時間(sec)を返します。

`df["人口総数"]` には待機した時間が記録され、この処理の進捗のプログレスバーが表示されます。


```python
import numpy as np
from time import sleep
from tqdm import tqdm

tqdm.pandas()


def rand_with_sleep(x: int) -> float:
    rnd = np.random.rand() / 50.0
    sleep(rnd)
    return f"{rnd} sec"


df["人口総数"] = df["人口総数"].progress_apply(lambda x: rand_with_sleep(x))
df.head()
```

    100%|█████████████████████████████████████| 100/100 [00:01<00:00, 78.99it/s]





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
      <td>0.00290391291282283 sec</td>
      <td>A町</td>
    </tr>
    <tr>
      <th>1</th>
      <td>大正</td>
      <td>9.0</td>
      <td>1920.0</td>
      <td>0.015489350572629845 sec</td>
      <td>B町</td>
    </tr>
    <tr>
      <th>2</th>
      <td>大正</td>
      <td>9.0</td>
      <td>1920.0</td>
      <td>0.007857787674584393 sec</td>
      <td>C町</td>
    </tr>
    <tr>
      <th>3</th>
      <td>大正</td>
      <td>9.0</td>
      <td>1920.0</td>
      <td>0.016366338258575495 sec</td>
      <td>D町</td>
    </tr>
    <tr>
      <th>4</th>
      <td>大正</td>
      <td>9.0</td>
      <td>1920.0</td>
      <td>0.017993046099223576 sec</td>
      <td>E町</td>
    </tr>
  </tbody>
</table>
</div>


