---
title: "tsfresh"
pre: "5.5.3 "
weight: 3
title_suffix: "Calcular características a partir de datos de series temporales"
---

{{% youtube "v9ZKyFhxkT8" %}}

<div class="pagetop-box">
    <p>Cuando se trabaja con datos de series temporales, es común calcular diversas características basadas en una columna de marcas de tiempo y valores numéricos. En esta página, utilizaremos tsfresh para calcular características a partir de datos de series temporales. Además, el video explica los diferentes enfoques para crear características.</p>
</div>


{{% notice document %}}
[tsfresh — tsfresh 0.18.1.dev documentation](https://tsfresh.readthedocs.io/en/latest/)
{{% /notice %}}

## tsfresh
[Overview on extracted features](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)を参考に、どんな特徴量が作成されるか確認してみます。


```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsfresh import extract_features

X = []
for id, it in enumerate(np.linspace(0.1, 100, 100)):
    for jt in range(10):
        X.append(
            [
                id,
                jt,
                jt + np.sin(it),
                jt % 2 + np.cos(it),
                jt % 3 + np.tan(it),
                np.log(it + jt),
            ]
        )

X = pd.DataFrame(X)
X.columns = ["id", "time", "fx1", "fx2", "fx3", "fx4"]
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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>time</th>
      <th>fx1</th>
      <th>fx2</th>
      <th>fx3</th>
      <th>fx4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0.099833</td>
      <td>0.995004</td>
      <td>0.100335</td>
      <td>-2.302585</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>1.099833</td>
      <td>1.995004</td>
      <td>1.100335</td>
      <td>0.095310</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>2.099833</td>
      <td>0.995004</td>
      <td>2.100335</td>
      <td>0.741937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>3</td>
      <td>3.099833</td>
      <td>1.995004</td>
      <td>0.100335</td>
      <td>1.131402</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>4</td>
      <td>4.099833</td>
      <td>0.995004</td>
      <td>1.100335</td>
      <td>1.410987</td>
    </tr>
  </tbody>
</table>
</div>




```python
X[X["id"] == 3].plot(subplots=True, sharex=True, figsize=(12, 10))
plt.show()
```


    
![png](/images/timeseries/shape/004-ts-extract-features_files/004-ts-extract-features_2_0.png)
    

## Calcular características

{{% notice document %}}
[Paquete tsfresh.feature_extraction](https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html)
{{% /notice %}}

Con `extract_features`, se pueden calcular todas las características de una sola vez. Además, utilizando las funciones disponibles en `tsfresh.feature_selection`, también es posible realizar la selección de características.


```python
extracted_features = extract_features(X, column_id="id", column_sort="time")
extracted_features.head()
```

    Feature Extraction: 100%|█
    




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
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fx1__variance_larger_than_standard_deviation</th>
      <th>fx1__has_duplicate_max</th>
      <th>fx1__has_duplicate_min</th>
      <th>fx1__has_duplicate</th>
      <th>fx1__sum_values</th>
      <th>fx1__abs_energy</th>
      <th>fx1__mean_abs_change</th>
      <th>fx1__mean_change</th>
      <th>fx1__mean_second_derivative_central</th>
      <th>fx1__median</th>
      <th>...</th>
      <th>fx4__permutation_entropy__dimension_6__tau_1</th>
      <th>fx4__permutation_entropy__dimension_7__tau_1</th>
      <th>fx4__query_similarity_count__query_None__threshold_0.0</th>
      <th>fx4__matrix_profile__feature_"min"__threshold_0.98</th>
      <th>fx4__matrix_profile__feature_"max"__threshold_0.98</th>
      <th>fx4__matrix_profile__feature_"mean"__threshold_0.98</th>
      <th>fx4__matrix_profile__feature_"median"__threshold_0.98</th>
      <th>fx4__matrix_profile__feature_"25"__threshold_0.98</th>
      <th>fx4__matrix_profile__feature_"75"__threshold_0.98</th>
      <th>fx4__mean_n_absolute_max__number_of_maxima_7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>45.998334</td>
      <td>294.084675</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-3.469447e-18</td>
      <td>4.599833</td>
      <td>...</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.915905</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>53.952941</td>
      <td>373.591982</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-6.938894e-18</td>
      <td>5.395294</td>
      <td>...</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.918724</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>53.538882</td>
      <td>369.141186</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000000e+00</td>
      <td>5.353888</td>
      <td>...</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.062001</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>45.143194</td>
      <td>286.290800</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-8.673617e-19</td>
      <td>4.514319</td>
      <td>...</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.186180</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>36.613658</td>
      <td>216.555992</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.000000e+00</td>
      <td>3.661366</td>
      <td>...</td>
      <td>-0.0</td>
      <td>-0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.295964</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3156 columns</p>
</div>


