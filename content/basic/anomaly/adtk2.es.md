---
title: "Detección de anomalías con ADTK (Parte 2) | Ventanas deslizantes y detectores estacionales"
linkTitle: "ADTK avanzado"
seo_title: "Detección de anomalías con ADTK (Parte 2) | Ventanas deslizantes y detectores estacionales"
pre: "2.9.2 "
weight: 2
---

Vamos a realizar detección de anomalías utilizando [Anomaly Detection Toolkit (ADTK)](https://adtk.readthedocs.io/en/stable/index.html). 
Aplicaremos la detección de anomalías a datos artificiales multidimensionales. Esta vez, trabajaremos con datos de múltiples dimensiones.


```python
import numpy as np
import pandas as pd
from adtk.data import validate_series

s_train = pd.read_csv("./training.csv", index_col="timestamp", parse_dates=True)
s_train = validate_series(s_train)
s_train["value2"] = s_train["value"].apply(lambda v: np.sin(v) + np.cos(v))
s_train
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
      <th>value</th>
      <th>value2</th>
    </tr>
    <tr>
      <th>timestamp</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-04-01 00:00:00</th>
      <td>18.090486</td>
      <td>0.037230</td>
    </tr>
    <tr>
      <th>2014-04-01 00:05:00</th>
      <td>20.359843</td>
      <td>1.058643</td>
    </tr>
    <tr>
      <th>2014-04-01 00:10:00</th>
      <td>21.105470</td>
      <td>0.141581</td>
    </tr>
    <tr>
      <th>2014-04-01 00:15:00</th>
      <td>21.151585</td>
      <td>0.076564</td>
    </tr>
    <tr>
      <th>2014-04-01 00:20:00</th>
      <td>18.137141</td>
      <td>0.103122</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2014-04-14 23:35:00</th>
      <td>18.269290</td>
      <td>0.288071</td>
    </tr>
    <tr>
      <th>2014-04-14 23:40:00</th>
      <td>19.087351</td>
      <td>1.207420</td>
    </tr>
    <tr>
      <th>2014-04-14 23:45:00</th>
      <td>19.594689</td>
      <td>1.413067</td>
    </tr>
    <tr>
      <th>2014-04-14 23:50:00</th>
      <td>19.767817</td>
      <td>1.401750</td>
    </tr>
    <tr>
      <th>2014-04-14 23:55:00</th>
      <td>20.479156</td>
      <td>0.939501</td>
    </tr>
  </tbody>
</table>
<p>4032 rows × 2 columns</p>
</div>




```python
from adtk.visualization import plot

plot(s_train)
```


    
![png](/images/basic/anomaly/adtk2_files/adtk2_2_1.png)
    

## Comparación de métodos de detección de anomalías

Realizaremos la detección de anomalías utilizando [SeasonalAD](https://adtk.readthedocs.io/en/stable/notebooks/demo.html?highlight=SeasonalAD#SeasonalAD). Para otros métodos, consulte [Detector](https://adtk.readthedocs.io/en/stable/notebooks/demo.html?highlight=SeasonalAD#Detector).



```python
import matplotlib.pyplot as plt
from adtk.detector import OutlierDetector, PcaAD, RegressionAD
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

model_dict = {
    "OutlierDetector": OutlierDetector(LocalOutlierFactor(contamination=0.05)),
    "RegressionAD": RegressionAD(regressor=LinearRegression(), target="value2", c=3.0),
    "PcaAD": PcaAD(k=2),
}

for model_name, model in model_dict.items():
    anomalies = model.fit_detect(s_train)

    plot(
        s_train,
        anomaly=anomalies,
        ts_linewidth=1,
        ts_markersize=3,
        anomaly_color="red",
        anomaly_alpha=0.3,
        curve_group="all",
    )
    plt.title(model_name)
    plt.show()
```
    


    
![png](/images/basic/anomaly/adtk2_files/adtk2_4_1.png)
    
    


    
![png](/images/basic/anomaly/adtk2_files/adtk2_4_3.png)
    

    
![png](/images/basic/anomaly/adtk2_files/adtk2_4_5.png)
    

