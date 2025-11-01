---
title: "Anomaly Detection Toolkit (ADTK) Basics | Threshold and Statistical Detectors"
linkTitle: "ADTK basics"
seo_title: "Anomaly Detection Toolkit (ADTK) Basics | Threshold and Statistical Detectors"
pre: "2.9.1 "
weight: 1
---

We will perform anomaly detection using the [Anomaly Detection Toolkit (ADTK)](https://adtk.readthedocs.io/en/stable/index.html).  
The original data is sourced from the [Numenta Anomaly Benchmark](https://github.com/numenta/NAB/blob/master/data/artificialWithAnomaly/art_daily_jumpsdown.csv).


```python
import pandas as pd

s_train = pd.read_csv(
    "./training.csv", index_col="timestamp", parse_dates=True, squeeze=True
)
from adtk.data import validate_series

s_train = validate_series(s_train)
print(s_train)
```

    timestamp
    2014-04-01 00:00:00    18.090486
    2014-04-01 00:05:00    20.359843
    2014-04-01 00:10:00    21.105470
    2014-04-01 00:15:00    21.151585
    2014-04-01 00:20:00    18.137141
                             ...    
    2014-04-14 23:35:00    18.269290
    2014-04-14 23:40:00    19.087351
    2014-04-14 23:45:00    19.594689
    2014-04-14 23:50:00    19.767817
    2014-04-14 23:55:00    20.479156
    Freq: 5T, Name: value, Length: 4032, dtype: float64
    

      s_train = pd.read_csv("./training.csv", index_col="timestamp", parse_dates=True, squeeze=True)
    


```python
from adtk.visualization import plot

plot(s_train)
```





    
![png](/images/basic/anomaly/adtk1_files/adtk1_2_1.png)
    

## Comparison of Anomaly Detection Methods

We will perform anomaly detection using [SeasonalAD](https://adtk.readthedocs.io/en/stable/notebooks/demo.html?highlight=SeasonalAD#SeasonalAD).  
For other methods, refer to [Detector](https://adtk.readthedocs.io/en/stable/notebooks/demo.html?highlight=SeasonalAD#Detector).



```python
import matplotlib.pyplot as plt
from adtk.detector import (
    AutoregressionAD,
    InterQuartileRangeAD,
    LevelShiftAD,
    PersistAD,
    SeasonalAD,
)

model_dict = {
    "LevelShiftAD": LevelShiftAD(window=5),
    "SeasonalAD": SeasonalAD(),
    "PersistAD": PersistAD(c=3.0, side="positive"),
    "InterQuartileRangeAD": InterQuartileRangeAD(c=1.5),
    "AutoregressionAD": AutoregressionAD(n_steps=7 * 2, step_size=24, c=3.0),
}

for model_name, model in model_dict.items():
    anomalies = model.fit_detect(s_train)
    plot(s_train, anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
    plt.title(model_name)
    plt.show()
```


    
![png](/images/basic/anomaly/adtk1_files/adtk1_4_0.png)
    



    
![png](/images/basic/anomaly/adtk1_files/adtk1_4_1.png)
    



    
![png](/images/basic/anomaly/adtk1_files/adtk1_4_2.png)
    



    
![png](/images/basic/anomaly/adtk1_files/adtk1_4_3.png)
    



    
![png](/images/basic/anomaly/adtk1_files/adtk1_4_4.png)
    

