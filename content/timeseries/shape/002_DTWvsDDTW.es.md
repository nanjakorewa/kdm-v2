---
title: "DTW y DDTW"
pre: "5.5.2 "
weight: 3
title_suffix: "Comprender el funcionamiento"
---


```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

from utils import DDTW

np.random.seed(777)
```

## Gráfica de dos formas de onda utilizadas en el experimento


```python
data1 = np.array([12.0 * np.sin(i / 2.1) + 20 for i in range(30)])
data2 = np.array([10.0 * np.sin(i / 2.0) + np.random.rand() for i in range(30)])

plt.figure(figsize=(12, 4))

# Gráfica de formas de onda
plt.plot(data1, label="data1", color="k")
plt.plot(data2, label="data2", color="r")
plt.legend()
plt.show()
```


    
![png](/images/timeseries/shape/002_DTWvsDDTW_files/002_DTWvsDDTW_4_0.png)
    


## DTW


```python
d, paths = dtw.warping_paths(
    data1,
    data2,
    window=25,
)
best_path = dtw.best_path(paths)
dtwvis.plot_warpingpaths(data1, data2, paths, best_path)
```




    (<Figure size 720x720 with 4 Axes>,
     [<AxesSubplot:>, <AxesSubplot:>, <AxesSubplot:>, <AxesSubplot:>])




    
![png](/images/timeseries/shape/002_DTWvsDDTW_files/002_DTWvsDDTW_6_1.png)
    


## DDTW


```python
γ_mat, arrows, ddtw = DDTW(np.array(data1), np.array(data2))

sns.set(rc={"figure.figsize": (18, 15)})
sns.set(font="IPAexGothic")
ax = sns.heatmap(-1 * γ_mat, cmap="YlGnBu")
ax.set_title(f"DDTW = {ddtw}")
ax.invert_xaxis()
ax.invert_yaxis()
ax.set_xlabel("w2")
ax.set_ylabel("w2")
plt.show()
```

    
![png](/images/timeseries/shape/002_DTWvsDDTW_files/002_DTWvsDDTW_8_1.png)
    

