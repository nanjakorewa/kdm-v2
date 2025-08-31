---
title: Proceso MA
weight: 6
pre: "<b>5.3.2 </b>"
---

```python
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
```

## Generar datos para un proceso MA

Preparar funciones para generar datos


```python
def create_MAdata(thetas=[0.1], mu=1, N=400, init=1, c=1, sigma=0.3):
    """GGenerate MA Process data"""
    epsilon = np.random.normal(loc=0, scale=sigma, size=N)
    data = np.zeros(N)
    data[0] = init

    for t in range(2, N):
        res = mu + epsilon[t]
        for j, theta_j in enumerate(thetas):
            res += theta_j * epsilon[t - j - 1]
        data[t] = res
    return data
```

### MA(1)


```python
plt.figure(figsize=(12, 6))
thetas = [0.5]
ma1_1 = create_MAdata(thetas=thetas)
plt.plot(ma1_1)
plt.show()
```


    
![png](/images/timeseries/model/005-2-MA-process_files/005-2-MA-process_4_1.png)
    


### MA(2)


```python
plt.figure(figsize=(12, 6))
thetas = [0.5, 0.5]
ma1_2 = create_MAdata(thetas=thetas)
plt.plot(ma1_1)
plt.show()
```


    
![png](/images/timeseries/model/005-2-MA-process_files/005-2-MA-process_6_1.png)
    


### MA(5)


```python
plt.figure(figsize=(12, 6))
thetas = [0.5 for _ in range(10)]
ma1_5 = create_MAdata(thetas=thetas)
plt.plot(ma1_5)
plt.show()
```


    
![png](/images/timeseries/model/005-2-MA-process_files/005-2-MA-process_8_1.png)
    
