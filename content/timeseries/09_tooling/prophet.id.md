---
title: "Menggunakan Prophet"
pre: "2.8.1 "
weight: 1
searchtitle: "Peramalan deret waktu dalam python menggunakan prophet"
---

Silakan merujuk ke "[prophet Installation](https://facebook.github.io/prophet/docs/installation.html)" untuk petunjuk instalasi prophet. Juga, silakan merujuk ke Quick Start ([prophet | Quick Start](https://facebook.github.io/prophet/docs/quick_start.html)) dalam dokumentasi resmi.

{{% notice seealso %}}
[K_DM - Serangkaian waktu > perkiraan > Prophet](https://k-dm.work/en/timeseries/forecast/001-prophet/)
{{% /notice %}}

## Membuat data deret waktu

Buat data deret waktu dummy.


```python
import numpy as np
import pandas as pd
import seaborn as sns
from prophet import Prophet

sns.set(rc={"figure.figsize": (15, 8)})
```

```python
date = pd.date_range("2020-01-01", periods=365, freq="D")
y = [np.cos(di.weekday()) + di.month % 2 + np.log(i + 1) for i, di in enumerate(date)]

df = pd.DataFrame({"ds": date, "y": y})
df.index = date
sns.lineplot(data=df)
```

    
![png](/images/basic/timeseries/prophet_files/prophet_3_1.png)
    


## Kereta Api Prophet


```python
m = Prophet(yearly_seasonality=False, daily_seasonality=True)
m.fit(df)
```

    Initial log joint probability = -24.5101
    <prophet.forecaster.Prophet at 0x7fe3b5d93d60>
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       798.528    0.00821602       204.832           1           1      136   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         136       799.486    0.00040141       83.1101    5.02e-06       0.001      225  LS failed, Hessian reset 
         158       799.529    0.00027729       48.4168   3.528e-06       0.001      291  LS failed, Hessian reset 
         199        799.55   3.15651e-05       54.5691           1           1      345   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         204       799.553   3.54297e-05       56.7445    5.36e-07       0.001      397  LS failed, Hessian reset 
         267       799.556   6.19351e-08       44.7029      0.2081           1      490   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance


## Membuat data untuk prakiraan dan menjalankan prakiraan


```python
future = m.make_future_dataframe(periods=90)
forecast = m.predict(future)
fig1 = m.plot(forecast)
```


    
![png](/images/basic/timeseries/prophet_files/prophet_8_0.png)
    

