---
title: "Prophet"
pre: "5.6.1 "
weight: 1
title_suffix: "mencoba menjalankannya dengan Python"
---

{{% youtube "uUMDo8HOcrI" %}}

Meta (Facebook) telah merilis perpustakaan OSS untuk prediksi deret waktu. Cara instalasi di Python dapat dilihat pada [**Installation in Python**](https://facebook.github.io/prophet/docs/installation.html#python). Pada dasarnya, Anda hanya perlu menjalankan `pip install prophet` untuk menginstalnya.


{{% notice ref %}}
Taylor, Sean J., and Benjamin Letham. "Forecasting at scale." The American Statistician 72.1 (2018): 37-45.
{{% /notice %}}

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
from prophet import Prophet
```

## Data yang Digunakan untuk Eksperimen
Data selama satu tahun akan disiapkan. Pada bulan-bulan genap, terdapat kecenderungan penurunan nilai. Selain itu, data menunjukkan pola periodik setiap minggu. Periode data yang digunakan adalah dari 2020/1/1 hingga 2020/12/31.


```python
date = pd.date_range("2020-01-01", periods=365, freq="D")

# Target Prediksi
y = [
    np.cos(di.weekday())
    + di.month % 2 / 2
    + np.log(i + 1) / 3.0
    + np.random.rand() / 10
    for i, di in enumerate(date)
]

# Komponen Tren
x = [18627 + i - 364 for i in range(365)]
trend_y = [np.log(i + 1) / 3.0 for i, di in enumerate(date)]
weekly_y = [np.cos(di.weekday()) for i, di in enumerate(date)]
seasonal_y = [di.month % 2 / 2 for i, di in enumerate(date)]
noise_y = [np.random.rand() / 10 for i in range(365)]

df = pd.DataFrame({"ds": date, "y": y})
df.index = date

# Data yang Digunakan untuk Eksperimen
plt.title("Data Contoh")
sns.lineplot(data=df)
plt.show()
```


    
![png](/images/timeseries/forecast/001-Prophet_files/001-Prophet_5_0.png)
    

## Elemen yang Membentuk Data Deret Waktu
Istilah "data deret waktu" mencakup berbagai jenis data. Di sini, kita akan menangani data dengan karakteristik sebagai berikut:

- Data hanya terdiri dari cap waktu (timestamp) dan angka.
- Cap waktu tidak memiliki nilai yang hilang dan tersusun dalam interval yang sama (evenly spaced).



```python
plt.figure(figsize=(14, 6))
plt.title("y yang telah diuraikan berdasarkan komponennya")
plt.subplot(511)
plt.plot(x, trend_y, "-.", color="red", label="Tren", alpha=0.9)
plt.subplot(512)
plt.plot(x, weekly_y, "-.", color="green", label="Perubahan Periodik (Mingguan)", alpha=0.9)
plt.subplot(513)
plt.plot(x, seasonal_y, "-.", color="orange", label="Perubahan Periodik (Bulanan)", alpha=0.9)
plt.subplot(514)
plt.plot(x, noise_y, "-.", color="k", label="Komponen Noise")
plt.subplot(515)
sns.lineplot(data=df)
plt.show()
```

    
![png](/images/timeseries/forecast/001-Prophet_files/001-Prophet_7_0.png)
    

## Memprediksi Januari hingga Maret 2021 dengan Prophet
Menggunakan data dari 2020/1/1 hingga 2020/12/31, kita akan mencoba memprediksi tiga bulan ke depan. Karena data hanya mencakup satu tahun, kita menetapkan `yearly_seasonality=False`. Namun, karena terdapat pola periodik mingguan, kita menetapkan `daily_seasonality=True`.

```python
def train_and_forecast_pf(
    data,
    periods=90,
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=True,
):
    """Melatih dan memprediksi menggunakan Prophet

    Args:
        data (pandas.DataFrame): Data deret waktu
        periods (int, optional): Panjang periode prediksi. Default adalah 90.
        yearly_seasonality (bool, optional): Apakah terdapat komponen musiman tahunan. Default adalah False.
        weekly_seasonality (bool, optional): Apakah terdapat komponen musiman mingguan. Default adalah True.
        daily_seasonality (bool, optional): Apakah terdapat komponen musiman harian. Default adalah True.

    Returns:
        _type_: Model prediksi, hasil prediksi
    """
    assert "ds" in data.columns and "y" in data.columns, "Data input harus memiliki kolom 'ds' dan 'y'."
    # Melatih model
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
    )
    m.fit(df)

    # Memprediksi masa depan
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return m, forecast
```


```python
# Memeriksa Hasil Prediksi
periods = 90
m, forecast = train_and_forecast_pf(
    df,
    periods=periods,
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=True,
)
fig = m.plot(forecast)
plt.axvspan(18627, 18627 + periods, color="coral", alpha=0.4, label="予測期間")
plt.legend()
plt.show()
```

    Initial log joint probability = -32.1541
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       772.276   5.98161e-05       56.7832           1           1      135   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         131        772.59    0.00128893       157.592   1.465e-05       0.001      217  LS failed, Hessian reset 
         181       772.678   3.78737e-05       49.0389   6.852e-07       0.001      326  LS failed, Hessian reset 
         199       772.681   1.42622e-06       43.2231      0.6929      0.6929      350   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         230       772.681   6.80165e-06       56.0478   7.185e-08       0.001      432  LS failed, Hessian reset 
         245       772.681   4.06967e-08       48.5475      0.1802      0.8285      454   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance



    
![png](/images/timeseries/forecast/001-Prophet_files/001-Prophet_10_1.png)
    


### Pengaruh Penetapan Periodisitas
```{warning}
Jika periodisitas ditetapkan meskipun data tidak memiliki pola musiman, hasil prediksi dapat menjadi tidak akurat.
```

Pada contoh berikut, musiman tahunan sengaja ditetapkan (yearly_seasonality=True). Akibat pengaruh komponen untuk menangkap pola tahunan, prediksi untuk tahun 2022 menunjukkan peningkatan yang agak tidak wajar.


```python
periods = 90
m, forecast = train_and_forecast_pf(
    df,
    periods=periods,
    yearly_seasonality=True,
)
fig = m.plot(forecast)
plt.title("prophetの予測結果")
plt.axvspan(18627, 18627 + periods, color="coral", alpha=0.4, label="予測期間")
plt.legend()
plt.show()
```

    Initial log joint probability = -32.1541
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
          99       1076.54   0.000445309       68.8033           1           1      133   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         199       1078.13   0.000151685       92.7241           1           1      256   
        Iter      log prob        ||dx||      ||grad||       alpha      alpha0  # evals  Notes 
         246       1078.14   1.78997e-06       84.0649    1.52e-08       0.001      353  LS failed, Hessian reset 
         261       1078.14   3.82403e-08       101.692      0.2973           1      372   
    Optimization terminated normally: 
      Convergence detected: relative gradient magnitude is below tolerance



    
![png](/images/timeseries/forecast/001-Prophet_files/001-Prophet_12_1.png)
    

