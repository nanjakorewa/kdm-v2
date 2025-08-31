---
title: "Profeta"
pre: "5.6.1 "
weight: 1
title_suffix: "Ejecutándolo en Python"
---

{{% youtube "uUMDo8HOcrI" %}}

Meta (Facebook) ha publicado una biblioteca de código abierto (OSS) para la predicción de series temporales. Puedes consultar las instrucciones de instalación en Python en [**Installation in Python**](https://facebook.github.io/prophet/docs/installation.html#python). Básicamente, solo necesitas ejecutar `pip install prophet` para instalarla.

{{% notice ref %}}
Taylor, Sean J., y Benjamin Letham. "Forecasting at scale." The American Statistician 72.1 (2018): 37-45.
{{% /notice %}}


```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib
from prophet import Prophet
```

## Datos utilizados para el experimento
Se utilizarán datos de un año. Los meses pares tienden a mostrar una disminución en los valores. Además, los datos presentan un comportamiento cíclico semanalmente. 

El período abarca desde el 01/01/2020 hasta el 31/12/2020.

```python
# Crear un rango de fechas diarias para un año
date = pd.date_range("2020-01-01", periods=365, freq="D")

# Generar valores objetivo para la predicción
y = [
    np.cos(di.weekday())  # Componente semanal cíclico
    + di.month % 2 / 2    # Componente estacional mensual
    + np.log(i + 1) / 3.0 # Componente de tendencia
    + np.random.rand() / 10  # Ruido aleatorio
    for i, di in enumerate(date)
]

# Crear componentes individuales
x = [18627 + i - 364 for i in range(365)]  # Línea de tendencia
trend_y = [np.log(i + 1) / 3.0 for i, di in enumerate(date)]  # Componente de tendencia
weekly_y = [np.cos(di.weekday()) for i, di in enumerate(date)]  # Componente semanal
seasonal_y = [di.month % 2 / 2 for i, di in enumerate(date)]  # Componente estacional
noise_y = [np.random.rand() / 10 for i in range(365)]  # Ruido aleatorio

# Crear un DataFrame para los datos
df = pd.DataFrame({"ds": date, "y": y})
df.index = date

# Visualizar los datos simulados
plt.title("Datos de ejemplo")
sns.lineplot(data=df)  # Gráfica de línea de los datos
plt.show()
```


    
![png](/images/timeseries/forecast/001-Prophet_files/001-Prophet_5_0.png)
    

## Componentes de los datos de series temporales
El término "datos de series temporales" incluye diferentes tipos de datos. Aquí nos centraremos en los siguientes casos:

- Los datos contienen solo marcas de tiempo y valores numéricos.
- Las marcas de tiempo no tienen valores faltantes y están espaciadas de manera uniforme (evenly spaced).

### Visualización de los componentes de los datos

```python
plt.figure(figsize=(14, 6))
plt.title("Descomposición de y en sus componentes")
# Componente de tendencia
plt.subplot(511)
plt.plot(x, trend_y, "-.", color="red", label="Tendencia", alpha=0.9)

# Componente cíclico semanal
plt.subplot(512)
plt.plot(x, weekly_y, "-.", color="green", label="Variación cíclica (semanal)", alpha=0.9)

# Componente estacional mensual
plt.subplot(513)
plt.plot(x, seasonal_y, "-.", color="orange", label="Variación cíclica (mensual)", alpha=0.9)

# Componente de ruido
plt.subplot(514)
```


    
![png](/images/timeseries/forecast/001-Prophet_files/001-Prophet_7_0.png)
    

## Predicción con Prophet para el período de enero a marzo de 2021
Utilizaremos los datos desde el 1 de enero de 2020 hasta el 31 de diciembre de 2020 para predecir los próximos tres meses (enero a marzo de 2021). Dado que solo contamos con un año de datos, desactivaremos la estacionalidad anual (`yearly_seasonality=False`). Sin embargo, dado que se observa periodicidad semanal, habilitaremos la estacionalidad diaria (`daily_seasonality=True`).

### Código para entrenar y predecir con Prophet

```python
def train_and_forecast_pf(
    data,
    periods=90,
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=True,
):
    """Entrena y realiza predicciones con Prophet

    Args:
        data (pandas.DataFrame): Datos de series temporales.
        periods (int, optional): Duración del período a predecir. Por defecto, 90 días.
        yearly_seasonality (bool, optional): Indica si hay estacionalidad anual. Por defecto, False.
        weekly_seasonality (bool, optional): Indica si hay estacionalidad semanal. Por defecto, True.
        daily_seasonality (bool, optional): Indica si hay estacionalidad diaria. Por defecto, True.

    Returns:
        _type_: Modelo entrenado y resultados de la predicción.
    """
    # Verificar que los datos contienen las columnas requeridas
    assert "ds" in data.columns and "y" in data.columns, "Los datos deben contener las columnas 'ds' y 'y'."

    # Entrenar el modelo Prophet
    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
    )
    m.fit(data)

    # Crear el DataFrame para el futuro y realizar la predicción
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return m, forecast
```


```python
# Comprueba los resultados de las previsiones
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
    


### Impacto de la especificación de periodicidad
```{warning}
Si se especifica una estacionalidad que no existe en los datos, podría generar predicciones incorrectas.
```

En el siguiente ejemplo, se fuerza la especificación de estacionalidad anual (yearly_seasonality=True).
Como resultado, debido al término añadido para capturar ciclos anuales, se observa un aumento algo inusual en las predicciones para 2022.


```python
# Comprueba los resultados de las previsiones
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
    

