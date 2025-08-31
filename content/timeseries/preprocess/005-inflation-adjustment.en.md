---
title: Adjustment
weight: 6
pre: "<b>5.1.5 </b>"
searchtitle: "Adjusting time series data for inflation using CPI"
---

Sometimes data affected by the value of money need to be adjusted before analysis.
Here we will use the Consumer Price Index (CPI) to adjust time series data for inflation. Specifically, in this page, I'll try to...

- Obtain CPI data from FRED using the python API
- Obtain median household income data for the U.S.
- Adjust the U.S. median household income data using the CPI
- Compare the median household income data (adjusted) in the U.S. with the data adjusted using the CPI.

{{% notice document %}}
[A Python library that quickly adjusts U.S. dollars for inflation using the Consumer Price Index (CPI).](https://palewi.re/docs/cpi/)
{{% /notice %}}

```python
import os
import cpi
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from full_fred.fred import Fred
from datetime import date


cpi.update()  # Make sure you do it
```

## Get data from FRED

Obtain the necessary data from [FRED](https://fred.stlouisfed.org/). The method of data acquisition is explained in a separate video, but it is necessary to issue an API key to access the data from python.


```python
# FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred()
print(f"FRED API key is set to an environment variable：{fred.env_api_key_found()}")


def get_fred_data(name, start="2013-01-01", end=""):
    df = fred.get_series_df(name)[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")

    if end == "":
        df = df.loc[f"{start}":]
    else:
        df = df.loc[f"{start}":f"{end}"]

    return df
```

    FRED API key is set to an environment variable：True


## Household Income in the United States

Let's check the data for [Median Household Income in the United States (MEHOINUSA646N)](https://fred.stlouisfed.org/series/MEHOINUSA646N). This data is in units of

> Units:  Current Dollars, Not Seasonally Adjusted

and this data is not adjusted for inflation or seasonally adjusted.
We adjust this household data using the inflation rate.


```python
data = get_fred_data("MEHOINUSA646N", start="2013-01-01", end="")
data.head()
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
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>53585</td>
    </tr>
    <tr>
      <th>2014-01-01</th>
      <td>53657</td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>56516</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>59039</td>
    </tr>
    <tr>
      <th>2017-01-01</th>
      <td>61136</td>
    </tr>
  </tbody>
</table>
</div>



### Adjust values using CPI


```python
data["adjusted"] = [
    cpi.inflate(dollers.value, date.year) for date, dollers in data.iterrows()
]
```

### Comparison before and after adjustment


```python
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=data,
    x="date",
    y="value",
    label="Current Dollars, Not Seasonally Adjusted(MEHOINUSA646N)",
)
sns.lineplot(data=data, x="date", y="adjusted", label="MEHOINUSA646N adjusted")
plt.legend()
```


    
![png](/images/timeseries/preprocess/005-inflation-adjustment_files/005-inflation-adjustment_9_1.png)
    


## Real Median Household Income in the United States

Adjusted data are provided in [Real Median Household Income in the United States (MEHOINUSA672N)](https://fred.stlouisfed.org/series/MEHOINUSA672N). Compare the values in MEHOINUSA672N with the inflation-adjusted data (values in the `adjusted` column in the `data`) from earlier. The unadjusted values, adjusted using the Consumer Price Index (CPI), are expected to match the adjusted values (MEHOINUSA672N) almost exactly.


```python
data_real = get_fred_data("MEHOINUSA672N", start="2013-01-01", end="")
data_real.head()
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
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>62425</td>
    </tr>
    <tr>
      <th>2014-01-01</th>
      <td>61468</td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>64631</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>66657</td>
    </tr>
    <tr>
      <th>2017-01-01</th>
      <td>67571</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_real, x="date", y="value", label="MEHOINUSA672N")
sns.lineplot(data=data, x="date", y="adjusted", label="MEHOINUSA646N adjusted")

for t, v in data_real.iterrows():
    plt.text(t, v[0] - 500, f"{v[0]:.2f}")

for t, v in data.iterrows():
    plt.text(t, v[1] + 500, f"{v[1]:.2f}")

plt.legend()
```


    
![png](/images/timeseries/preprocess/005-inflation-adjustment_files/005-inflation-adjustment_12_1.png)
    

