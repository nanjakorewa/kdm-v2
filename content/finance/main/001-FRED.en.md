---
title: FRED
weight: 1
pre: "<b>7.1.1 </b>"
not_use_colab: true
searchtitle: "Working with FRED data API in Python!"
---

{{% youtube "hGTfTueJ328" %}}

```python
import os
import pandas as pd
import seaborn as sns
from full_fred.fred import Fred

# FRED_API_KEY = os.getenv('FRED_API_KEY')
fred = Fred()
print(f"FRED API key is set to an environment variable: {fred.env_api_key_found()}")


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

    FRED API key is set to an environment variable:True


### Unemployment Rate

Original source：https://fred.stlouisfed.org/series/UNRATE


```python
df_UNRATE = get_fred_data("UNRATE", start="2010-01-01")
df_UNRATE.head(5)
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
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01</th>
      <td>9.8</td>
    </tr>
    <tr>
      <th>2010-02-01</th>
      <td>9.8</td>
    </tr>
    <tr>
      <th>2010-03-01</th>
      <td>9.9</td>
    </tr>
    <tr>
      <th>2010-04-01</th>
      <td>9.9</td>
    </tr>
    <tr>
      <th>2010-05-01</th>
      <td>9.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(rc={"figure.figsize": (15, 8)})
sns.lineplot(data=df_UNRATE, x="date", y="value")
```




    
![png](/images/finance/main/001-FRED_files/001-FRED_3_1.png)
    


### ICE BofA US High Yield Index Option-Adjusted Spread

Source of data:https://fred.stlouisfed.org/series/BAMLH0A0HYM2


```python
df_BAMLH0A0HYM2 = get_fred_data("BAMLH0A0HYM2", start="2010-01-01")
df_BAMLH0A0HYM2.head(5)
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
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-04</th>
      <td>6.34</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>6.30</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>6.17</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>6.03</td>
    </tr>
  </tbody>
</table>
</div>



### Wilshire US Large-Cap Growth Total Market Index

Source of data:https://fred.stlouisfed.org/series/WILLLRGCAPGR


```python
df_WILLLRGCAPGR = get_fred_data("WILLLRGCAPGR", start="2010-01-01")
df_WILLLRGCAPGR.head(5)
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
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-01</th>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2010-01-04</th>
      <td>26423.03</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>26494.24</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>26473.92</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>26512.36</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.set(rc={"figure.figsize": (15, 8)})
sns.lineplot(data=df_WILLLRGCAPGR, x="date", y="value")
sns.lineplot(data=df_BAMLH0A0HYM2 * 10000, x="date", y="value")
```



    
![png](/images/finance/main/001-FRED_files/001-FRED_8_1.png)
    

