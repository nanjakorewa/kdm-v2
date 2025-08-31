---
title: "ETFと利回りを比較"
pre: "7.1.6 "
weight: 6
not_use_colab: true
---
<div class="pagetop-box">
代表的なETFと債権利回りの推移を比較します。データは <a href="https://stooq.pl/">stooq</a>から取得したものを取得しています。

『<a href="https://elf-c.he.u-tokyo.ac.jp/courses/466/files/11157/download?download_frd=1">メディアプログラミング入門 WebスクレイピングとWebAPI</a>』の講義資料45pで紹介されていたPandas Datareaderを使用しています。
</div>

```python
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import japanize_matplotlib
from IPython.display import display, HTML
from utils import get_finance_data, get_rsi
```
## ETF
{{% notice document %}}
[pandas.DataFrame.sort_index](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sort_index.html)
{{% /notice %}}


```python
# 銘柄名、期間、保存先ファイル
start = "2021-08-01"
end = "2022-01-31"

# データを取得する
voo = get_finance_data("VOO", source="stooq", start=start, end=end)
display(HTML(f"<h3>VOO</h3>"))
display(voo.head())

vti = get_finance_data("VTI", source="stooq", start=start, end=end)
display(HTML(f"<h3>VTI</h3>"))
display(vti.head())

spx = get_finance_data("^SPX", source="stooq", start=start, end=end)
display(HTML(f"<h3>S&P500</h3>"))
display(spx.head())

ndq = get_finance_data("^NDQ", source="stooq", start=start, end=end)
display(HTML(f"<h3>Nasdaq</h3>"))
display(ndq.head())
```


<h3>VOO</h3>



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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-31</th>
      <td>405.67</td>
      <td>413.9700</td>
      <td>404.3500</td>
      <td>413.69</td>
      <td>9200435</td>
    </tr>
    <tr>
      <th>2022-01-28</th>
      <td>397.82</td>
      <td>406.3300</td>
      <td>393.3000</td>
      <td>406.26</td>
      <td>12096807</td>
    </tr>
    <tr>
      <th>2022-01-27</th>
      <td>402.93</td>
      <td>405.9700</td>
      <td>394.8200</td>
      <td>396.54</td>
      <td>12455226</td>
    </tr>
    <tr>
      <th>2022-01-26</th>
      <td>405.66</td>
      <td>408.1954</td>
      <td>394.3400</td>
      <td>398.56</td>
      <td>14377346</td>
    </tr>
    <tr>
      <th>2022-01-25</th>
      <td>398.17</td>
      <td>404.2600</td>
      <td>392.7325</td>
      <td>399.46</td>
      <td>16836326</td>
    </tr>
  </tbody>
</table>
</div>



<h3>VTI</h3>



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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-31</th>
      <td>222.09</td>
      <td>226.950</td>
      <td>221.22</td>
      <td>226.81</td>
      <td>5306436</td>
    </tr>
    <tr>
      <th>2022-01-28</th>
      <td>217.50</td>
      <td>222.170</td>
      <td>214.93</td>
      <td>222.09</td>
      <td>4961209</td>
    </tr>
    <tr>
      <th>2022-01-27</th>
      <td>220.81</td>
      <td>222.400</td>
      <td>215.90</td>
      <td>216.75</td>
      <td>5791524</td>
    </tr>
    <tr>
      <th>2022-01-26</th>
      <td>222.72</td>
      <td>224.160</td>
      <td>216.05</td>
      <td>218.37</td>
      <td>7096416</td>
    </tr>
    <tr>
      <th>2022-01-25</th>
      <td>219.10</td>
      <td>222.015</td>
      <td>215.64</td>
      <td>219.22</td>
      <td>6600378</td>
    </tr>
  </tbody>
</table>
</div>



<h3>S&P500</h3>



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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-31</th>
      <td>4431.79</td>
      <td>4516.89</td>
      <td>4414.02</td>
      <td>4515.55</td>
      <td>2960132803</td>
    </tr>
    <tr>
      <th>2022-01-28</th>
      <td>4336.19</td>
      <td>4432.72</td>
      <td>4292.46</td>
      <td>4431.85</td>
      <td>2926083817</td>
    </tr>
    <tr>
      <th>2022-01-27</th>
      <td>4380.58</td>
      <td>4428.74</td>
      <td>4309.50</td>
      <td>4326.51</td>
      <td>3070684348</td>
    </tr>
    <tr>
      <th>2022-01-26</th>
      <td>4408.43</td>
      <td>4453.23</td>
      <td>4304.80</td>
      <td>4349.93</td>
      <td>3239353450</td>
    </tr>
    <tr>
      <th>2022-01-25</th>
      <td>4366.64</td>
      <td>4411.01</td>
      <td>4287.11</td>
      <td>4356.45</td>
      <td>3069079477</td>
    </tr>
  </tbody>
</table>
</div>



<h3>Nasdaq</h3>



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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-31</th>
      <td>13812.20</td>
      <td>14242.90</td>
      <td>13767.71</td>
      <td>14239.88</td>
      <td>3268652504</td>
    </tr>
    <tr>
      <th>2022-01-28</th>
      <td>13436.71</td>
      <td>13771.91</td>
      <td>13236.55</td>
      <td>13770.57</td>
      <td>3092819850</td>
    </tr>
    <tr>
      <th>2022-01-27</th>
      <td>13710.99</td>
      <td>13765.91</td>
      <td>13322.66</td>
      <td>13352.78</td>
      <td>3373437394</td>
    </tr>
    <tr>
      <th>2022-01-26</th>
      <td>13871.77</td>
      <td>14002.65</td>
      <td>13392.19</td>
      <td>13542.12</td>
      <td>3664304374</td>
    </tr>
    <tr>
      <th>2022-01-25</th>
      <td>13610.87</td>
      <td>13781.63</td>
      <td>13414.14</td>
      <td>13539.29</td>
      <td>3265336637</td>
    </tr>
  </tbody>
</table>
</div>


## 債権利回り


```python
usy10 = get_finance_data("10USY.B", source="stooq", start=start, end=end)
display(HTML(f"<h3>10-Year U.S. Bond Yield</h3>"))
display(usy10.head())

usy2 = get_finance_data("2USY.B", source="stooq", start=start, end=end)
display(HTML(f"<h3>2-Year U.S. Bond Yield</h3>"))
display(usy2.head())
```


<h3>10-Year U.S. Bond Yield</h3>



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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-31</th>
      <td>1.789</td>
      <td>1.816</td>
      <td>1.771</td>
      <td>1.780</td>
    </tr>
    <tr>
      <th>2022-01-28</th>
      <td>1.830</td>
      <td>1.848</td>
      <td>1.773</td>
      <td>1.777</td>
    </tr>
    <tr>
      <th>2022-01-27</th>
      <td>1.840</td>
      <td>1.851</td>
      <td>1.783</td>
      <td>1.799</td>
    </tr>
    <tr>
      <th>2022-01-26</th>
      <td>1.769</td>
      <td>1.876</td>
      <td>1.769</td>
      <td>1.867</td>
    </tr>
    <tr>
      <th>2022-01-25</th>
      <td>1.753</td>
      <td>1.792</td>
      <td>1.735</td>
      <td>1.776</td>
    </tr>
  </tbody>
</table>
</div>



<h3>2-Year U.S. Bond Yield</h3>



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
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-31</th>
      <td>1.1986</td>
      <td>1.2145</td>
      <td>1.1587</td>
      <td>1.1827</td>
    </tr>
    <tr>
      <th>2022-01-28</th>
      <td>1.2141</td>
      <td>1.2260</td>
      <td>1.1603</td>
      <td>1.1703</td>
    </tr>
    <tr>
      <th>2022-01-27</th>
      <td>1.1822</td>
      <td>1.2061</td>
      <td>1.1603</td>
      <td>1.1882</td>
    </tr>
    <tr>
      <th>2022-01-26</th>
      <td>1.0313</td>
      <td>1.1603</td>
      <td>1.0154</td>
      <td>1.1544</td>
    </tr>
    <tr>
      <th>2022-01-25</th>
      <td>1.0055</td>
      <td>1.0353</td>
      <td>0.9897</td>
      <td>1.0254</td>
    </tr>
  </tbody>
</table>
</div>



```python
plt.figure(figsize=(12, 4))
plt.plot(usy10.Close, label="米国債券10年 年利回り")
plt.plot(usy2.Close, label="米国債券2年 年利回り")
plt.legend()
plt.tick_params(rotation=90)
plt.grid()
plt.show()
```


    
![png](/images/finance/main/004_ETF_and_k-Year_U.S._Bond_Yield_files/004_ETF_and_k-Year_U.S._Bond_Yield_7_0.png)
    


## S&P500と１０年債利回りの比較
{{% notice document %}}
- [Invert Axes](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/invert_axes.html)
{{% /notice %}}


```python
import mplfinance as mpf

fig = mpf.figure(figsize=(16, 7), tight_layout=True, style="default")

ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
mpf.plot(spx, type="candle", style="yahoo", datetime_format="%Y/%m/%d", ax=ax1)
ax1.set_ylabel("S&P500")
ax1.invert_xaxis()

mpf.plot(
    usy10,
    type="line",
    style="starsandstripes",
    datetime_format="%Y/%m/%d",
    ax=ax2,
)

# 左側にラベルを表示する
ax2.tick_params(labelleft=True, labelright=False)
ax2.set_ylabel("10-Year U.S. Bond Yield")
ax2.yaxis.set_label_position("left")
ax2.legend(["10-Year U.S. Bond Yield"])
```




    <matplotlib.legend.Legend at 0x7fcc4de60970>




    
![png](/images/finance/main/004_ETF_and_k-Year_U.S._Bond_Yield_files/004_ETF_and_k-Year_U.S._Bond_Yield_9_1.png)
    

