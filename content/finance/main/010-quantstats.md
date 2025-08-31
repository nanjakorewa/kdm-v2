---
title: Quantstats
weight: 12
pre: "<b>7.1.12 </b>"
not_use_colab: true
---

{{% youtube "ziCCwGq7EuM" %}}

<div class="pagetop-box">
    <p>QuantStatsは特定の銘柄や複数銘柄のポートフォリオに対し種々のリスクメトリクスやパフォーマンスの分析を簡単に実行できるpythonライブラリです。以下のコードでは、GOOGとVTIの比較、そして複数銘柄のポートフォリオとVTIの比較をするレポートをQuantStatsを使って作成します。</p>
</div>


{{% notice ref %}}
[ranaroussi/quantstats: Portfolio analytics for quants](https://github.com/ranaroussi/quantstats)
{{% /notice %}}


```python
import numpy as np
import pandas as pd
import quantstats as qs
import yfinance as yf

qs.extend_pandas()
```

## VTIと特定の銘柄の比較レポート


```python
GOOG_returns = qs.utils.download_returns("GOOG", period="2y")
GOOG_returns.head(10)
qs.stats.sharpe(GOOG_returns)
qs.plots.snapshot(GOOG_returns, title="GOOG Performance")
qs.reports.html(GOOG_returns, "VTI", download_filename="GOOGとVTIの比較.html")
```

![png](/images/finance/main/010-quantstats_files/010-quantstats_3_0.png)

## 複数銘柄のパフォーマンス比較レポート
すべての銘柄を等しい割合で保有していた場合のリターンを、SP500と比較して見ます。
銘柄のオアフォーマンスを計算するには株価の変化率の系列が必要なので、`pandas.DataFrame.pct_change` を用いて変化率の系列を求めています。

{{% notice document %}}
[pandas.DataFrame.pct_change](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pct_change.html)
{{% /notice %}}


```python
# ポートフォリオ内での各銘柄の比重
stock_dict = {
    "MSFT": 0.3,
    "AAPL": 0.3,
    "AMZN": 0.2,
    "GOOG": 0.1,
    "TSLA": 0.1,
}

# 株価の系列
stock_prices_df = yf.download(
    list(stock_dict.keys()), start="2021-01-01", end="2022-01-01", adjusted=True
).dropna()

stock_prices_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">Adj Close</th>
      <th colspan="5" halign="left">Close</th>
      <th>...</th>
      <th colspan="5" halign="left">Open</th>
      <th colspan="5" halign="left">Volume</th>
    </tr>
    <tr>
      <th></th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>TSLA</th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>TSLA</th>
      <th>...</th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>TSLA</th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>TSLA</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-12-31</th>
      <td>131.333542</td>
      <td>162.846497</td>
      <td>87.594002</td>
      <td>219.139343</td>
      <td>235.223328</td>
      <td>132.690002</td>
      <td>162.846497</td>
      <td>87.594002</td>
      <td>222.419998</td>
      <td>235.223328</td>
      <td>...</td>
      <td>134.080002</td>
      <td>163.750000</td>
      <td>86.771004</td>
      <td>221.699997</td>
      <td>233.330002</td>
      <td>99116600</td>
      <td>59144000</td>
      <td>20238000</td>
      <td>20942100</td>
      <td>148949700</td>
    </tr>
    <tr>
      <th>2021-01-04</th>
      <td>128.087051</td>
      <td>159.331497</td>
      <td>86.412003</td>
      <td>214.479111</td>
      <td>243.256668</td>
      <td>129.410004</td>
      <td>159.331497</td>
      <td>86.412003</td>
      <td>217.690002</td>
      <td>243.256668</td>
      <td>...</td>
      <td>133.520004</td>
      <td>163.500000</td>
      <td>87.876999</td>
      <td>222.529999</td>
      <td>239.820007</td>
      <td>143301900</td>
      <td>88228000</td>
      <td>38038000</td>
      <td>37130100</td>
      <td>145914600</td>
    </tr>
    <tr>
      <th>2021-01-05</th>
      <td>129.670715</td>
      <td>160.925507</td>
      <td>87.045998</td>
      <td>214.685989</td>
      <td>245.036667</td>
      <td>131.009995</td>
      <td>160.925507</td>
      <td>87.045998</td>
      <td>217.899994</td>
      <td>245.036667</td>
      <td>...</td>
      <td>128.889999</td>
      <td>158.300507</td>
      <td>86.250000</td>
      <td>217.259995</td>
      <td>241.220001</td>
      <td>97664900</td>
      <td>53110000</td>
      <td>22906000</td>
      <td>23823000</td>
      <td>96735600</td>
    </tr>
    <tr>
      <th>2021-01-06</th>
      <td>125.305786</td>
      <td>156.919006</td>
      <td>86.764503</td>
      <td>209.119339</td>
      <td>251.993332</td>
      <td>126.599998</td>
      <td>156.919006</td>
      <td>86.764503</td>
      <td>212.250000</td>
      <td>251.993332</td>
      <td>...</td>
      <td>127.720001</td>
      <td>157.324005</td>
      <td>85.131500</td>
      <td>212.169998</td>
      <td>252.830002</td>
      <td>155088000</td>
      <td>87896000</td>
      <td>52042000</td>
      <td>35930700</td>
      <td>134100000</td>
    </tr>
    <tr>
      <th>2021-01-07</th>
      <td>129.581635</td>
      <td>158.108002</td>
      <td>89.362503</td>
      <td>215.070267</td>
      <td>272.013336</td>
      <td>130.919998</td>
      <td>158.108002</td>
      <td>89.362503</td>
      <td>218.289993</td>
      <td>272.013336</td>
      <td>...</td>
      <td>128.360001</td>
      <td>157.850006</td>
      <td>87.002998</td>
      <td>214.039993</td>
      <td>259.209991</td>
      <td>109578200</td>
      <td>70290000</td>
      <td>45300000</td>
      <td>27694500</td>
      <td>154496700</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>





```python
returns_df = stock_prices_df["Adj Close"].pct_change().dropna()
returns_df.head()
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
      <th>AAPL</th>
      <th>AMZN</th>
      <th>GOOG</th>
      <th>MSFT</th>
      <th>TSLA</th>
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
      <th>2021-01-04</th>
      <td>-0.024719</td>
      <td>-0.021585</td>
      <td>-0.013494</td>
      <td>-0.021266</td>
      <td>0.034152</td>
    </tr>
    <tr>
      <th>2021-01-05</th>
      <td>0.012364</td>
      <td>0.010004</td>
      <td>0.007337</td>
      <td>0.000965</td>
      <td>0.007317</td>
    </tr>
    <tr>
      <th>2021-01-06</th>
      <td>-0.033662</td>
      <td>-0.024897</td>
      <td>-0.003234</td>
      <td>-0.025929</td>
      <td>0.028390</td>
    </tr>
    <tr>
      <th>2021-01-07</th>
      <td>0.034123</td>
      <td>0.007577</td>
      <td>0.029943</td>
      <td>0.028457</td>
      <td>0.079447</td>
    </tr>
    <tr>
      <th>2021-01-08</th>
      <td>0.008631</td>
      <td>0.006496</td>
      <td>0.011168</td>
      <td>0.006093</td>
      <td>0.078403</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5 columns</p>
</div>




```python
pf_returns = np.dot(list(stock_dict.values()), returns_df.T)
print(pf_returns[:10])
```

    [-0.01530147  0.00900607 -0.01796814  0.02928912  0.0152212  -0.02670266
      0.00147395  0.01133958 -0.01243583 -0.00920841]
    


```python
pf_returns_series = pd.Series(pf_returns, index=returns_df.index)

pf_returns_series.head(10)
```




    Date
    2021-01-04   -0.015301
    2021-01-05    0.009006
    2021-01-06   -0.017968
    2021-01-07    0.029289
    2021-01-08    0.015221
    2021-01-11   -0.026703
    2021-01-12    0.001474
    2021-01-13    0.011340
    2021-01-14   -0.012436
    2021-01-15   -0.009208
    dtype: float64




```python
qs.reports.html(pf_returns_series, "VTI", download_filename="ポートフォリオのパフォーマンス.html")
```
