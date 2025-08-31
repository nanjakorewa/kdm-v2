---
title: "J-Quants API"
pre: " 7.4.4 "
weight: 32
searchtitle: "pythonでJ-Quants APIを使い日本企業のEPSを比較してみる"
---

{{% youtube "ObXT8X-NmMM" %}}

## J-Quants APIを使ってみる
- [J-Quants](https://jpx-jquants.com/dashboard/menu/?lang=ja)
- [jquants-api-client](https://github.com/J-Quants/jquants-api-client-python)
- [API仕様書](https://jpx.gitbook.io/j-quants-ja/api-reference)


```python
import os
import time
from datetime import datetime

import jquantsapi
import pandas as pd
import requests
from dateutil import tz

REFRESH_TOKEN: str = os.environ.get("JQ_REFRESH_TOKEN")
my_mail_address: str = os.environ.get("JQ_MAIL_ADDRESS")
my_password: str = os.environ.get("JQ_PASSWORD")
cli = jquantsapi.Client(mail_address=my_mail_address, password=my_password)
df = cli.get_price_range(
    start_dt=datetime(2022, 7, 25, tzinfo=tz.gettz("Asia/Tokyo")),
    end_dt=datetime(2022, 7, 26, tzinfo=tz.gettz("Asia/Tokyo")),
)
print(df)
```

               Date   Code    Open    High     Low   Close UpperLimit LowerLimit  \
    0    2022-07-25  13010  3615.0  3660.0  3615.0  3630.0          0          0   
    0    2022-07-26  13010  3615.0  3640.0  3610.0  3635.0          0          0   
    1    2022-07-25  13050  2026.5  2037.0  2022.0  2023.0          0          0   
    1    2022-07-26  13050  2026.0  2029.5  2022.0  2023.5          0          0   
    2    2022-07-25  13060  2002.5  2015.0  2000.0  2001.0          0          0   
    ...         ...    ...     ...     ...     ...     ...        ...        ...   
    4191 2022-07-26  99950   403.0   404.0   402.0   404.0          0          0   
    4192 2022-07-25  99960  1274.0  1274.0  1263.0  1267.0          0          0   
    4192 2022-07-26  99960  1254.0  1266.0  1254.0  1255.0          0          0   
    4193 2022-07-25  99970   829.0   831.0   816.0   826.0          0          0   
    4193 2022-07-26  99970   826.0   827.0   816.0   825.0          0          0   
    
            Volume  TurnoverValue  AdjustmentFactor  AdjustmentOpen  \
    0       8100.0   2.942050e+07               1.0          3615.0   
    0       8500.0   3.083550e+07               1.0          3615.0   
    1      54410.0   1.103787e+08               1.0          2026.5   
    1      22950.0   4.646586e+07               1.0          2026.0   
    2     943830.0   1.891360e+09               1.0          2002.5   
    ...        ...            ...               ...             ...   
    4191   13000.0   5.240900e+06               1.0           403.0   
    4192    1500.0   1.902700e+06               1.0          1274.0   
    4192    4000.0   5.021300e+06               1.0          1254.0   
    4193  151200.0   1.245601e+08               1.0           829.0   
    4193  133600.0   1.099946e+08               1.0           826.0   
    
          AdjustmentHigh  AdjustmentLow  AdjustmentClose  AdjustmentVolume  
    0             3660.0         3615.0           3630.0            8100.0  
    0             3640.0         3610.0           3635.0            8500.0  
    1             2037.0         2022.0           2023.0           54410.0  
    1             2029.5         2022.0           2023.5           22950.0  
    2             2015.0         2000.0           2001.0          943830.0  
    ...              ...            ...              ...               ...  
    4191           404.0          402.0            404.0           13000.0  
    4192          1274.0         1263.0           1267.0            1500.0  
    4192          1266.0         1254.0           1255.0            4000.0  
    4193           831.0          816.0            826.0          151200.0  
    4193           827.0          816.0            825.0          133600.0  
    
    [8388 rows x 16 columns]
    

## 上場銘柄一覧(/listed/info)
- [J-Quants API - listed info](https://jpx.gitbook.io/j-quants-ja/api-reference/listed_info)


```python
def get_listed_companies(idToken: str):
    """上場銘柄一覧を習得する

    Args:
        idToken (str): idToken

    Returns:
        listed_companies (pd.DataFrame): 上場銘柄が記録されたデータフレーム
    """
    r = requests.get(
        "https://api.jquants.com/v1/listed/info",
        headers={"Authorization": "Bearer {}".format(idToken)},
    )
    if r.status_code == requests.codes.ok:
        listed_companies = pd.DataFrame(r.json()["info"]).set_index("Code")
        return listed_companies
    else:
        return None
```


```python
r_post = requests.post(
    f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={REFRESH_TOKEN}"
)
idToken = r_post.json()["idToken"]

listed_companies = get_listed_companies(idToken)
listed_companies.head()
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
      <th>Date</th>
      <th>CompanyName</th>
      <th>CompanyNameEnglish</th>
      <th>Sector17Code</th>
      <th>Sector17CodeName</th>
      <th>Sector33Code</th>
      <th>Sector33CodeName</th>
      <th>ScaleCategory</th>
      <th>MarketCode</th>
      <th>MarketCodeName</th>
    </tr>
    <tr>
      <th>Code</th>
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
      <th>13010</th>
      <td>2023-12-06</td>
      <td>極洋</td>
      <td>KYOKUYO CO.,LTD.</td>
      <td>1</td>
      <td>食品</td>
      <td>0050</td>
      <td>水産・農林業</td>
      <td>TOPIX Small 2</td>
      <td>0111</td>
      <td>プライム</td>
    </tr>
    <tr>
      <th>13050</th>
      <td>2023-12-06</td>
      <td>大和アセットマネジメント株式会社　ｉＦｒｅｅＥＴＦ　ＴＯＰＩＸ（年１回決算型）</td>
      <td>iFreeETF TOPIX (Yearly Dividend Type)</td>
      <td>99</td>
      <td>その他</td>
      <td>9999</td>
      <td>その他</td>
      <td>-</td>
      <td>0109</td>
      <td>その他</td>
    </tr>
    <tr>
      <th>13060</th>
      <td>2023-12-06</td>
      <td>野村アセットマネジメント株式会社　ＮＥＸＴ　ＦＵＮＤＳ　ＴＯＰＩＸ連動型上場投信</td>
      <td>NEXT FUNDS TOPIX Exchange Traded Fund</td>
      <td>99</td>
      <td>その他</td>
      <td>9999</td>
      <td>その他</td>
      <td>-</td>
      <td>0109</td>
      <td>その他</td>
    </tr>
    <tr>
      <th>13080</th>
      <td>2023-12-06</td>
      <td>日興アセットマネジメント株式会社　　上場インデックスファンドＴＯＰＩＸ</td>
      <td>Nikko Exchange Traded Index Fund TOPIX</td>
      <td>99</td>
      <td>その他</td>
      <td>9999</td>
      <td>その他</td>
      <td>-</td>
      <td>0109</td>
      <td>その他</td>
    </tr>
    <tr>
      <th>13090</th>
      <td>2023-12-06</td>
      <td>野村アセットマネジメント株式会社　ＮＥＸＴ　ＦＵＮＤＳ　ＣｈｉｎａＡＭＣ・中国株式・上証５０...</td>
      <td>NEXT FUNDS ChinaAMC SSE50 Index Exchange Trade...</td>
      <td>99</td>
      <td>その他</td>
      <td>9999</td>
      <td>その他</td>
      <td>-</td>
      <td>0109</td>
      <td>その他</td>
    </tr>
  </tbody>
</table>
</div>




```python
paint_companies = listed_companies[
    listed_companies["CompanyName"].str.contains("塗料|ペイント")
]
paint_companies
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
      <th>Date</th>
      <th>CompanyName</th>
      <th>CompanyNameEnglish</th>
      <th>Sector17Code</th>
      <th>Sector17CodeName</th>
      <th>Sector33Code</th>
      <th>Sector33CodeName</th>
      <th>ScaleCategory</th>
      <th>MarketCode</th>
      <th>MarketCodeName</th>
    </tr>
    <tr>
      <th>Code</th>
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
      <th>46110</th>
      <td>2023-12-06</td>
      <td>大日本塗料</td>
      <td>Dai Nippon Toryo Company,Limited</td>
      <td>4</td>
      <td>素材・化学</td>
      <td>3200</td>
      <td>化学</td>
      <td>TOPIX Small 2</td>
      <td>0111</td>
      <td>プライム</td>
    </tr>
    <tr>
      <th>46120</th>
      <td>2023-12-06</td>
      <td>日本ペイントホールディングス</td>
      <td>NIPPON PAINT HOLDINGS CO.,LTD.</td>
      <td>4</td>
      <td>素材・化学</td>
      <td>3200</td>
      <td>化学</td>
      <td>TOPIX Mid400</td>
      <td>0111</td>
      <td>プライム</td>
    </tr>
    <tr>
      <th>46130</th>
      <td>2023-12-06</td>
      <td>関西ペイント</td>
      <td>KANSAI PAINT CO.,LTD.</td>
      <td>4</td>
      <td>素材・化学</td>
      <td>3200</td>
      <td>化学</td>
      <td>TOPIX Mid400</td>
      <td>0111</td>
      <td>プライム</td>
    </tr>
    <tr>
      <th>46150</th>
      <td>2023-12-06</td>
      <td>神東塗料</td>
      <td>SHINTO PAINT COMPANY,LIMITED</td>
      <td>4</td>
      <td>素材・化学</td>
      <td>3200</td>
      <td>化学</td>
      <td>TOPIX Small 2</td>
      <td>0112</td>
      <td>スタンダード</td>
    </tr>
    <tr>
      <th>46160</th>
      <td>2023-12-06</td>
      <td>川上塗料</td>
      <td>KAWAKAMI PAINT MANUFACTURING CO.,LTD.</td>
      <td>4</td>
      <td>素材・化学</td>
      <td>3200</td>
      <td>化学</td>
      <td>-</td>
      <td>0112</td>
      <td>スタンダード</td>
    </tr>
    <tr>
      <th>46170</th>
      <td>2023-12-06</td>
      <td>中国塗料</td>
      <td>Chugoku Marine Paints,Ltd.</td>
      <td>4</td>
      <td>素材・化学</td>
      <td>3200</td>
      <td>化学</td>
      <td>TOPIX Small 1</td>
      <td>0111</td>
      <td>プライム</td>
    </tr>
    <tr>
      <th>46190</th>
      <td>2023-12-06</td>
      <td>日本特殊塗料</td>
      <td>Nihon Tokushu Toryo Co.,Ltd.</td>
      <td>4</td>
      <td>素材・化学</td>
      <td>3200</td>
      <td>化学</td>
      <td>TOPIX Small 2</td>
      <td>0112</td>
      <td>スタンダード</td>
    </tr>
    <tr>
      <th>46210</th>
      <td>2023-12-06</td>
      <td>ロックペイント</td>
      <td>ROCK PAINT CO.,LTD.</td>
      <td>4</td>
      <td>素材・化学</td>
      <td>3200</td>
      <td>化学</td>
      <td>-</td>
      <td>0112</td>
      <td>スタンダード</td>
    </tr>
    <tr>
      <th>46240</th>
      <td>2023-12-06</td>
      <td>イサム塗料</td>
      <td>Isamu Paint Co.,Ltd.</td>
      <td>4</td>
      <td>素材・化学</td>
      <td>3200</td>
      <td>化学</td>
      <td>-</td>
      <td>0112</td>
      <td>スタンダード</td>
    </tr>
  </tbody>
</table>
</div>



## セクターの分布
- [東証業種別株価指数・TOPIX-17シリーズ](https://www.jpx.co.jp/markets/indices/line-up/files/fac_13_sector.pdf)
- [matplotlib.pyplot.pie — Matplotlib 3.8.3 documentation](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html)


```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sector17_distribution = listed_companies["Sector17CodeName"].value_counts()
colors = sns.color_palette("Set2")

plt.figure(figsize=(10, 10))
plt.pie(
    sector17_distribution,
    labels=sector17_distribution.index,
    colors=colors,
    autopct="%.0f%%",
)
plt.show()
```


    
![png](/images/finance/misc/jquants_files/jquants_7_0.png)
    


## 規模コード
- [東証規模別株価指数](https://www.jpx.co.jp/markets/indices/line-up/files/fac_12_size.pdf)
- [上場銘柄一覧(/listed/info)](https://jpx.gitbook.io/j-quants-ja/api-reference/listed_info)


```python
import plotly.express as px

midcap_categories = ["TOPIX Mid400", "TOPIX Large70", "TOPIX Core30"]

df = px.data.tips()
fig = px.treemap(
    listed_companies[listed_companies["ScaleCategory"].isin(midcap_categories)],
    path=["Sector17CodeName", "Sector33CodeName"],
)
fig.show()

df = px.data.tips()
fig = px.treemap(
    listed_companies[listed_companies["ScaleCategory"].isin(midcap_categories)],
    path=["Sector17CodeName", "Sector33CodeName", "ScaleCategory"],
)
fig.show()
```





## 財務情報の取得
- [財務情報(/fins/statements)](https://jpx.gitbook.io/j-quants-ja/api-reference/statements)
- [Ⅰ．インターフェイス仕様](https://www.jpx.co.jp/markets/paid-info-equities/listing/co3pgt0000005o97-att/tdnetapi_specifications.pdf)


```python
def get_statements(idToken, code):
    time.sleep(0.5)
    r = requests.get(
        f"https://api.jquants.com/v1/fins/statements?code={code}",
        headers={"Authorization": "Bearer {}".format(idToken)},
    )
    if r.status_code == requests.codes.ok:
        return r.json()["statements"]
    else:
        return None
```


```python
res = get_statements(idToken, 86970)
res
```




    [{'DisclosedDate': '2022-01-27',
      'DisclosedTime': '12:00:00',
      'LocalCode': '86970',
      'DisclosureNumber': '20220126573026',
      'TypeOfDocument': '3QFinancialStatements_Consolidated_IFRS',
      'TypeOfCurrentPeriod': '3Q',
      'CurrentPeriodStartDate': '2021-04-01',
      'CurrentPeriodEndDate': '2021-12-31',
      'CurrentFiscalYearStartDate': '2021-04-01',
      'CurrentFiscalYearEndDate': '2022-03-31',
      'NextFiscalYearStartDate': '',
      'NextFiscalYearEndDate': '',
      'NetSales': '100586000000',
      'OperatingProfit': '55967000000',
      'OrdinaryProfit': '',
      'Profit': '38013000000',
      'EarningsPerShare': '71.71',
      'DilutedEarningsPerShare': '',
      'TotalAssets': '62076519000000',
      'Equity': '311381000000',
      ...
      'NonConsolidatedProfit': '',
      'NonConsolidatedEarningsPerShare': '',
      'NonConsolidatedTotalAssets': '',
      'NonConsolidatedEquity': '',
      'NonConsolidatedEquityToAssetRatio': '',
      'NonConsolidatedBookValuePerShare': '',
      'ForecastNonConsolidatedNetSales2ndQuarter': '',
      'ForecastNonConsolidatedOperatingProfit2ndQuarter': '',
      'ForecastNonConsolidatedOrdinaryProfit2ndQuarter': '',
      'ForecastNonConsolidatedProfit2ndQuarter': '',
      'ForecastNonConsolidatedEarningsPerShare2ndQuarter': '',
      'NextYearForecastNonConsolidatedNetSales2ndQuarter': '',
      'NextYearForecastNonConsolidatedOperatingProfit2ndQuarter': '',
      'NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter': '',
      'NextYearForecastNonConsolidatedProfit2ndQuarter': '',
      'NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter': '',
      'ForecastNonConsolidatedNetSales': '',
      'ForecastNonConsolidatedOperatingProfit': '',
      'ForecastNonConsolidatedOrdinaryProfit': '',
      'ForecastNonConsolidatedProfit': '',
      'ForecastNonConsolidatedEarningsPerShare': '',
      'NextYearForecastNonConsolidatedNetSales': '',
      'NextYearForecastNonConsolidatedOperatingProfit': '',
      'NextYearForecastNonConsolidatedOrdinaryProfit': '',
      'NextYearForecastNonConsolidatedProfit': '',
      'NextYearForecastNonConsolidatedEarningsPerShare': ''}]




```python
paint_companies_statements = pd.concat(
    [pd.DataFrame(get_statements(idToken, code)) for code in paint_companies.index]
)

for c in paint_companies_statements.filter(
    regex="Sales|Assets|CashFlows|Profit|Equity|EarningsPerShare"
).columns:
    paint_companies_statements[c] = pd.to_numeric(paint_companies_statements[c])

for c in paint_companies_statements.filter(regex="Date").columns:
    paint_companies_statements[c] = pd.to_datetime(paint_companies_statements[c])
```


```python
sorted_data = paint_companies_statements.groupby(["LocalCode"]).apply(
    lambda x: x.sort_values(["DisclosureNumber"], ascending=False)
)
sorted_data["決算期"] = sorted_data.apply(
    lambda row: f"{row['DisclosedDate'].year}-{row['TypeOfCurrentPeriod']}", axis=1
)
sorted_data["会社名"] = [
    paint_companies.at[code, "CompanyName"] for code in sorted_data["LocalCode"]
]
sorted_data
```




<div>
<style scoped>
    .dataframe tbody tr td {
      font-size: 0.5em;
    }
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
      <th></th>
      <th>DisclosedDate</th>
      <th>DisclosedTime</th>
      <th>LocalCode</th>
      <th>DisclosureNumber</th>
      <th>TypeOfDocument</th>
      <th>TypeOfCurrentPeriod</th>
      <th>...</th>
      <th>決算期</th>
      <th>会社名</th>
    </tr>
    <tr>
      <th>LocalCode</th>
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
      <th rowspan="5" valign="top">46110</th>
      <th>8</th>
      <td>2023-11-09</td>
      <td>14:00:00</td>
      <td>46110</td>
      <td>20231020569248</td>
      <td>2QFinancialStatements_Consolidated_JP</td>
      <td>2Q</td>
      <td>...</td>
      <td>2023-2Q</td>
      <td>大日本塗料</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-08-08</td>
      <td>14:00:00</td>
      <td>46110</td>
      <td>20230721525034</td>
      <td>1QFinancialStatements_Consolidated_JP</td>
      <td>1Q</td>
      <td>...</td>
      <td>2023-1Q</td>
      <td>大日本塗料</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-05-11</td>
      <td>14:00:00</td>
      <td>46110</td>
      <td>20230421551091</td>
      <td>FYFinancialStatements_Consolidated_JP</td>
      <td>FY</td>
      <td>...</td>
      <td>2023-FY</td>
      <td>大日本塗料</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-02-09</td>
      <td>14:00:00</td>
      <td>46110</td>
      <td>20230123592325</td>
      <td>3QFinancialStatements_Consolidated_JP</td>
      <td>3Q</td>
      <td>...</td>
      <td>2023-3Q</td>
      <td>大日本塗料</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-10-27</td>
      <td>14:00:00</td>
      <td>46110</td>
      <td>20221024548184</td>
      <td>EarnForecastRevision</td>
      <td>2Q</td>
      <td>...</td>
      <td>2022-2Q</td>
      <td>大日本塗料</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">46240</th>
      <th>4</th>
      <td>2023-02-09</td>
      <td>13:00:00</td>
      <td>46240</td>
      <td>20230123592504</td>
      <td>3QFinancialStatements_Consolidated_JP</td>
      <td>3Q</td>
      <td>...</td>
      <td>2023-3Q</td>
      <td>イサム塗料</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-11-07</td>
      <td>13:00:00</td>
      <td>46240</td>
      <td>20221024547981</td>
      <td>2QFinancialStatements_Consolidated_JP</td>
      <td>2Q</td>
      <td>...</td>
      <td>2022-2Q</td>
      <td>イサム塗料</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-08-05</td>
      <td>13:00:00</td>
      <td>46240</td>
      <td>20220722503491</td>
      <td>1QFinancialStatements_Consolidated_JP</td>
      <td>1Q</td>
      <td>...</td>
      <td>2022-1Q</td>
      <td>イサム塗料</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-05-10</td>
      <td>13:00:00</td>
      <td>46240</td>
      <td>20220426528461</td>
      <td>FYFinancialStatements_Consolidated_JP</td>
      <td>FY</td>
      <td>...</td>
      <td>2022-FY</td>
      <td>イサム塗料</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2022-02-09</td>
      <td>13:00:00</td>
      <td>46240</td>
      <td>20220124571237</td>
      <td>3QFinancialStatements_Consolidated_JP</td>
      <td>3Q</td>
      <td>...</td>
      <td>2022-3Q</td>
      <td>イサム塗料</td>
    </tr>
  </tbody>
</table>
<p>91 rows × 108 columns</p>
</div>




```python
plt.figure(figsize=(20, 5))
hue_order = [
    "2021-FY",
    "2021-1Q",
    "2021-2Q",
    "2022-3Q",
    "2022-FY",
    "2022-1Q",
    "2022-2Q",
    "2023-3Q",
    "2023-FY",
    "2023-1Q",
    "2023-2Q",
]
ax = sns.barplot(
    data=sorted_data,
    x="会社名",
    y="EarningsPerShare",
    hue="決算期",
    hue_order=hue_order,
)
for c in ax.containers:
    ax.bar_label(c, rotation=90, fontsize=10)

plt.xticks(rotation=90)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.show()
```


    
![四半期ごとのEPSの比較](/images/finance/misc/jquants_files/jquants_15_0.png)
    

