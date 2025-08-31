---
title: "EDGARのデータを取得"
pre: "3.6.2 "
weight: 2
title_replace: "pythonで米国企業の財務諸表から表（テーブル）を抽出してみる"
---

{{% youtube "twu1fOw7ZBo" %}}

EDGAR(Electronic Data Gathering, Analysis, and Retrieval system)とは、米国の証券取引委員会の運営するサイトです。
米国の法による法定開示書類が管理されています。ここでは米国企業の財務諸表も管理されています。

今回は指定した企業の財務諸表を取得して、データをプロットしてみようと思います。


```python
import os
from sec_edgar_downloader import Downloader

dl = Downloader("./data/")
ticker_symbol = "MSFT"

if os.path.exists(f"./data/sec-edgar-filings/{ticker_symbol}/10-K/"):
    print("ダウンロード済みです。")
elif dl.get("10-K", ticker_symbol, after="2021-01-01", before="2021-12-31") > 0:
    print("ダウンロードに成功しました。")
else:
    print("ダウンロードに失敗しました。")
```

    ダウンロード済みです。


## 10-Kに含まれる表を抜き出す
pandasの`read_html`を用いることでテーブルをDataFrameの形で抜き出すことができます。

{{% notice document %}}
[pandas.read_html](https://pandas.pydata.org/docs/reference/api/pandas.read_html.html)
{{% /notice %}}


```python
import glob
import pandas as pd

filing_details_filepath = glob.glob(
    f"./data/sec-edgar-filings/{ticker_symbol}/10-K/*/filing-details.html"
)[0]
tables = pd.read_html(filing_details_filepath)
```

## CASH FLOWS STATEMENTSのテーブルを抽出
「CASH FLOWS STATEMENTS」のページのテーブルを抽出します。
様々な方法が考えられますが、ここでは「*Cash and cash equivalents, end of period*」というワードを見つけたらそのテーブルを抜き出すように指定しています。


```python
cs_table = None  # CASH FLOWS STATEMENTSのページのテーブル
for table in tables:
    tab_html = table.to_html()
    if "Cash and cash equivalents, end of period" in tab_html:
        cs_table = table

cs_table.head(10)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(In millions)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Year Ended June 30,</td>
      <td>NaN</td>
      <td>2021</td>
      <td>2021</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020</td>
      <td>2020</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019</td>
      <td>2019</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Operations</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Net income</td>
      <td>NaN</td>
      <td>$</td>
      <td>61271</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>$</td>
      <td>44281</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>$</td>
      <td>39240</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Adjustments to reconcile net income to net cas...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Depreciation, amortization, and other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11686</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12796</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11682</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stock-based compensation expense</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6118</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5289</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4652</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## データの前処理
NaNが多く含まれる上、文字列扱いになっていてこのままでは数値を読み取れません。
NaNや不要な記号を取り除き、文字列を数値に変換します。


```python
import numpy as np

# 「(」「)」「,」の文字列を削除する
cs_table = cs_table.replace([",", "\)", "\("], "", regex=True)
cs_table = cs_table.replace([""], np.nan)  # 空白セルはnan扱いにする
cs_table.head(10)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>In millions</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Year Ended June 30</td>
      <td>NaN</td>
      <td>2021</td>
      <td>2021</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2020</td>
      <td>2020</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2019</td>
      <td>2019</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Operations</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Net income</td>
      <td>NaN</td>
      <td>$</td>
      <td>61271</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>$</td>
      <td>44281</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>$</td>
      <td>39240</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Adjustments to reconcile net income to net cas...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Depreciation amortization and other</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11686</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12796</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11682</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stock-based compensation expense</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6118</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5289</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4652</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# データがない列を削除する
row_num = cs_table.shape[0]

for colname in cs_table.columns:
    if cs_table[colname].isna().sum() > row_num * 0.8:  # 8割以上の行がNaNの列は
        cs_table.drop(colname, inplace=True, axis=1)  # 不要な列として削除する
cs_table.head(10)
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
      <th>0</th>
      <th>3</th>
      <th>7</th>
      <th>11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>In millions</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Year Ended June 30</td>
      <td>2021</td>
      <td>2020</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Operations</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Net income</td>
      <td>61271</td>
      <td>44281</td>
      <td>39240</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Adjustments to reconcile net income to net cas...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Depreciation amortization and other</td>
      <td>11686</td>
      <td>12796</td>
      <td>11682</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Stock-based compensation expense</td>
      <td>6118</td>
      <td>5289</td>
      <td>4652</td>
    </tr>
  </tbody>
</table>
</div>




```python
# NaNの多い行も削除する
col_num = cs_table.shape[1]
cs_table[cs_table.isna().sum(axis=1) < col_num * 0.8]
cs_table.columns = ["item", "2021", "2020", "2019"]
cs_table = cs_table[["item", "2019", "2020", "2021"]]
# years = cs_table.fillna('').query("item.str.contains('Year Ended')").iloc[0, 1:]
```

## データをプロットする


```python
import matplotlib.pyplot as plt

years = ["2019", "2020", "2021"]

for item_name in cs_table["item"][:10]:
    try:
        data = [float(v) for v in cs_table.query(f"item=='{item_name}'").iloc[0, 1:]]
        plt.plot(data)
        plt.xticks([i for i in range(len(years))], years)
        plt.title(item_name)
        plt.grid()
        plt.show()
    except IndexError:
        pass
```


    
![png](/images/prep/special/SEC_EDGAR_files/SEC_EDGAR_11_0.png)
    



    
![png](/images/prep/special/SEC_EDGAR_files/SEC_EDGAR_11_1.png)
    



    
![png](/images/prep/special/SEC_EDGAR_files/SEC_EDGAR_11_2.png)
    



    
![png](/images/prep/special/SEC_EDGAR_files/SEC_EDGAR_11_3.png)
    



    
![png](/images/prep/special/SEC_EDGAR_files/SEC_EDGAR_11_4.png)
    



    
![png](/images/prep/special/SEC_EDGAR_files/SEC_EDGAR_11_5.png)
    



    
![png](/images/prep/special/SEC_EDGAR_files/SEC_EDGAR_11_6.png)
    

