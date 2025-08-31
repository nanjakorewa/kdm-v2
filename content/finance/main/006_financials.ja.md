---
title: "企業の財務情報をパース"
pre: "7.1.9 "
weight: 9
not_use_colab: true
---


<div class="pagetop-box">
<p>pythonで複数の会社の財務データを比較します。
ここでは、近い業界にある３つの会社の期ごとの売上と当期純利益の推移を見てみようと思います。</p>
</div>



{{% notice ref %}}
このページの日英訳は以下のサイトのものを参考しつつ作成していますが、正確性に欠ける可能性があるのであくまで参考程度にお願い致します。
[TOMAコンサルタンツグループ株式会社 海外決算書の科目　英語→日本語簡易対訳　損益計算書編](https://toma.co.jp/blog/overseas/%e6%b5%b7%e5%a4%96%e6%b1%ba%e7%ae%97%e6%9b%b8%e3%81%ae%e7%a7%91%e7%9b%ae%e3%80%80%e8%8b%b1%e8%aa%9e%e2%86%92%e6%97%a5%e6%9c%ac%e8%aa%9e%e7%b0%a1%e6%98%93%e5%af%be%e8%a8%b3%e3%80%80%e6%90%8d%e7%9b%8a/)
{{% /notice %}}

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
from IPython.core.display import display
```


```python
# 英日辞書
en_US_ja_JP_table = {
    "Revenue": "売上",
    "Cost of revenue": "収益コスト",
    "Gross profit": "粗利益",
    "Sales, General and administrative": "販売費及び一般管理費",
    "Other operating expenses": "その他営業費用",
    "Total operating expenses": "営業費用",
    "Operating income": "営業利益",
    "Interest Expense": "支払利息",
    "Other income (expense)": "Other income (expense)",
    "Income before taxes": "Income before taxes",
    "Provision for income taxes": "Provision for income taxes",
    "Net income from continuing operations": "Net income from continuing operations",
    "Net income available to common shareholders": "普通株式に係る当期純利益",
    "Net income": "当期純利益",
    "Basic": "Basic",
    "Diluted": "Diluted",
    "EBITDA": "EBITDA",
    "Revenue ": "売上",
    "Gross Margin %": "	売上総利益率",
    "Operating Income ": "営業利益",
    "Operating Margin %": "営業利益率",
    "Net Income ": "純利益",
    "Earnings Per Share USD": "EPS (USD)",
    "Dividends USD": "配当 (USD)",
    "Payout Ratio % *": "配当性向",
    "Shares Mil": "株数 (Mil)",
    "Book Value Per Share * USD": "1株あたり純資産",
    "Operating Cash Flow ": "営業キャッシュフロー",
    "Cap Spending ": "資本的支出",
    "Free Cash Flow ": "フリーキャッシュフロー（FCF）",
    "Free Cash Flow Per Share * USD": "1株あたりFCF",
    "Working Capital ": "運転資本",
    "Key Ratios -> Profitability": "",
    "Margins % of Sales": "Margins % of Sales",
    "COGS": "売上原価",
    "Gross Margin": "売上高総利益率",
    "SG&A": "販売費及び一般管理費",
    "R&D": "研究開発費",
    "Operating Margin": "営業利益率",
    "Net Int Inc & Other": "資金運用利益+その他",
    "EBT Margin": "EBTマージン",
    "Debt/Equity": "負債比率",
    "Receivables Turnover": "売上債権回転率",
    "Inventory Turnover": "棚卸資産回転率",
    "Fixed Assets Turnover": "固定資産回転率",
    "Asset Turnover": "総資産回転率",
    "USD Mil": "(USD Mil)",
    "-Year Average": "年平均",
}


def get_preprocessed_df(
    filepath,
    is_transpose=True,
    drop_ttm_lastqrt=True,
    ttm_datetime="2021-12",
    lastqtr_datetime="2021-12",
    add_ticker_column=True,
):
    """Morningstar, Incのデータを整形する

    Args:
        filepath (str): ファイルパス
        is_transpose (bool, optional): タイムスタンプ列を縦にするかどうか. Defaults to True.
        drop_ttm_lastqrt (bool, optional): TTM/Last Qrtの記録は削除する. Defaults to True.
        ttm_datetime (str, optional): 「TTM」をどの日付に置き換えるか. Defaults to "2021-12".
        lastqtr_datetime (str, optional): 「Latest Qtr」をどの日付に置き換えるか. Defaults to "2021-12".
        add_ticker_column (bool, optional): ticker symbolを示した列を追加するか. Defaults to True.

    Returns:
        DataFrame: 整形済みデータフレーム
    """
    df_preprocessed = []
    df_header = []
    row_header = ""
    df = pd.read_table(filepath, header=None)

    if not drop_ttm_lastqrt:
        print(f"[Note] TTM は {ttm_datetime} 、Last Qtrは {lastqtr_datetime} の日付として扱われます。")

    for idx, row in enumerate(df[0]):
        # 数値中の「,」を置換する
        row = re.sub('"(-?[0-9]+),', '"\\1', row)
        row = re.sub(',(-?[0-9]+)",', '\\1",', row)

        # 英語を対応する日本語に置き換える
        for str_en, str_jp in en_US_ja_JP_table.items():
            if str_en in row:
                row = row.replace(str_en, str_jp)

        # TTMがある行はタイムスタンプなのでヘッダー扱いにする
        if "TTM" in row or "Latest Qtr" in row:
            if drop_ttm_lastqrt:
                row = row.replace("TTM", "###IGNORE###")
                row = row.replace("Latest Qtr", "###IGNORE###")
            else:
                assert ttm_datetime not in row, "その日付はすでに存在しています！"
                assert lastqtr_datetime not in row, "その日付はすでに存在しています！"
                row = row.replace("TTM", ttm_datetime)
                row = row.replace("Latest Qtr", lastqtr_datetime)
            df_header = row.split(",")

            if is_transpose:
                df_header[0] = "月"
            else:
                df_header = ["月"] + df_header
            continue

        # 数値に変換できるデータは数値に変換してDataFrameに追加
        if len(row_splited := row.split(",")) > 1:
            row_data = [
                float(v) if re.match(r"^-?\d+(?:\.\d+)$", v) is not None else v
                for v in row_splited
            ]

            if is_transpose:
                row_data[0] = (
                    f"{row_header}/{row_data[0]}" if row_header else f"{row_data[0]}"
                )
            else:
                row_data = [row_header] + row_data
            df_preprocessed.append(row_data)
        else:
            # 先頭の行はファイルのタイトルが入っているので無視
            row_header = f"{row}" if idx > 0 else ""

    # データフレーム作成
    df_preprocessed = pd.DataFrame(df_preprocessed)
    df_preprocessed.columns = df_header
    if drop_ttm_lastqrt:
        df_preprocessed.drop("###IGNORE###", axis=1, inplace=True)

    # 不要な文字列を削除
    df_preprocessed.fillna(np.nan, inplace=True)

    if is_transpose:
        df_preprocessed = df_preprocessed.T.reset_index()
        df_preprocessed.columns = df_preprocessed.iloc[0, :]
        df_preprocessed.drop(0, inplace=True)
        df_preprocessed["月"] = pd.to_datetime(df_preprocessed["月"])
        for colname in df_preprocessed.columns:
            if colname != "月":
                df_preprocessed[colname] = pd.to_numeric(
                    df_preprocessed[colname], errors="coerce"
                )

    if add_ticker_column:
        filename = os.path.basename(filepath)
        ticker_symbol = filename[: filename.index(" ")]
        df_preprocessed["ticker"] = [
            ticker_symbol for _ in range(df_preprocessed.shape[0])
        ]

    return df_preprocessed
```

## データを読み込む
以下の例では[Morningstar, Inc](https://www.morningstar.com/)社から提供されているデータを一部引用して使用しています。
このサイトで、指定した企業の財務情報をまとめたcsvファイルを取得します。ここの例では[Golden Ocean Group Ltd](https://www.morningstar.com/stocks/xnas/gogl/quote)などのデータを使用しています。

<span style="color:#ccc">※あくまで表示例であり正確性は保証しません。万一この情報に基づいて被ったいかなる損害についても一切責任を負い兼ねます。</span>


```python
df_is_gogl = get_preprocessed_df("data/GOGL Income Statement.csv")
df_kr_gogl = get_preprocessed_df("data/GOGL Key Ratios.csv")
df_is_zim = get_preprocessed_df("data/ZIM Income Statement.csv")
df_kr_zim = get_preprocessed_df("data/ZIM Key Ratios.csv")
df_is_sblk = get_preprocessed_df("data/SBLK Income Statement.csv")
df_kr_sblk = get_preprocessed_df("data/SBLK Key Ratios.csv")

df_income_statement = pd.concat([df_is_gogl, df_is_sblk, df_is_zim])
df_key_ratio = pd.concat([df_kr_gogl, df_kr_zim, df_kr_sblk])

display(df_income_statement.head())
display(df_key_ratio.head())
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
      <th>月</th>
      <th>売上</th>
      <th>収益コスト</th>
      <th>粗利益</th>
      <th>Operating expenses/販売費及び一般管理費</th>
      <th>Operating expenses/その他営業費用</th>
      <th>Operating expenses/営業費用</th>
      <th>Operating expenses/営業利益</th>
      <th>Operating expenses/支払利息</th>
      <th>Operating expenses/Other income (expense)</th>
      <th>...</th>
      <th>Operating expenses/当期純利益 from continuing operations</th>
      <th>Operating expenses/当期純利益</th>
      <th>Operating expenses/普通株式に係る当期純利益</th>
      <th>Earnings per share/Basic</th>
      <th>Earnings per share/Diluted</th>
      <th>Weighted average shares outstanding/Basic</th>
      <th>Weighted average shares outstanding/Diluted</th>
      <th>Weighted average shares outstanding/EBITDA</th>
      <th>ticker</th>
      <th>Operating expenses/Other</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2016-12-01</td>
      <td>258</td>
      <td>313</td>
      <td>-55</td>
      <td>13</td>
      <td>2</td>
      <td>15</td>
      <td>-70</td>
      <td>42</td>
      <td>-16</td>
      <td>...</td>
      <td>-128</td>
      <td>-128</td>
      <td>-128</td>
      <td>-1.34</td>
      <td>-1.34</td>
      <td>95</td>
      <td>95</td>
      <td>-22</td>
      <td>GOGL</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-12-01</td>
      <td>460</td>
      <td>400</td>
      <td>60</td>
      <td>13</td>
      <td>-4</td>
      <td>9</td>
      <td>51</td>
      <td>57</td>
      <td>3</td>
      <td>...</td>
      <td>-2</td>
      <td>-2</td>
      <td>-2</td>
      <td>-0.02</td>
      <td>-0.02</td>
      <td>125</td>
      <td>125</td>
      <td>133</td>
      <td>GOGL</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-12-01</td>
      <td>656</td>
      <td>499</td>
      <td>158</td>
      <td>15</td>
      <td>-3</td>
      <td>12</td>
      <td>146</td>
      <td>73</td>
      <td>12</td>
      <td>...</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>0.59</td>
      <td>0.59</td>
      <td>144</td>
      <td>144</td>
      <td>250</td>
      <td>GOGL</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2019-12-01</td>
      <td>706</td>
      <td>590</td>
      <td>116</td>
      <td>14</td>
      <td>1</td>
      <td>15</td>
      <td>101</td>
      <td>57</td>
      <td>-6</td>
      <td>...</td>
      <td>37</td>
      <td>37</td>
      <td>37</td>
      <td>0.26</td>
      <td>0.26</td>
      <td>144</td>
      <td>144</td>
      <td>188</td>
      <td>GOGL</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2020-12-01</td>
      <td>608</td>
      <td>564</td>
      <td>44</td>
      <td>14</td>
      <td>-3</td>
      <td>11</td>
      <td>33</td>
      <td>45</td>
      <td>-126</td>
      <td>...</td>
      <td>-138</td>
      <td>-138</td>
      <td>-138</td>
      <td>-0.96</td>
      <td>-0.96</td>
      <td>143</td>
      <td>143</td>
      <td>18</td>
      <td>GOGL</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



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
      <th>月</th>
      <th>Financials/売上 (USD Mil)</th>
      <th>Financials/\t売上総利益率</th>
      <th>Financials/営業利益(USD Mil)</th>
      <th>Financials/営業利益率</th>
      <th>Financials/純利益(USD Mil)</th>
      <th>Financials/EPS (USD)</th>
      <th>Financials/配当 (USD)</th>
      <th>Financials/配当性向</th>
      <th>Financials/株数 (Mil)</th>
      <th>...</th>
      <th>Key Ratios -&gt; Financial Health/負債比率</th>
      <th>Key Ratios -&gt; Efficiency Ratios/Days Sales Outstanding</th>
      <th>Key Ratios -&gt; Efficiency Ratios/Days Inventory</th>
      <th>Key Ratios -&gt; Efficiency Ratios/Payables Period</th>
      <th>Key Ratios -&gt; Efficiency Ratios/Cash Conversion Cycle</th>
      <th>Key Ratios -&gt; Efficiency Ratios/売上債権回転率</th>
      <th>Key Ratios -&gt; Efficiency Ratios/棚卸資産回転率</th>
      <th>Key Ratios -&gt; Efficiency Ratios/固定資産回転率</th>
      <th>Key Ratios -&gt; Efficiency Ratios/総資産回転率</th>
      <th>ticker</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2011-12-01</td>
      <td>95.0</td>
      <td>68.2</td>
      <td>38.0</td>
      <td>40.4</td>
      <td>33.0</td>
      <td>6.12</td>
      <td>9.00</td>
      <td>NaN</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.42</td>
      <td>18.79</td>
      <td>32.12</td>
      <td>36.01</td>
      <td>14.89</td>
      <td>19.43</td>
      <td>11.36</td>
      <td>0.21</td>
      <td>0.18</td>
      <td>GOGL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2012-12-01</td>
      <td>37.0</td>
      <td>38.2</td>
      <td>10.0</td>
      <td>26.8</td>
      <td>-53.0</td>
      <td>-10.08</td>
      <td>5.40</td>
      <td>488.7</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.39</td>
      <td>48.04</td>
      <td>39.37</td>
      <td>26.39</td>
      <td>61.02</td>
      <td>7.60</td>
      <td>9.27</td>
      <td>0.11</td>
      <td>0.08</td>
      <td>GOGL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013-12-01</td>
      <td>38.0</td>
      <td>31.3</td>
      <td>7.0</td>
      <td>18.2</td>
      <td>-4.0</td>
      <td>-0.69</td>
      <td>3.15</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.30</td>
      <td>26.25</td>
      <td>20.60</td>
      <td>19.16</td>
      <td>27.68</td>
      <td>13.91</td>
      <td>17.72</td>
      <td>0.13</td>
      <td>0.09</td>
      <td>GOGL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-12-01</td>
      <td>97.0</td>
      <td>25.4</td>
      <td>19.0</td>
      <td>20.1</td>
      <td>16.0</td>
      <td>1.38</td>
      <td>2.81</td>
      <td>184.2</td>
      <td>11.0</td>
      <td>...</td>
      <td>0.39</td>
      <td>11.45</td>
      <td>37.85</td>
      <td>16.10</td>
      <td>33.20</td>
      <td>31.88</td>
      <td>9.64</td>
      <td>0.13</td>
      <td>0.12</td>
      <td>GOGL</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-12-01</td>
      <td>190.0</td>
      <td>-28.6</td>
      <td>-72.0</td>
      <td>-37.6</td>
      <td>-221.0</td>
      <td>-7.30</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30.0</td>
      <td>...</td>
      <td>0.80</td>
      <td>11.90</td>
      <td>21.19</td>
      <td>5.57</td>
      <td>27.51</td>
      <td>30.68</td>
      <td>17.22</td>
      <td>0.13</td>
      <td>0.11</td>
      <td>GOGL</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 87 columns</p>
</div>


## 変化を可視化する
seabornを使ってグラフを作成してみます。

{{% notice document %}}
[seaborn.barplot](https://seaborn.pydata.org/generated/seaborn.barplot.html)
{{% /notice %}}


```python
def show_barchart(data, x="月", y="売上", hue="ticker", is_log_scale=False):
    """バーチャートを表示する

    Args:
        data (pandas.DataFrame): データフレーム
        x (str, optional): 時間軸. Defaults to "月".
        y (str, optional): 比較する指標. Defaults to "売上".
        hue (str, optional): 何基準で比較するか. Defaults to "ticker".
        is_log_scale (bool, optional): logスケールで表示するかどうか. Defaults to False.
    """
    sns.set_theme(style="whitegrid", rc={"figure.figsize": (10, 4)})
    japanize_matplotlib.japanize()

    g = sns.barplot(data=data, x=x, y=y, hue=hue)
    g.set_xticklabels(
        [xt.get_text().split("-01")[0] for xt in g.get_xticklabels()]
    )  # TODO: mdates.DateFormatterで日付を表示するとなぜか日付がずれるのでラベルを直接書き換える
    g.tick_params(axis="x", rotation=90)
    if is_log_scale:
        g.set_yscale("log")
    plt.legend(loc="upper left", title="Ticker Name")
    plt.title(f"{y}の比較", fontsize=14)
    plt.show()
```


```python
show_barchart(df_income_statement, x="月", y="売上", hue="ticker")
show_barchart(df_income_statement, x="月", y="Operating expenses/当期純利益", hue="ticker")
```


    
![png](/images/finance/main/006_financials_files/006_financials_8_0.png)
    



    
![png](/images/finance/main/006_financials_files/006_financials_8_1.png)
    

