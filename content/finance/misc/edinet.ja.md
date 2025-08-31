---
title: "EDINET"
pre: " 7.4.2 "
weight: 30
searchtitle: "pythonでEDINETのAPIを使用し、四半期報告書のデータを取得する。"
---


{{% youtube "LAHV8tmzNso" %}}

## EDINETのAPIを使用してみる


<div class="pagetop-box">
    <p>Pythonで四半期報告書のpdfデータを取得するために、EDINETを使います。以下のコンテンツではEDINET閲覧サイト（<a href="https://disclosure2.edinet-fsa.go.jp/WEEE0030.aspx?bXVsPeWVhuiIueS4ieS6lSZjdGY9b2ZmJmZscz1vbiZscHI9b2ZmJnJwcj1vZmYmb3RoPW9mZiZwZnM9NiZ5ZXI9Jm1vbj0=">url</a>）から取得できる情報をもとにデータを加工して作成しています。本ページに記載されているコードの実行や取得したデータの利用についてはEDINETの利用規約を確認し理解した上で利用してください。また、Version2のAPIを使用する際はログイン認証＋APIキーの発行が必要になります。</p>
    <p>※注意：使用時には必ず利用規約（<a href="https://disclosure2dl.edinet-fsa.go.jp/guide/static/disclosure/download/ESE140191.pdf">pdf</a>・<a href="https://disclosure2dl.edinet-fsa.go.jp/guide/static/submit/WZEK0030.html">web</a>)を確認した上で常識の範囲内での使用にとどめてください。</p>
</div>








```python
import os
import requests
import pandas as pd

API_ENDPOINT = "https://disclosure.edinet-fsa.go.jp/api/v2"  # v2を使用する
```

### 書類一覧APIのリクエストURL
11_EDINET_API仕様書に従って書類一覧を取得してみます。


```python
request_params = {
    "date": "2024-02-09",
    "type": 2,  # 1=メタデータのみ、2=提出書類一覧及びメタデータ
    "Subscription-Key": os.environ.get(
        "EDINET_API_KEY"
    ),  # v1を使用する場合は不要, 2024年３月29日（金）まで利用可能
}

docs_submitted_json = requests.get(
    f"{API_ENDPOINT}/documents.json", request_params
).json()
```

## 取得データの確認
今は四半期報告書を取得したいので、『四半期報告書』が文書の説明（`docDescription`）に含まれている行のみを抽出して確認します。


```python
sd_df = pd.DataFrame(docs_submitted_json["results"])
sd_df = sd_df[sd_df["docDescription"].str.contains("四半期報告書", na=False)]
sd_df.head()
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
      <th>seqNumber</th>
      <th>docID</th>
      <th>edinetCode</th>
      <th>secCode</th>
      <th>JCN</th>
      <th>filerName</th>
      <th>fundCode</th>
      <th>ordinanceCode</th>
      <th>formCode</th>
      <th>docTypeCode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>S100SSMQ</td>
      <td>E04505</td>
      <td>95070</td>
      <td>9470001001933</td>
      <td>四国電力株式会社</td>
      <td>None</td>
      <td>010</td>
      <td>043000</td>
      <td>140</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>S100SSHR</td>
      <td>E01441</td>
      <td>59730</td>
      <td>5122001016280</td>
      <td>株式会社トーアミ</td>
      <td>None</td>
      <td>010</td>
      <td>043000</td>
      <td>140</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>S100SQAH</td>
      <td>E30982</td>
      <td>71750</td>
      <td>9220001001223</td>
      <td>今村証券株式会社</td>
      <td>None</td>
      <td>010</td>
      <td>043000</td>
      <td>140</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>S100SPH6</td>
      <td>E03562</td>
      <td>83600</td>
      <td>3090001002315</td>
      <td>株式会社　山梨中央銀行</td>
      <td>None</td>
      <td>010</td>
      <td>043000</td>
      <td>140</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>S100SRF2</td>
      <td>E00783</td>
      <td>40910</td>
      <td>7010701015826</td>
      <td>日本酸素ホールディングス株式会社</td>
      <td>None</td>
      <td>010</td>
      <td>043000</td>
      <td>140</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



### DocIDの確認
商船三井の四半期報告書のDocIDを確認してみます。`filerName`に商船三井が含まれる行のみを抽出します。


```python
sd_df[sd_df.filerName.str.contains("商船三井")]
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
      <th>seqNumber</th>
      <th>docID</th>
      <th>edinetCode</th>
      <th>secCode</th>
      <th>JCN</th>
      <th>filerName</th>
      <th>fundCode</th>
      <th>ordinanceCode</th>
      <th>formCode</th>
      <th>docTypeCode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1307</th>
      <td>1308</td>
      <td>S100STH6</td>
      <td>E04236</td>
      <td>91040</td>
      <td>4010401082896</td>
      <td>株式会社商船三井</td>
      <td>None</td>
      <td>010</td>
      <td>043000</td>
      <td>140</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 29 columns</p>
</div>



### 四半期報告書の取得

`docID` をもとに四半期報告書の取得に必要なデータを取得します。pdfを取得したい場合はマニュアルに従い、type=2を指定します。


```python
docID = "S100STH6"
pdf_response = requests.get(
    f"{API_ENDPOINT}/documents/{docID}",
    {
        "type": 2,
        "Subscription-Key": os.environ.get("EDINET_API_KEY"),
    },
)

with open("sample.pdf", "wb") as f:
    f.write(pdf_response.content)
```

#### ダウンロードしたpdfを表示する


```python
from IPython.display import display_pdf

with open("sample.pdf", "rb") as f:
    display_pdf(f.read(), raw=True)
```

（pdfの表示は省略します）
