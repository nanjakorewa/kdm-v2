---
title: "pdfから表を抽出"
pre: "3.5.1 "
weight: 1
title_replace: "pythonでpdfから表(テーブル)を抽出する"
---

{{% youtube "iuBG4s-xbOo" %}}

## camelotを使う場合
### 必要なライブラリをインストール
今回はCamelotというライブラリを使ってpdfからテーブルを抽出します。
`opencv-contrib-python`、`camelot`、`tabula-py`が必要なので適宜インストールします。

```
poetry add opencv-contrib-python camelot tabula-py ghostscript
```

Ghostscriptというソフトウェアも必要なのでインストールします。
OSによってインストール方法が異なるので注意してください。
インストール方法は[こちら](https://camelot-py.readthedocs.io/en/master/user/install-deps.html)を参照してください。

### ghostscriptがインストールされているか確認する


```python
from ctypes.util import find_library

find_library("gs")  # gsが実行可能ならば /usr/local/lib/libgs.dylibなどの表示がされます
```




    '/usr/local/lib/libgs.dylib'



### pdfからテーブルを抽出する
例として[総務省のページで公開されている「政策ごとの予算との対応について」のpdf](https://www.soumu.go.jp/menu_yosan/yosan.html#r4)からテーブルを抽出します。テーブルがパースできたことがわかります。

※jupyterbookでエラーが出る場合があるためコメントアウトしています


```python
import camelot

# pdfを読み込んでテーブルを抽出
# pdf_name = "000788423.pdf"
# tables = camelot.read_pdf(pdf_name)
# print("パースできたテーブル数", tables.n)

# 先頭５行のみ表示
# tables[0].df.head()
```

今度はFLEX LNGという会社の決算情報をパースしてみます。
データは[FLEXLNG｜Investor Home](https://www.flexlng.com/investor-home/)で取得したファイルで実行しています。
今後はテーブルのパースに失敗してしましました。


```python
# pdfを読み込んでテーブルを抽出
pdf_name = "flex-lng-earnings-release-q3-2021.pdf"
tables = camelot.read_pdf(pdf_name)
print("パースできたテーブル数", tables.n)
# 先頭５行のみ表示
# tables[0].df.head()
```

    パースできたテーブル数 0


## tabula-py を使う場合
`poetry add tabula-py`などとしてtabulaをインストールしてください。
tabulaはバックグラウンドで[tabula-java
](https://github.com/tabulapdf/tabula-java)を使用していますが、Javaのバージョンが古い場合こちらがエラーになる場合があるようです。

参考文献：[subprocess.CalledProcessError While extracting table from PDF using tabula-py](https://github.com/chezou/tabula-py/issues/206)


```python
from tabula import read_pdf

tables = read_pdf("flex-lng-earnings-release-q3-2021.pdf", pages="all")
data = tables[1]
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
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ASSETS</th>
      <th>Unnamed: 0</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Current assets</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cash and cash equivalents</td>
      <td>4</td>
      <td>138,116</td>
      <td>144,151</td>
      <td>128,878</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Restricted cash</td>
      <td>4</td>
      <td>47</td>
      <td>56</td>
      <td>84</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Inventory</td>
      <td>NaN</td>
      <td>5,915</td>
      <td>4,075</td>
      <td>3,656</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Other current assets</td>
      <td>5</td>
      <td>12,503</td>
      <td>8,886</td>
      <td>25,061</td>
    </tr>
  </tbody>
</table>
</div>



### パースしたテーブルを可視化する
tabulateでパースしたテーブルは文字列になっているので、そこから数値を読み取ります。


```python
import matplotlib.pyplot as plt

d = data.query("ASSETS=='Cash and cash equivalents'").iloc[0][2:]

# パースしたテーブルは文字列になっているので数値に変換する
d = [int(v.replace(",", "")) for v in d]

# プロット
plt.bar([0, 1, 2], d)
plt.xticks([0, 1, 2], ["2021 Sep 30", "2021 June 30", "2020 Dec 31"])
plt.show()
```


    
![png](/images/prep/table/extract_table_from_pdf_files/extract_table_from_pdf_9_0.png)
    

