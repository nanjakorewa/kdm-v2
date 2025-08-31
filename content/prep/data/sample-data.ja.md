---
title: "サンプルデータ"
pre: "3.2.1 "
weight: 1
title_replace: "総務省のデータをpandas.DataFrameで読み取る"
---

## 総務省のデータ
総務省のデータをつかってpandasを動かしてみましょう。

> 当ホームページ(総務省)で公開している情報（以下「コンテンツ」といいます。）は、どなたでも以下の1）～7）に従って、複製、公衆送信、翻訳・変形等の翻案等、自由に利用できます。商用利用も可能です。また、数値データ、簡単な表・グラフ等は著作権の対象ではありませんので、これらについては本利用ルールの適用はなく、自由に利用できます。コンテンツ利用に当たっては、本利用ルールに同意したものとみなします。 [引用元：当省ホームページについて](https://www.soumu.go.jp/menu_kyotsuu/policy/tyosaku.html)

```python
import pandas as pd
import os
from urllib.parse import urlparse
from IPython.display import display, HTML


def getfn(url: str) -> str:
    """urlからファイル名を取得する"""
    return os.path.basename(urlparse(url).path)


def disp(df: pd.DataFrame, text: str):
    """notebook上にデータとテキストを表示する"""
    display(HTML(f"<h3>{text}</h3>"))
    display(df.head(6))
    display(HTML(f"<hr />"))
```

## 国内総生産の増加率に対する寄与度
出典：[令和3年版地方財政白書](https://www.soumu.go.jp/menu_seisaku/hakusyo/chihou/r03data/2021data/r03czb01-01.html#p010102)<span class="somu-kink">（https://www.soumu.go.jp/menu_seisaku/hakusyo/chihou/r03data/2021data/r03czb01-01.html#p010102）</span>


```python
df = pd.read_csv("z-006-pre.csv")
disp(df, "加工前")
df.columns = df.iloc[1]
df = df.drop([0, 1]).replace("-", 0)
df.to_csv("z-006.csv", encoding="utf-8-sig")
disp(df, "加工後")
```


<h3>加工前</h3>



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
      <th>第6図　国内総生産（支出側、名目）の増加率に対する寄与度</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Unnamed: 11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>（単位　％）</td>
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
      <td>平成21年度</td>
      <td>平成22年度</td>
      <td>平成23年度</td>
      <td>平成24年度</td>
      <td>平成25年度</td>
      <td>平成26年度</td>
      <td>平成27年度</td>
      <td>平成28年度</td>
      <td>平成29年度</td>
      <td>平成30年度</td>
      <td>令和元年度</td>
    </tr>
    <tr>
      <th>2</th>
      <td>中央政府</td>
      <td>0.1</td>
      <td>△ 0.4</td>
      <td>-</td>
      <td>△ 0.1</td>
      <td>0.4</td>
      <td>-</td>
      <td>△ 0.1</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>地方政府</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>-</td>
      <td>△ 0.1</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.1</td>
      <td>-</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>社会保障基金</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.3</td>
      <td>-</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>公的企業</td>
      <td>-</td>
      <td>△ 0.1</td>
      <td>-</td>
      <td>0.1</td>
      <td>-</td>
      <td>-</td>
      <td>0.1</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
</table>
</div>



<hr />



<h3>加工後</h3>



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
      <th>1</th>
      <th>NaN</th>
      <th>平成21年度</th>
      <th>平成22年度</th>
      <th>平成23年度</th>
      <th>平成24年度</th>
      <th>平成25年度</th>
      <th>平成26年度</th>
      <th>平成27年度</th>
      <th>平成28年度</th>
      <th>平成29年度</th>
      <th>平成30年度</th>
      <th>令和元年度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>中央政府</td>
      <td>0.1</td>
      <td>△ 0.4</td>
      <td>0</td>
      <td>△ 0.1</td>
      <td>0.4</td>
      <td>0</td>
      <td>△ 0.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>地方政府</td>
      <td>0.1</td>
      <td>0.1</td>
      <td>0</td>
      <td>△ 0.1</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.1</td>
      <td>0</td>
      <td>0.1</td>
      <td>0.2</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>社会保障基金</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>0.3</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.3</td>
      <td>0</td>
      <td>0.2</td>
      <td>0.1</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>公的企業</td>
      <td>0</td>
      <td>△ 0.1</td>
      <td>0</td>
      <td>0.1</td>
      <td>0</td>
      <td>0</td>
      <td>0.1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>企業部門</td>
      <td>△ 3.5</td>
      <td>1.3</td>
      <td>0.7</td>
      <td>0</td>
      <td>0.6</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>△ 0.1</td>
      <td>0.9</td>
      <td>0.4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>家計部門</td>
      <td>△ 1.8</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.6</td>
      <td>2.3</td>
      <td>△ 0.4</td>
      <td>0.4</td>
      <td>△ 0.2</td>
      <td>0.8</td>
      <td>0.3</td>
      <td>△ 0.1</td>
    </tr>
  </tbody>
</table>
</div>



<hr />


## 国・地方を通じた純計歳出規模
出典：[令和3年版地方財政白書](https://www.soumu.go.jp/menu_seisaku/hakusyo/chihou/r03data/2021data/r03czb01-01.html#p010102)
<span class="somu-kink">（https://www.soumu.go.jp/menu_seisaku/hakusyo/chihou/r03data/2021data/r03czb01-01.html#p010102）</span>


```python
df = pd.read_csv("z-002-pre.csv")
df.to_csv("z-002-pre.csv", index=None)
disp(df, "加工前")
df = df.drop([0, 1, 2])
df.to_csv("z-002.csv", encoding="utf-8-sig")
disp(df, "加工後")
```


<h3>加工前</h3>



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
      <th>第2図　国・地方を通じた純計歳出規模（目的別）</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>（単位　％）</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>地方の割合</td>
      <td>57.4</td>
      <td>国の割合</td>
      <td>42.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>衛生費</td>
      <td>3.8</td>
      <td>保健所・ごみ処理等</td>
      <td>98</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>学校教育費</td>
      <td>8.9</td>
      <td>小・中学校、幼稚園等</td>
      <td>87</td>
      <td>NaN</td>
      <td>13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>司法警察消防費</td>
      <td>4.1</td>
      <td>NaN</td>
      <td>77</td>
      <td>NaN</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



<hr />



<h3>加工後</h3>



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
      <th>第2図　国・地方を通じた純計歳出規模（目的別）</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>衛生費</td>
      <td>3.8</td>
      <td>保健所・ごみ処理等</td>
      <td>98</td>
      <td>NaN</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>学校教育費</td>
      <td>8.9</td>
      <td>小・中学校、幼稚園等</td>
      <td>87</td>
      <td>NaN</td>
      <td>13</td>
    </tr>
    <tr>
      <th>5</th>
      <td>司法警察消防費</td>
      <td>4.1</td>
      <td>NaN</td>
      <td>77</td>
      <td>NaN</td>
      <td>23</td>
    </tr>
    <tr>
      <th>6</th>
      <td>社会教育費等</td>
      <td>3.0</td>
      <td>公民館、図書館、博物館等</td>
      <td>81</td>
      <td>NaN</td>
      <td>19</td>
    </tr>
    <tr>
      <th>7</th>
      <td>民生費（年金関係を除く。）</td>
      <td>22.2</td>
      <td>児童福祉、介護などの老人福祉、生活保護等</td>
      <td>70</td>
      <td>NaN</td>
      <td>30</td>
    </tr>
    <tr>
      <th>8</th>
      <td>国土開発費</td>
      <td>8.4</td>
      <td>都市計画、道路、橋りょう、公営住宅等</td>
      <td>72</td>
      <td>NaN</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



<hr />


## 国・地方を通じた目的別歳出構成比の推移
出典：[令和3年版地方財政白書](https://www.soumu.go.jp/menu_seisaku/hakusyo/chihou/r03data/2021data/r03czb01-01.html#p010102)<span class="somu-kink">（https://www.soumu.go.jp/menu_seisaku/hakusyo/chihou/r03data/2021data/r03czb01-01.html#p010102）</span>


```python
df = pd.read_csv("z-001-pre.csv")
disp(df, "加工前")
df.columns = df.iloc[1]
df = df.drop([0, 1, 8])
df.to_csv("z-001.csv", encoding="utf-8-sig")
disp(df, "加工後")
```


<h3>加工前</h3>



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
      <th>第1図　国・地方を通じた目的別歳出額構成比の推移</th>
      <th>Unnamed: 1</th>
      <th>Unnamed: 2</th>
      <th>Unnamed: 3</th>
      <th>Unnamed: 4</th>
      <th>Unnamed: 5</th>
      <th>Unnamed: 6</th>
      <th>Unnamed: 7</th>
      <th>Unnamed: 8</th>
      <th>Unnamed: 9</th>
      <th>Unnamed: 10</th>
      <th>Unnamed: 11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>（単位　％）</td>
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
      <td>平成21年度</td>
      <td>平成22年度</td>
      <td>平成23年度</td>
      <td>平成24年度</td>
      <td>平成25年度</td>
      <td>平成26年度</td>
      <td>平成27年度</td>
      <td>平成28年度</td>
      <td>平成29年度</td>
      <td>平成30年度</td>
      <td>令和元年度</td>
    </tr>
    <tr>
      <th>2</th>
      <td>社会保障関係費</td>
      <td>29.8</td>
      <td>31.2</td>
      <td>32.7</td>
      <td>32.7</td>
      <td>31.8</td>
      <td>32.8</td>
      <td>33.7</td>
      <td>34.4</td>
      <td>34.6</td>
      <td>34.4</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>公債費</td>
      <td>18.9</td>
      <td>20.3</td>
      <td>19.8</td>
      <td>20.9</td>
      <td>20.9</td>
      <td>21.4</td>
      <td>21.3</td>
      <td>20.6</td>
      <td>20.9</td>
      <td>20.6</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>機関費</td>
      <td>11.9</td>
      <td>12.5</td>
      <td>11.7</td>
      <td>10.6</td>
      <td>11.8</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.4</td>
      <td>11.5</td>
      <td>11.9</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>教育費</td>
      <td>11.7</td>
      <td>11.9</td>
      <td>11.7</td>
      <td>12.1</td>
      <td>11.6</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.8</td>
      <td>11.9</td>
    </tr>
  </tbody>
</table>
</div>



<hr />



<h3>加工後</h3>



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
      <th>1</th>
      <th>NaN</th>
      <th>平成21年度</th>
      <th>平成22年度</th>
      <th>平成23年度</th>
      <th>平成24年度</th>
      <th>平成25年度</th>
      <th>平成26年度</th>
      <th>平成27年度</th>
      <th>平成28年度</th>
      <th>平成29年度</th>
      <th>平成30年度</th>
      <th>令和元年度</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>社会保障関係費</td>
      <td>29.8</td>
      <td>31.2</td>
      <td>32.7</td>
      <td>32.7</td>
      <td>31.8</td>
      <td>32.8</td>
      <td>33.7</td>
      <td>34.4</td>
      <td>34.6</td>
      <td>34.4</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>公債費</td>
      <td>18.9</td>
      <td>20.3</td>
      <td>19.8</td>
      <td>20.9</td>
      <td>20.9</td>
      <td>21.4</td>
      <td>21.3</td>
      <td>20.6</td>
      <td>20.9</td>
      <td>20.6</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>機関費</td>
      <td>11.9</td>
      <td>12.5</td>
      <td>11.7</td>
      <td>10.6</td>
      <td>11.8</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.4</td>
      <td>11.5</td>
      <td>11.9</td>
      <td>11.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>教育費</td>
      <td>11.7</td>
      <td>11.9</td>
      <td>11.7</td>
      <td>12.1</td>
      <td>11.6</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.7</td>
      <td>11.8</td>
      <td>11.8</td>
      <td>11.9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>国土保全及び開発費</td>
      <td>12.0</td>
      <td>11.3</td>
      <td>10.6</td>
      <td>10.2</td>
      <td>10.9</td>
      <td>10.5</td>
      <td>9.9</td>
      <td>10.4</td>
      <td>10.5</td>
      <td>10.5</td>
      <td>10.8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>産業経済費</td>
      <td>9.9</td>
      <td>8.2</td>
      <td>9.4</td>
      <td>8.2</td>
      <td>7.1</td>
      <td>6.8</td>
      <td>6.9</td>
      <td>6.7</td>
      <td>6.3</td>
      <td>6.2</td>
      <td>6.4</td>
    </tr>
  </tbody>
</table>
</div>



<hr />

