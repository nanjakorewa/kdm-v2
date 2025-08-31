---
title: "ツリーマップ"
pre: "6.1.2 "
weight: 2
not_use_colab: true
searchtitle: "pythonでツリーマップを作成する"
---

ツリーマップとは階層的なカテゴリを持つ数値データの可視化に使用できる図です。日経平均や[S&P 500](https://finviz.com/map.ashx)が代表的な例です。このnotebookでは[squarify](https://github.com/laserson/squarify)を使用します。



{{% notice info %}}
ツリーマップはplotlyを使うことでも作成できます。（参考：
[Treemap charts with Python - Plotly](https://plotly.com/python/treemaps/)）
{{% /notice %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import squarify

np.random.seed(0)  # 乱数を固定する

labels = ["A" * i for i in range(1, 5)]
sizes = [i * 10 for i in range(1, 5)]
colors = ["#%02x%02x%02x" % (i * 50, 0, 0) for i in range(1, 5)]

plt.figure(figsize=(5, 5))
squarify.plot(sizes, color=colors, label=labels)
plt.axis("off")
plt.show()
```


    
![png](/images/visualize/category-groupby/treemap_files/treemap_2_0.png)
    


## ポートフォリオを可視化する

持っている銘柄ごとの取得価格・現在価格のデータがあったとします。
そこから、[finviz](https://finviz.com/)のようなヒートマップを作ってみます。

### データを作成する
仮に、csvから以下のようなデータを読み込んだとします。

<b>※ここに示しているのは架空のデータです</b>


```python
import pandas as pd

data = [
    ["PBR", 80.20, 130.00],
    ["GOOG", 1188.0, 1588.0],
    ["FLNG", 70.90, 230.00],
    ["ZIM", 400.22, 630.10],
    ["GOGL", 120.20, 90.90],
    ["3466\nラサールロジ", 156.20, 147.00],  # 日本語表示のテスト用
]

df = pd.DataFrame(data)
df.columns = ["銘柄名", "取得価額", "現在の価額"]
df["評価損益"] = df["現在の価額"] - df["取得価額"]
df.head(6)
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
      <th>銘柄名</th>
      <th>取得価額</th>
      <th>現在の価額</th>
      <th>評価損益</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBR</td>
      <td>80.20</td>
      <td>130.0</td>
      <td>49.80</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GOOG</td>
      <td>1188.00</td>
      <td>1588.0</td>
      <td>400.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FLNG</td>
      <td>70.90</td>
      <td>230.0</td>
      <td>159.10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ZIM</td>
      <td>400.22</td>
      <td>630.1</td>
      <td>229.88</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GOGL</td>
      <td>120.20</td>
      <td>90.9</td>
      <td>-29.30</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3466\nラサールロジ</td>
      <td>156.20</td>
      <td>147.0</td>
      <td>-9.20</td>
    </tr>
  </tbody>
</table>
</div>



### ツリーマップの色を指定する

利益が出ている箇所を緑色、損が出ている箇所を赤色にします。


```python
colors = []
percents = []
for p_or_l, oac in zip(df["評価損益"], df["取得価額"]):
    percent = p_or_l / oac * 100

    if p_or_l > 0:
        g = np.min([percent * 255 / 100 + 100, 255.0])
        color = "#%02x%02x%02x" % (0, int(g), 0)
        colors.append(color)
    else:
        r = np.min([-percent * 255 / 100 + 100, 255])
        color = "#%02x%02x%02x" % (int(r), 0, 0)
        colors.append(color)

    percents.append(percent)

print(df["銘柄名"].values)
print(colors)
print(percents)
```

    ['PBR' 'GOOG' 'FLNG' 'ZIM' 'GOGL' '3466\nラサールロジ']
    ['#00ff00', '#00b900', '#00ff00', '#00f600', '#a20000', '#730000']
    [62.094763092269325, 33.670033670033675, 224.4005641748942, 57.43840887511868, -24.376039933444257, -5.8898847631241935]
    

### ツリーマップを表示する

損益を色で表現し、損益の割合もツリーマップ上に表示してみます。
日本語が文字化けしていないのは冒頭で `import japanize_matplotlib` としているためです。


```python
current_prices = [cp for cp in df["現在の価額"]]
labels = [
    f"{name}\n{np.round(percent, 2)}％".replace("-", "▼")
    for name, percent in zip(df["銘柄名"], percents)
]

plt.figure(figsize=(10, 10))
plt.rcParams["font.size"] = 18
squarify.plot(current_prices, color=colors, label=labels)
plt.axis("off")
plt.show()
```


    
![png](/images/visualize/category-groupby/treemap_files/treemap_8_0.png)
    


### キャッシュの表示を追加する

キャッシュの表示も追加してみます。色はグレーにします。


```python
plt.figure(figsize=(10, 10))
plt.rcParams["font.size"] = 18
squarify.plot(
    current_prices + [3500], color=colors + ["#ccc"], label=labels + ["キャッシュ"]
)
plt.axis("off")
plt.show()
```


    
![png](/images/visualize/category-groupby/treemap_files/treemap_10_0.png)
    

