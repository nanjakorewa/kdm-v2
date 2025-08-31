---
title: "Tree map"
pre: "6.1.2 "
weight: 2
not_use_colab: true
searchtitle: "Creación de un treemap en python"
---

Un treemap es un diagrama que puede utilizarse para visualizar datos numéricos con categorías jerárquicas. Un ejemplo típico es un mapa de calor del Nikkei 225 o del [S&P 500](https://finviz.com/map.ashx). Este cuaderno utiliza [squarify](https://github.com/laserson/squarify).



{{% notice info %}}
A treemap can also be created by using plotly.（
[Treemap charts with Python - Plotly](https://plotly.com/python/treemaps/)）
{{% /notice %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import squarify

np.random.seed(0)  # Fix random numbers

labels = ["A" * i for i in range(1, 5)]
sizes = [i * 10 for i in range(1, 5)]
colors = ["#%02x%02x%02x" % (i * 50, 0, 0) for i in range(1, 5)]

plt.figure(figsize=(5, 5))
squarify.plot(sizes, color=colors, label=labels)
plt.axis("off")
plt.show()
```


    
![png](/images/visualize/category-groupby/treemap_files/treemap_2_0.png)
    


## Visualizar mi cartera

Supongamos que tengo datos sobre el precio de adquisición y el precio actual de cada acción que poseo.
A partir de ahí, crearía un mapa de calor como [finviz](https://finviz.com/).


Supongamos que leemos los siguientes datos del csv.

<b>※Los datos mostrados aquí son ficticios.</b>


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



### Especifique el color del mapa de árbol

Verde para las zonas rentables y rojo para las zonas con pérdidas.


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
    

### Visualización de un mapa de árbol

Vamos a mostrar las ganancias/pérdidas en colores y el porcentaje de ganancias/pérdidas en el treemap.
Los caracteres japoneses no se confunden porque se utiliza `import japanize_matplotlib` al principio.


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
    


### Añadir la visualización de la caché al treemap

Añadamos también una visualización de la caché al treemap. El color debe ser gris.


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
    

