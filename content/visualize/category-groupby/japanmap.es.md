---
title: "Mapa de Japón"
pre: "6.1.1 "
weight: 1
not_use_colab: true
searchtitle: "Crear un mapa de Japón en python"
---

- [japanmap](https://pypi.org/project/japanmap/)
- [SaitoTsutomu/japanmap](https://github.com/SaitoTsutomu/japanmap)
- [Visualization of data by prefecture (page with library author's explanation of usage)](https://qiita.com/SaitoTsutomu/items/6d17889ba47357e44131)

Este paquete es útil para crear mapas de calor de Japón. Debe funcionar con python 3.9 o una versión superior.

## Mostrar mapa de Japón

```python
import matplotlib.pyplot as plt
import numpy as np
from japanmap import picture

np.random.seed(77)

plt.figure(figsize=(10, 10))
plt.xticks([])
plt.yticks([])
plt.imshow(picture())
```


    
![png](/images/visualize/category-groupby/japanmap_files/japanmap_3_1.png)
    


## Colorear el mapa de Japón

Por ejemplo, puedes colorear un mapa de Japón preparando el siguiente diccionario con los nombres de las prefecturas como clave y los nombres de los colores como valor.

```
{'北海道': '#a9e5bb', '青森': '#fcf6b1', '沖縄': '#fcf6b1'}
```


```python
colors = [
    "#e3170a",
    "#a9e5bb",
    "#fcf6b1",
    "#f7b32b",
    "#2d1e2f",
]

prefectures = [
    "北海道",
    "青森",
    "岩手",
    "宮城",
    "秋田",
    "山形",
    "福島",
    "茨城",
    "栃木",
    "群馬",
    "埼玉",
    "千葉",
    "東京",
    "神奈川",
    "新潟",
    "富山",
    "石川",
    "福井",
    "山梨",
    "長野",
    "岐阜",
    "静岡",
    "愛知",
    "三重",
    "滋賀",
    "京都",
    "大阪",
    "兵庫",
    "奈良",
    "和歌山",
    "鳥取",
    "島根",
    "岡山",
    "広島",
    "山口",
    "徳島",
    "香川",
    "愛媛",
    "高知",
    "福岡",
    "佐賀",
    "長崎",
    "熊本",
    "大分",
    "宮崎",
    "鹿児島",
    "沖縄",
]

pref_color_dict = {prefecture: np.random.choice(colors) for prefecture in prefectures}

plt.figure(figsize=(10, 10))
plt.xticks([])
plt.yticks([])
plt.imshow(picture(pref_color_dict))

print(f"入力：{pref_color_dict}")
```

    入力：{'北海道': '#2d1e2f', '青森': '#2d1e2f', '岩手': '#f7b32b', '宮城': '#e3170a', '秋田': '#e3170a', '山形': '#a9e5bb', '福島': '#2d1e2f', '茨城': '#f7b32b', '栃木': '#e3170a', '群馬': '#f7b32b', '埼玉': '#2d1e2f', '千葉': '#2d1e2f', '東京': '#fcf6b1', '神奈川': '#f7b32b', '新潟': '#f7b32b', '富山': '#a9e5bb', '石川': '#f7b32b', '福井': '#a9e5bb', '山梨': '#a9e5bb', '長野': '#2d1e2f', '岐阜': '#e3170a', '静岡': '#2d1e2f', '愛知': '#a9e5bb', '三重': '#e3170a', '滋賀': '#a9e5bb', '京都': '#fcf6b1', '大阪': '#f7b32b', '兵庫': '#2d1e2f', '奈良': '#f7b32b', '和歌山': '#a9e5bb', '鳥取': '#2d1e2f', '島根': '#e3170a', '岡山': '#2d1e2f', '広島': '#e3170a', '山口': '#a9e5bb', '徳島': '#f7b32b', '香川': '#2d1e2f', '愛媛': '#f7b32b', '高知': '#fcf6b1', '福岡': '#2d1e2f', '佐賀': '#2d1e2f', '長崎': '#2d1e2f', '熊本': '#fcf6b1', '大分': '#a9e5bb', '宮崎': '#fcf6b1', '鹿児島': '#a9e5bb', '沖縄': '#2d1e2f'}
    


    
![png](/images/visualize/category-groupby/japanmap_files/japanmap_5_1.png)
    


## Acercarse sólo a una región específica

También puedes ampliar una región específica de Japón (por ejemplo, Kanto, Kansai, etc.).


```python
from japanmap import get_data, groups, pref_map

print(f"groups: {groups}")
pref_map(
    groups["近畿"], cols=[np.random.choice(colors) for _ in groups["近畿"]], qpqo=get_data()
)
```

    groups: {'北海道': [1], '東北': [2, 3, 4, 5, 6, 7], '関東': [8, 9, 10, 11, 12, 13, 14], '中部': [15, 16, 17, 18, 19, 20, 21, 22, 23], '近畿': [24, 25, 26, 27, 28, 29, 30], '中国': [31, 32, 33, 34, 35], '四国': [36, 37, 38, 39], '九州': [40, 41, 42, 43, 44, 45, 46, 47]}
    




    
![svg](/images/visualize/category-groupby/japanmap_files/japanmap_7_1.svg)
    


