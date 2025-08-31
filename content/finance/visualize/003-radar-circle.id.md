---
title: "Radar chart"
pre: "7.2.3 "
weight: 3
searchtitle: "Memplot grafik radar dalam python"
---

Diagram radar adalah salah satu metode untuk membandingkan beberapa item secara bersamaan. Ini berguna ketika memeriksa apakah beberapa item seimbang tinggi atau rendah.
Lebih mudah untuk membandingkan ketika semua item adalah "lebih tinggi lebih baik" atau "lebih rendah lebih baik".

> Grafik radar adalah grafik yang mengekspresikan variabel dengan banyak item pada poligon biasa tanpa mengubahnya menjadi rasio komposisi. Pusat grafik adalah 0, dan semakin besar nilai setiap item, semakin jauh ke luar itu diwakili. Ini terutama digunakan untuk membandingkan kinerja entitas yang memiliki item-item ini sebagai atribut.


```python
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

df = pd.DataFrame(
    index=["$AAA", "$BBB", "$CCC"],
    data={
        "EPS": [1, 2, 3],
        "Revenue": [3, 3, 2],
        "Guidance": [1, 2, 3],
        "D/E": [3, 2, 1],
        "PER": [1, 2, 3],
        "Dividend": [2, 3, 3],
    },
)
```

## Memplot grafik radar

- [matplotlib.projections](https://matplotlib.org/stable/api/projections_api.html)
- [set_theta_offset(offset)](https://matplotlib.org/stable/api/projections_api.html#matplotlib.projections.polar.PolarAxes.set_theta_offset)


```python
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2.0)
ax.set_theta_direction(-1)

# menyesuaikan posisi setiap label
angles = [2 * n * pi / len(df.columns) for n in range(len((df.columns)))]
plt.xticks(angles, df.columns, size=20)
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3], ["★", "★★", "★★★"], color="grey", size=13)
plt.ylim(0, 3.5)

# mengisi area yang ditentukan
for ticker_symbol in ["$AAA", "$BBB", "$CCC"]:
    values = df.loc[ticker_symbol].values.flatten().tolist()
    ax.plot(
        angles + [0],
        values + [values[0]],
        linewidth=1,
        linestyle="solid",
        c="#000",
        label=ticker_symbol,
    )
    ax.fill(angles + [0], values + [values[0]], "#aaa", alpha=0.2)

plt.legend(bbox_to_anchor=(0.9, 1.1))
plt.show()
```


    
![png](/images/finance/visualize/003-radar-circle_files/003-radar-circle_3_0.png)
    

