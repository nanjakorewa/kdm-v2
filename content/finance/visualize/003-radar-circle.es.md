---
title: "Carta del radar"
pre: "7.2.3 "
weight: 3
searchtitle: "Trazado de un gráfico de radar en python"
---

Un gráfico de radar es un método para comparar varios elementos juntos. Es útil cuando se comprueba si varios elementos están equilibrados en alto o en bajo.
Es más fácil comparar cuando todos los elementos son "cuanto más alto, mejor" o "cuanto más bajo, mejor".

> Un gráfico de radar es un gráfico que expresa una variable con múltiples elementos en un polígono regular sin convertirlo en una relación de composición. El centro del gráfico es 0, y cuanto mayor sea el valor de cada elemento, más hacia afuera se representa. Se utiliza principalmente para comparar el rendimiento de las entidades que tienen estos elementos como atributos.


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

## Trazado de una carta de radar

- [matplotlib.projections](https://matplotlib.org/stable/api/projections_api.html)
- [set_theta_offset(offset)](https://matplotlib.org/stable/api/projections_api.html#matplotlib.projections.polar.PolarAxes.set_theta_offset)


```python
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2.0)
ax.set_theta_direction(-1)

# ajustar la posición de cada etiqueta
angles = [2 * n * pi / len(df.columns) for n in range(len((df.columns)))]
plt.xticks(angles, df.columns, size=20)
ax.set_rlabel_position(0)
plt.yticks([1, 2, 3], ["★", "★★", "★★★"], color="grey", size=13)
plt.ylim(0, 3.5)

# Rellenar el área especificada
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
    

