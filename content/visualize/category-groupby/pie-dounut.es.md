---
title: "Gráfico de rosquillas"
pre: "6.1.3 "
weight: 3
not_use_colab: true
searchtitle: "Creación de un gráfico de donuts en python"
---

Un gráfico de donuts es un tipo de gráfico circular que se utiliza para mostrar las proporciones por categoría, con un espacio en blanco en el centro. El espacio en blanco no tiene ningún significado especial, pero puede utilizarse para mostrar estadísticas generales (por ejemplo, "Total XXX yenes"). Los gráficos de donuts se crean en python utilizando [matplotlib.pyplot.pie](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pie.html).


```python
import matplotlib.pyplot as plt

# Datos
percent = [40, 20, 20, 10, 10]
explode = [0, 0, 0, 0, 0]
labels = ["米国", "エマージング", "日本", "欧州", "その他"]

percent.reverse()
explode.reverse()
labels.reverse()

# Crear un gráfico circular
plt.figure(figsize=(7, 7))
plt.pie(x=percent, labels=labels, explode=explode, autopct="%1.0f%%", startangle=90)

# Añade un círculo en blanco en el centro
background_color = "#fff"
p = plt.gcf()
p.gca().add_artist(plt.Circle((0, 0), 0.8, color=background_color))

plt.show()
```


    
![png](/images/visualize/category-groupby/pie-dounut_files/pie-dounut_1_0.png)
    

