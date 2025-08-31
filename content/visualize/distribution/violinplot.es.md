---
title: "Violin plot"
pre: "6.2.4 "
weight: 4
title_replace: "Creación de gráficos de violín en python"
---

El diagrama de violín es un diagrama de caja y bigotes con un gráfico de densidad girado 90 grados a cada lado. El diagrama de violín permite comparar la distribución de los valores de varios grupos.

{{% notice tip %}}[Violin plot - Wikipedia](https://en.wikipedia.org/wiki/Violin_plot)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("iris")

sns.set(rc={"figure.figsize": (12, 8)})
sns.violinplot(x=df["species"], y=df["sepal_length"])
```


    
![png](/images/visualize/distribution/violinplot_files/violinplot_1_1.png)
    

