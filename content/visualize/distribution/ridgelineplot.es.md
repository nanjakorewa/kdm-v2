---
title: "parcela de la cresta"
pre: "6.2.3 "
weight: 3
title_replace: "Creación de un gráfico de cresta en python"
---

Gráficos utilizados para visualizar la distribución de múltiples grupos y sus diferencias. Como el gráfico de distribución está superpuesto, es fácil visualizar pequeñas diferencias en la distribución y diferencias/cambios en la posición de los vértices de cada grupo.

{{% notice document %}}
[ridgeplot: beautiful ridgeline plots in Python](https://github.com/tpvasconcelos/ridgeplot)
{{% /notice %}}

```python
import numpy as np
import seaborn as sns
from ridgeplot import ridgeplot


# Lista de columnas a visualizar
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Datos de la muestra
df = sns.load_dataset("iris")
df = df[columns]

# parcela de la cresta
fig = ridgeplot(
samples=df.values.T, labels=columns, colorscale="viridis", coloralpha=0.6
)
fig.update_layout(height=500, width=800)
fig.show()
```

![png](/images/visualize/distribution/ridgeline.png)!
