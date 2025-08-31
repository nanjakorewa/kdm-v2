---
title: "Plot ridgeline"
pre: "6.2.3 "
weight: 3
title_replace: "Membuat plot ridgeline dalam python"
---

Grafik yang digunakan untuk memvisualisasikan distribusi dari beberapa kelompok dan perbedaan-perbedaannya. Karena grafik distribusi ditumpangkan, maka mudah untuk memvisualisasikan perbedaan-perbedaan kecil dalam distribusi dan perbedaan-perbedaan/perubahan-perubahan dalam posisi simpul-simpul untuk setiap kelompok.

{{% notice document %}}
[ridgeplot: beautiful ridgeline plots in Python](https://github.com/tpvasconcelos/ridgeplot)
{{% /notice %}}

```python
import numpy as np
import seaborn as sns
from ridgeplot import ridgeplot


# Daftar kolom yang akan divisualisasikan
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Data sampel
df = sns.load_dataset("iris")
df = df[columns]

# plot ridgeline
fig = ridgeplot(
samples=df.values.T, labels=columns, colorscale="viridis", coloralpha=0.6
)
fig.update_layout(height=500, width=800)
fig.show()
```

![png](/images/visualize/distribution/ridgeline.png)!
