---
title: "Histogram"
pre: "6.2.1 "
weight: 1
title_replace: "Memplot histogram dalam python"
---

Mirip dengan plot densitas, ini memvisualisasikan bagaimana data numerik didistribusikan.

{{% notice document %}}
[sns.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn.histplot)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import seaborn as sns


df = sns.load_dataset("iris")
sns.set(rc={"figure.figsize": (12, 8)})
sns.histplot(data=df["sepal_length"], binwidth=0.5)
```


    
![png](/images/visualize/distribution/histogram_files/histogram_1_1.png)
    

