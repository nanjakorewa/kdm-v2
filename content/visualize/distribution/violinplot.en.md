---
title: "Violin plot"
pre: "6.2.4 "
weight: 4
title_replace: "Creating violin plots in python"
---

Violin plot is a box-and-whisker diagram with a density graph rotated 90 degrees on each side. Violin plot allows comparison of the distribution of values for several groups.

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
    

