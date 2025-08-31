---
title: "密度プロット"
pre: "6.2.2 "
weight: 2
title_replace: "pythonで密度プロットをプロットする"
---

数値データがどのように分布しているかを可視化します。

{{% notice document %}}
[seaborn.kdeplot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import seaborn as sns


df = sns.load_dataset("iris")
sns.set(rc={"figure.figsize": (12, 8)})
sns.kdeplot(df["sepal_length"])
sns.kdeplot(df["sepal_width"])
sns.kdeplot(df["petal_length"])
sns.kdeplot(df["petal_width"])
plt.legend(labels=["sepal_length", "sepal_width", "petal_length", "petal_width"])
```



    
![png](/images/visualize/distribution/densityplot_files/densityplot_1_1.png)
    

