---
title: "Plot biola"
pre: "6.2.4 "
weight: 4
title_replace: "Membuat plot biola dalam python"
---

Violin plot adalah diagram kotak-dan-whisker dengan grafik densitas yang diputar 90 derajat pada setiap sisinya. Violin plot memungkinkan perbandingan distribusi nilai untuk beberapa kelompok.

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
    

