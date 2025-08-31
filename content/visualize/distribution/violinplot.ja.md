---
title: "バイオリン図"
pre: "6.2.4 "
weight: 4
title_replace: "pythonでバイオリン図を作成する"
---

箱ひげ図の両脇に90度回転させた密度グラフを付加したものであり、複数のグループの数値の分布を比較することができます。

{{% notice tip %}}[バイオリン図 - Wikipedia](https://ja.wikipedia.org/wiki/%E3%83%90%E3%82%A4%E3%82%AA%E3%83%AA%E3%83%B3%E5%9B%B3)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("iris")

sns.set(rc={"figure.figsize": (12, 8)})
sns.violinplot(x=df["species"], y=df["sepal_length"])
```



    
![png](/images/visualize/distribution/violinplot_files/violinplot_1_1.png)
    

