---
title: "YeoJonson変換"
pre: "3.3.3 "
weight: 3
title_replace: "pythonでYeoJonson変換を実行する"
---

<div class="pagetop-box">
    <p><b>YeoJonson変換</b>とは、数値データを正規分布に近づける変換手法のひとつです。負の値が含まれるデータには適用ができないBoxCox変換と異なり、負の値が含まれる場合でも適用することができます。</p>
</div>


{{% notice document %}}
[scipy.stats.yeojohnson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html)
{{% /notice %}}

数値を正規分布に近いかたちの分布に変換したい時の手法の一つ、box-cox変換と異なり負の値が含まれている場合でも対応できる。


{{% notice ref %}}
I. Yeo and R.A. Johnson, “A New Family of Power Transformations to Improve Normality or Symmetry”, Biometrika 87.4 (2000):
{{% /notice %}}


```python
from scipy import stats
import matplotlib.pyplot as plt

x = stats.loggamma.rvs(1, size=1000) - 0.5
plt.hist(x)
plt.axvline(x=0, color="r")  # 0以下にもデータがあることを確認する
plt.show()
```


    
![png](/images/prep/numerical/YeoJonson_files/YeoJonson_1_0.png)
    



```python
import numpy as np
from scipy.stats import yeojohnson

plt.hist(yeojohnson(x))
plt.show()
```


    
![png](/images/prep/numerical/YeoJonson_files/YeoJonson_2_0.png)
    

