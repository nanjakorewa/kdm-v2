---
title: "BoxCox変換"
pre: "3.3.2 "
weight: 2
title_replace: "pythonでBoxCox変換を実行する"
---

{{% youtube "sMKsXqFWo-Q" %}}

<div class="pagetop-box">
    <p><b>Box-Cox変換</b>とは、数値データを正規分布に近づける変換手法のひとつです。負の値が含まれるデータには適用ができない点、また正規分布と比較して分布が異なりすぎている場合は適用しても有効な結果を得られない点に注意してください。</p>
</div>

{{% notice document %}}
[scipy.stats.boxcox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html)
{{% /notice %}}

以下のような変換を数値に対して適用して、分布の形を正規分布に近づけます。
$
y = \begin{cases} 
\displaystyle \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0\\ 
\log x & \lambda = 0\end{cases}
$

式の形から、この変換を数値データに適用するには、\\(x\\)は必ず非負の値を取る必要があります。負の値が含まれる場合には、全てを０より大きくするために定数を足すかYeoJonson変換を使用することが考えられます。


```python
from scipy import stats
import matplotlib.pyplot as plt

x = stats.loggamma.rvs(1, size=1000) + 10
plt.hist(x)
plt.show()
```


    
![png](/images/prep/numerical/BoxCox_files/BoxCox_1_0.png)
    



```python
import numpy as np
from scipy.stats import boxcox

plt.hist(boxcox(x))
plt.show()
```


    
![png](/images/prep/numerical/BoxCox_files/BoxCox_2_0.png)
    

