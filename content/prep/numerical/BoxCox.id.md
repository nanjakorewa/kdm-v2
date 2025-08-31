---
title: "BoxCox transformation"
pre: "3.3.2 "
weight: 2
title_replace: "Melakukan transformasi BoxCox dalam python"
---

<div class="pagetop-box">
    <p>Transformasi Box-Cox adalah teknik transformasi yang membawa data numerik lebih dekat ke distribusi normal. Perhatikan bahwa transformasi ini tidak dapat diterapkan pada data yang mengandung nilai negatif, dan bahwa transformasi ini tidak akan memberikan hasil yang valid jika distribusinya terlalu berbeda dibandingkan dengan distribusi normal.</p>
</div>

{{% notice document %}}
[scipy.stats.boxcox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html)
{{% /notice %}}

Transformasi berikut ini diterapkan pada angka-angka untuk membuat bentuk distribusi lebih dekat ke distribusi normal.

$
y = \begin{cases} 
\displaystyle \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0\\ 
\log x & \lambda = 0\end{cases}
$

Dari bentuk persamaan, $x$ harus selalu mengambil nilai non-negatif untuk menerapkan transformasi ini pada data numerik. Jika nilai negatif terkandung, seseorang dapat menambahkan konstanta atau menggunakan transformasi YeoJonson untuk membuat semuanya lebih besar dari 0.


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
    

