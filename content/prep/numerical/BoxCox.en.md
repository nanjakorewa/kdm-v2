---
title: "BoxCox transformation"
pre: "3.3.2 "
weight: 2
title_replace: "Perform BoxCox transform in python"
---

<div class="pagetop-box">
    <p>Box-Cox transformation is a transformation technique that brings numerical data closer to a normal distribution. Note that it cannot be applied to data containing negative values, and that it will not yield valid results if the distribution is too different compared to the normal distribution.</p>
</div>

{{% notice document %}}
[scipy.stats.boxcox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html)
{{% /notice %}}

The following transformation is applied to the numbers to make the shape of the distribution closer to a normal distribution.

$
y = \begin{cases} 
\displaystyle \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0\\ 
\log x & \lambda = 0\end{cases}
$

From the form of the equation, \\(x\\) must always take on non-negative values to apply this transformation to numerical data. If negative values are contained, one might add a constant or use the YeoJonson transformation to make everything greater than 0.


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
    

