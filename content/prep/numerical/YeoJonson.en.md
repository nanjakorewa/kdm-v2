---
title: "YeoJonson transformation"
pre: "3.3.3 "
weight: 3
title_replace: "Performing YeoJonson transformation in python"
---

<div class="pagetop-box">
    <p>YeoJonson transform is one of the transform methods that bring numerical data closer to a normal distribution. Unlike the BoxCox transformation, which cannot be applied to data containing negative values, it can be applied even when negative values are included.</p>
</div>


{{% notice document %}}
[scipy.stats.yeojohnson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html)
{{% /notice %}}

Unlike the box-cox transformation, this method can be used even when negative values are included.


{{% notice ref %}}
I. Yeo and R.A. Johnson, “A New Family of Power Transformations to Improve Normality or Symmetry”, Biometrika 87.4 (2000):
{{% /notice %}}


```python
from scipy import stats
import matplotlib.pyplot as plt

x = stats.loggamma.rvs(1, size=1000) - 0.5
plt.hist(x)
plt.axvline(x=0, color="r")  # verify that there is data below 0 as well
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
    

