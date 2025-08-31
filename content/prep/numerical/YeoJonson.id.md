---
title: "YeoJonson transformation"
pre: "3.3.3 "
weight: 3
title_replace: "Performing YeoJonson transformation in python"
---

<div class="pagetop-box">
    <p>Transformasi YeoJonson adalah salah satu metode transformasi yang membawa data numerik lebih dekat ke distribusi normal. Tidak seperti transformasi BoxCox, yang tidak dapat diterapkan pada data yang mengandung nilai negatif, transformasi ini dapat diterapkan bahkan ketika nilai negatif disertakan.</p>
</div>


{{% notice document %}}
[scipy.stats.yeojohnson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html)
{{% /notice %}}

Tidak seperti transformasi box-cox, metode ini bisa digunakan bahkan ketika nilai negatif disertakan.


{{% notice ref %}}
I. Yeo and R.A. Johnson, “A New Family of Power Transformations to Improve Normality or Symmetry”, Biometrika 87.4 (2000):
{{% /notice %}}


```python
from scipy import stats
import matplotlib.pyplot as plt

x = stats.loggamma.rvs(1, size=1000) - 0.5
plt.hist(x)
plt.axvline(x=0, color="r")  # verifikasi bahwa ada data di bawah 0 juga
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
    

