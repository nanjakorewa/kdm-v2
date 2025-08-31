---
title: "Transformación Yeo Johnson"
pre: "3.3.3 "
weight: 3
title_replace: "Realización de la transformaciones Yeo Johnson en python"
---

<div class="pagetop-box">
    <p>La transformación YeoJonson es uno de los métodos de transformación que acercan los datos numéricos a una distribución normal. A diferencia de la transformación BoxCox, que no puede aplicarse a datos que contienen valores negativos, puede aplicarse incluso cuando se incluyen valores negativos.</p>
</div>


{{% notice document %}}
[scipy.stats.yeojohnson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.yeojohnson.html)
{{% /notice %}}

A diferencia de la transformación box-cox, este método puede utilizarse incluso cuando se incluyen valores negativos.


{{% notice ref %}}
I. Yeo and R.A. Johnson, “A New Family of Power Transformations to Improve Normality or Symmetry”, Biometrika 87.4 (2000):
{{% /notice %}}


```python
from scipy import stats
import matplotlib.pyplot as plt

x = stats.loggamma.rvs(1, size=1000) - 0.5
plt.hist(x)
plt.axvline(x=0, color="r")  # verificar que también hay datos por debajo de 0
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
    

