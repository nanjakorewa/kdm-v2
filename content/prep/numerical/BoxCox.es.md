---
title: "Transformaciones BoxCox"
pre: "3.3.2 "
weight: 2
title_replace: "Realizar la transformaciones BoxCox en python"
---

<div class="pagetop-box">
    <p>La transformación de Box-Cox es una técnica de transformación que acerca los datos numéricos a una distribución normal. Tenga en cuenta que no puede aplicarse a datos que contengan valores negativos, y que no dará resultados válidos si la distribución es demasiado diferente en comparación con la distribución normal.</p>
</div>

{{% notice document %}}
[scipy.stats.boxcox](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html)
{{% /notice %}}

Se aplica la siguiente transformación a los números para que la forma de la distribución se acerque más a una distribución normal.

$
y = \begin{cases} 
\displaystyle \frac{x^\lambda - 1}{\lambda} & \lambda \neq 0\\ 
\log x & \lambda = 0\end{cases}
$

Por la forma de la ecuación, \\(x\\) debe tomar siempre valores no negativos para aplicar esta transformación a los datos numéricos. Si se contienen valores negativos, se podría añadir una constante para que todo sea mayor que 0. O bien, se puede utilizar la transformación de YeoJonson.


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
    

