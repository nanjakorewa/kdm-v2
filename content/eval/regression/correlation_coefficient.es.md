---
title: "Coeficiente de correlación"
pre: "4.2.1 "
weight: 1
searchtitle: "Calcular el coeficiente de correlación en python"
---


El coeficiente de correlación mide la fuerza de una relación lineal entre dos datos o variables aleatorias.
Es un indicador que permite comprobar si existe un cambio de tendencia de forma lineal entre dos variables, que puede expresarse en la siguiente ecuación.

$
\frac{\Sigma_{i=1}^N (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\Sigma_{i=1}^N(x_i - \bar{x})^2 \Sigma_{i=1}^N(y_i - \bar{y})^2 }}
$

Tiene las siguientes propiedades

- -1 a menos o igual que 1.
- Si el coeficiente de correlación se acerca a 1, \\(x\\) aumenta → \\(y\\) también aumenta
- El valor del coeficiente de correlación no cambia cuando \\(x, y\\) se multiplican por un número bajo

## Calcular el coeficiente de correlación entre dos columnas numéricas


```python
import numpy as np

np.random.seed(777)  # para fijar números aleatorios
```


```python
import matplotlib.pyplot as plt
import numpy as np

x = [xi + np.random.rand() for xi in np.linspace(0, 100, 40)]
y = [yi + np.random.rand() for yi in np.linspace(1, 50, 40)]

plt.figure(figsize=(5, 5))
plt.scatter(x, y)
plt.show()

coef = np.corrcoef(x, y)
print(coef)
```


    
![png](/images/eval/regression/correlation_coefficient_files/correlation_coefficient_3_0.png)
    


    [[1.         0.99979848]
     [0.99979848 1.        ]]


## Calcular colectivamente el coeficiente de correlación entre múltiples variables

{{% notice document %}}
[pandas.io.formats.style.Styler.background_gradient](https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.background_gradient.html)
{{% /notice %}}


```python
import seaborn as sns

df = sns.load_dataset("iris")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



### Comprobar las COEFICIENCIAS DE CORRELACIÓN entre todas las variables

Utilizando el conjunto de datos del iris, veamos la correlación entre las variables.

{{% notice document %}}
- [The Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- [pandas.io.formats.style.Styler.background_gradient](https://pandas.pydata.org/docs/reference/api/pandas.io.formats.style.Styler.background_gradient.html)
{{% /notice %}}


```python
df.corr().style.background_gradient(cmap="YlOrRd")
```




<style type="text/css">
#T_dbd76_row0_col0, #T_dbd76_row1_col1, #T_dbd76_row2_col2, #T_dbd76_row3_col3 {
  background-color: #800026;
  color: #f1f1f1;
}
#T_dbd76_row0_col1 {
  background-color: #fede82;
  color: #000000;
}
#T_dbd76_row0_col2 {
  background-color: #aa0026;
  color: #f1f1f1;
}
#T_dbd76_row0_col3 {
  background-color: #c00225;
  color: #f1f1f1;
}
#T_dbd76_row1_col0, #T_dbd76_row1_col2, #T_dbd76_row1_col3, #T_dbd76_row2_col1 {
  background-color: #ffffcc;
  color: #000000;
}
#T_dbd76_row2_col0 {
  background-color: #b70026;
  color: #f1f1f1;
}
#T_dbd76_row2_col3, #T_dbd76_row3_col2 {
  background-color: #8b0026;
  color: #f1f1f1;
}
#T_dbd76_row3_col0 {
  background-color: #c80723;
  color: #f1f1f1;
}
#T_dbd76_row3_col1 {
  background-color: #fff9bd;
  color: #000000;
}
</style>
<table id="T_dbd76_">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th class="col_heading level0 col0" >sepal_length</th>
      <th class="col_heading level0 col1" >sepal_width</th>
      <th class="col_heading level0 col2" >petal_length</th>
      <th class="col_heading level0 col3" >petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_dbd76_level0_row0" class="row_heading level0 row0" >sepal_length</th>
      <td id="T_dbd76_row0_col0" class="data row0 col0" >1.000000</td>
      <td id="T_dbd76_row0_col1" class="data row0 col1" >-0.117570</td>
      <td id="T_dbd76_row0_col2" class="data row0 col2" >0.871754</td>
      <td id="T_dbd76_row0_col3" class="data row0 col3" >0.817941</td>
    </tr>
    <tr>
      <th id="T_dbd76_level0_row1" class="row_heading level0 row1" >sepal_width</th>
      <td id="T_dbd76_row1_col0" class="data row1 col0" >-0.117570</td>
      <td id="T_dbd76_row1_col1" class="data row1 col1" >1.000000</td>
      <td id="T_dbd76_row1_col2" class="data row1 col2" >-0.428440</td>
      <td id="T_dbd76_row1_col3" class="data row1 col3" >-0.366126</td>
    </tr>
    <tr>
      <th id="T_dbd76_level0_row2" class="row_heading level0 row2" >petal_length</th>
      <td id="T_dbd76_row2_col0" class="data row2 col0" >0.871754</td>
      <td id="T_dbd76_row2_col1" class="data row2 col1" >-0.428440</td>
      <td id="T_dbd76_row2_col2" class="data row2 col2" >1.000000</td>
      <td id="T_dbd76_row2_col3" class="data row2 col3" >0.962865</td>
    </tr>
    <tr>
      <th id="T_dbd76_level0_row3" class="row_heading level0 row3" >petal_width</th>
      <td id="T_dbd76_row3_col0" class="data row3 col0" >0.817941</td>
      <td id="T_dbd76_row3_col1" class="data row3 col1" >-0.366126</td>
      <td id="T_dbd76_row3_col2" class="data row3 col2" >0.962865</td>
      <td id="T_dbd76_row3_col3" class="data row3 col3" >1.000000</td>
    </tr>
  </tbody>
</table>



En el mapa de calor, es difícil ver dónde es mayor la correlación. Compruebe el gráfico de barras para ver qué variables tienen la mayor correlación con `longitud_de_sepal`.


{{% notice document %}}
[pandas.DataFrame.plot.bar](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.bar.html)
{{% /notice %}}


```python
df.corr()["sepal_length"].plot.bar(grid=True, ylabel="corr")
```


    
![png](/images/eval/regression/correlation_coefficient_files/correlation_coefficient_9_1.png)
    


## Cuando el coeficiente de correlación es bajo

Compruebe la distribución de los datos cuando el coeficiente de correlación es bajo y confirme que el coeficiente de correlación puede ser bajo incluso cuando existe una relación entre las variables.

{{% notice document %}}
[numpy.random.multivariate_normal — NumPy v1.22 Manual](numpy.random.multivariate_normal)
{{% /notice %}}


```python
n_samples = 1000

plt.figure(figsize=(12, 12))
for i, ci in enumerate(np.linspace(-1, 1, 16)):
    ci = np.round(ci, 4)

    mean = np.array([0, 0])
    cov = np.array([[1, ci], [ci, 1]])

    v1, v2 = np.random.multivariate_normal(mean, cov, size=n_samples).T

    plt.subplot(4, 4, i + 1)
    plt.plot(v1, v2, "x")
    plt.title(f"r={ci}")

plt.tight_layout()
plt.show()
```


    
![png](/images/eval/regression/correlation_coefficient_files/correlation_coefficient_11_0.png)
    


En algunos casos, existe una relación entre las variables aunque el coeficiente de correlación sea bajo.
Intentaremos crear un ejemplo de este tipo, aunque sea sencillo.


```python
import japanize_matplotlib
from sklearn import datasets

japanize_matplotlib.japanize()

n_samples = 1000
circle, _ = datasets.make_circles(n_samples=n_samples, factor=0.1, noise=0.05)
moon, _ = datasets.make_moons(n_samples=n_samples, noise=0.05)

corr_circle = np.round(np.corrcoef(circle[:, 0], circle[:, 1])[1, 0], 4)
plt.title(f"correlation coefficient={corr_circle}", fontsize=23)
plt.scatter(circle[:, 0], circle[:, 1])
plt.show()

corr_moon = np.round(np.corrcoef(moon[:, 0], moon[:, 1])[1, 0], 4)
plt.title(f"correlation coefficient={corr_moon}", fontsize=23)
plt.scatter(moon[:, 0], moon[:, 1])
plt.show()
```


    
![png](/images/eval/regression/correlation_coefficient_files/correlation_coefficient_13_0.png)
    



    
![png](/images/eval/regression/correlation_coefficient_files/correlation_coefficient_13_1.png)
    

