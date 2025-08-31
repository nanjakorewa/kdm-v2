---
title: "Pencilan dan Kekokohan"
pre: "2.1.3 "
weight: 3
title_suffix: "Menangani outlier dan melakukan regresi linier yang kuat dalam python"
---

<div class="pagetop-box">
    <p>Outlier adalah istilah umum untuk nilai yang anomali (sangat besar atau sebaliknya kecil) dibandingkan dengan nilai lainnya. Nilai apa yang merupakan outlier tergantung pada pengaturan masalah dan sifat data. Pada halaman ini, kita akan mengulas perbedaan hasil antara "regresi dengan kesalahan kuadrat" dan "regresi dengan Huber loss" untuk data dengan outlier.</p>
</div>


```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
```

## Visualisasi kerugian Huber.


```python
def huber_loss(y_pred: float, y: float, delta=1.0):
    """HuberLoss"""
    huber_1 = 0.5 * (y - y_pred) ** 2
    huber_2 = delta * (np.abs(y - y_pred) - 0.5 * delta)
    return np.where(np.abs(y - y_pred) <= delta, huber_1, huber_2)


delta = 1.5
x_vals = np.arange(-2, 2, 0.01)
y_vals = np.where(
    np.abs(x_vals) <= delta,
    0.5 * np.square(x_vals),
    delta * (np.abs(x_vals) - 0.5 * delta),
)

# plot grafik
fig = plt.figure(figsize=(8, 8))
plt.plot(x_vals, x_vals ** 2, "red", label=r"$(y-\hat{y})^2$")  ## Kesalahan kuadrat
plt.plot(x_vals, np.abs(x_vals), "orange", label=r"$|y-\hat{y}|$")  ## Kesalahan mutlak
plt.plot(
    x_vals, huber_loss(x_vals * 2, x_vals), "green", label=r"huber-loss"
)  # Kesalahan huber
plt.axhline(y=0, color="k")
plt.grid(True)
plt.legend()
plt.show()
```


    
![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_5_0.png)
    


## Perbandingan dengan metode kuadrat terkecil
### Mempersiapkan data untuk percobaan
Untuk membandingkan regresi dengan Huber loss dan regresi linier normal, kita hanya memiliki satu pencilan.


```python
N = 30
x1 = np.array([i for i in range(N)])
x2 = np.array([i for i in range(N)])
X = np.array([x1, x2]).T
epsilon = np.array([np.random.random() for i in range(N)])
y = 5 * x1 + 10 * x2 + epsilon * 10
y[5] = 500

plt.figure(figsize=(8, 8))
plt.plot(x1, y, "ko", label="data")
plt.legend()
plt.show()
```


    
![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_7_0.png)
    


### Bandingkan dengan fitting least squares dan regresi ridge
{{% notice document %}}
[sklearn.linear_model.HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor)
{{% /notice %}}

{{% notice seealso %}}
[Pre-processing method: labeling outliers①](https://k-dm.work/ja/prep/numerical/add_label_to_anomaly/)
{{% /notice %}}


```python
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.linear_model import LinearRegression


plt.figure(figsize=(8, 8))
huber = HuberRegressor(alpha=0.0, epsilon=3)
huber.fit(X, y)
plt.plot(x1, huber.predict(X), "green", label="regresi huber")

ridge = Ridge(alpha=0.0, random_state=0)
ridge.fit(X, y)
plt.plot(x1, ridge.predict(X), "orange", label="regresi ridge regression")

lr = LinearRegression()
lr.fit(X, y)
plt.plot(x1, lr.predict(X), "r-", label="regresi least square")
plt.plot(x1, y, "x")

plt.plot(x1, y, "ko", label="data")
plt.legend()
plt.show()
```


    
![png](/images/basic/regression/03_Robust_Regression_files/03_Robust_Regression_9_0.png)
    

