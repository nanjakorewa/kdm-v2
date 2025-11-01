---
title: "Parameter Pohon Keputusan | Memeriksa pengaruh dari setiap parameter pohon keputusan"
linkTitle: "Parameter Pohon Keputusan"
seo_title: "Parameter Pohon Keputusan | Memeriksa pengaruh dari setiap parameter pohon keputusan"
pre: "2.3.3 "
weight: 3
searchtitle: "Memeriksa pengaruh dari setiap parameter pohon keputusan"
---

<div class="pagetop-box">
    <p>Ada berbagai parameter dalam pohon keputusan, dan hasilnya berubah tergantung pada bagaimana parameter-parameter tersebut ditentukan. Dalam halaman ini, kita akan mencoba memvisualisasikan dan memeriksa bagaimana setiap parameter bekerja.</p>
</div>

- `max_depth` menentukan kedalaman maksimum pohon
- `min_samples_split` menentukan jumlah minimum data yang diperlukan untuk membuat cabang.
- `min_samples_leaf` menentukan jumlah minimum data yang diperlukan untuk membuat daun.
- `max_leaf_nodes` menentukan jumlah maksimum daun.
- `ccp_alpha` adalah parameter untuk memangkas pohon keputusan untuk memperhitungkan kompleksitas pohon
- `class_weight` menentukan bobot kelas dalam klasifikasi.

{{% notice document %}}
- [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)
- [parrt/dtreeviz](https://github.com/parrt/dtreeviz)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
from dtreeviz.trees import dtreeviz, rtreeviz_bivar_3D
```

## Menerapkan pohon keputusan ke dataset sederhana


```python
# dataset
X, y = make_regression(n_samples=100, n_features=2, random_state=11)

# pohon keputusan
dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X, y)

# memvisualisasikan
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="MPG",
    elev=40,
    azim=120,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_5_0.png)
    


## Mempelajari pohon keputusan dengan berbagai parameter

Mari kita periksa bagaimana pohon keputusan dengan struktur yang sedikit kompleks berperilaku ketika parameter pohon keputusan diubah. Pertama, periksa pohon keputusan dengan nilai default untuk semua parameter kecuali `max_depth=3`.


```python
X, y = make_regression(
    n_samples=500, n_features=2, effective_rank=4, noise=0.1, random_state=1
)
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

dt = DecisionTreeRegressor(max_depth=3, random_state=117117)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_7_0.png)
    



    
![png](/images/basic/tree/Parameter_files/Parameter_7_1.png)
    


### max_depth = 10

Ketika nilai `max_depth` besar, pohon yang lebih dalam dan lebih kompleks akan dibuat.
Ini dapat mewakili aturan yang kompleks, tetapi mungkin terlalu terlatih jika jumlah data sedikit.


```python
dt = DecisionTreeRegressor(max_depth=10, random_state=117117)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_9_0.png)
    


### max-depth=5


```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_11_0.png)
    


### min_samples_split=60

Menentukan jumlah minimum data yang diperlukan untuk membuat split tunggal.
Jumlah `min_samples_split` yang lebih kecil memungkinkan untuk aturan yang lebih rinci. Jika anda meningkatkan jumlahnya, anda dapat menghindari over-fittinging.


```python
dt = DecisionTreeRegressor(max_depth=5, min_samples_split=60, random_state=117117)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_13_0.png)
    


### ccp_alpha=0.4

Parameter ini memberikan penalti terhadap kompleksitas pohon. Semakin tinggi nilai `ccp_alpha`, semakin sederhana pohonnya.

```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117, ccp_alpha=0.4)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_15_0.png)
    


### max_leaf_nodes=5

Parameter ini menspesifikasikan jumlah daun yang pada akhirnya akan dibuat. Jumlah `max_leaf_nodes` sesuai dengan jumlah parsel.


```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117, max_leaf_nodes=5)
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_17_0.png)
    


## ketika dataset mengandung outlier

Tentukan kriteria mana yang akan diterapkan saat membuat cabang.
Mari kita lihat bagaimana pohon berubah ketika `kriteria="squared_error"` ditentukan dengan outlier.
Karena `squared_error` menghukum outlier lebih kuat daripada `absolute_error`, diharapkan cabang pohon keputusan akan dibuat jika `squared_error` ditentukan.


```python
## Kalikan beberapa nilai data dengan 5 sebagai pencilan
X, y = make_regression(n_samples=100, n_features=2, random_state=11)
y[1:20] = y[1:20] * 5
```


```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117, criterion="absolute_error")
dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_20_0.png)
    



```python
dt = DecisionTreeRegressor(max_depth=5, random_state=117117, criterion="squared_error")

dt.fit(X, y)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
t = rtreeviz_bivar_3D(
    dt,
    X,
    y,
    feature_names=["x1", "x2"],
    target_name="y",
    elev=40,
    azim=240,
    dist=8.0,
    show={"splits", "title"},
    ax=ax,
)
plt.show()
```


    
![png](/images/basic/tree/Parameter_files/Parameter_21_0.png)
    

