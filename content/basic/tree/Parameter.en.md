---
title: "Decision Tree Parameters | Examine the influence of each parameter of the decision tree"
linkTitle: "Decision Tree Parameters"
seo_title: "Decision Tree Parameters | Examine the influence of each parameter of the decision tree"
pre: "2.3.3 "
weight: 3
searchtitle: "Examine the influence of each parameter of the decision tree"
---

<div class="pagetop-box">
    <p>There are various parameters in a decision tree, and the results change depending on how they are specified. In this page, we will try to visualize and check how each parameter works.</p>
</div>

- `max_depth` specifies the maximum depth of the tree
- `min_samples_split` specifies the minimum number of data required to create a branch.
- `min_samples_leaf` specifies the minimum number of data required to create a leaf.
- `max_leaf_nodes` specifies the maximum number of leaves.
- `ccp_alpha` is a parameter for pruning the decision tree to account for tree complexity
- `class_weight` specifies the weighting of classes in classification.

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

## Applying a decision tree to simple dataset


```python
# dataset
X, y = make_regression(n_samples=100, n_features=2, random_state=11)

# train decision tree
dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X, y)

# visualize
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
When the value of `max_depth` is large, a deeper and more complex tree is created.
This can represent complex rules, but may be over-fitting if the number of data is small.


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

Specifies the minimum number of data required to create a single split.
Smaller numbers of `min_samples_split` allow for more detailed rules. If you increase the number, you can avoid over-fittinging.


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

This parameter penalizes the complexity of the tree. The higher the value of `ccp_alpha`, the simpler the tree will be.

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

This parameter specifies the number of leaves that will eventually be created. The number of `max_leaf_nodes` matches the number of parcels.


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
    


## when dataset contain outliers

Specify which criterion to apply when creating a branch.
Let's see how the tree changes when `criterion="squared_error"` is specified with outliers.
Since `squared_error` penalizes outliers more strongly than `absolute_error`, it is expected that a decision tree branch will be created if `squared_error` is specified.


```python
## Multiply some data values by 5 as outlier
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
    

