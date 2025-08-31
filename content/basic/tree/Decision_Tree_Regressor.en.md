---
title: "Decision tree (regression)"
pre: "2.3.2 "
weight: 2
searchtitle: "Visualizing decision trees (regression) in python"
---

<div class="pagetop-box">
    <p>A decision tree (regression) is a type of model that uses a combination of rules. The collection of rules is represented by a tree-shaped graph (tree structure), which is easy to interpret. This page runs a regression of a decision tree and further visualizes the resulting tree.</p>
</div>

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from dtreeviz.trees import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dtreeviz.trees import dtreeviz
```

{{% notice document %}}
[dtreeviz : Decision Tree Visualization](https://github.com/parrt/dtreeviz)
{{% /notice %}}

## Generate sample data for decision trees

```python
X, y = make_regression(n_samples=100, n_features=2, random_state=777)
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```


    
![png](/images/basic/tree/Decision_Tree_Regressor_files/Decision_Tree_Regressor_6_0.png)
    


## Check how the regression tree branches.


```python
tree = DecisionTreeRegressor(max_depth=3, random_state=117117)
model = tree.fit(X, y)
viz = dtreeviz(tree, X, y, target_name="y")
viz.save("./regression_tree.svg")
```

### Visualize the branching of a regression tree


```python
from IPython.display import SVG

SVG(filename="regression_tree.svg")
```




    
![svg](/images/basic/tree/Decision_Tree_Regressor_files/Decision_Tree_Regressor_10_0.svg)
    


