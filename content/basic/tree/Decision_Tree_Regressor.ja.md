---
title: "決定木(回帰)"
pre: "2.3.2 "
weight: 2
title_suffix: "について仕組みを理解する"
---

{{% youtube "E5WOgzoEs1M" %}}

<div class="pagetop-box">
    <p><b>決定木（回帰）</b>とは、ルールの組合せで分類をするモデルの一種。ルールの集まりは木の形をしたグラフ（<b>木構造</b>）で表現されていて解釈がしやすいです。</p>
    <p>このページでは決定木の回帰を実行し、さらにその結果できた木を可視化します。</p>
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

## 決定木を作るためのサンプルデータを作成


```python
X, y = make_regression(n_samples=100, n_features=2, random_state=777)
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```


    
![png](/images/basic/tree/Decision_Tree_Regressor_files/Decision_Tree_Regressor_6_0.png)
    


## 回帰木の分岐の仕方を確認する


```python
tree = DecisionTreeRegressor(max_depth=3, random_state=117117)
model = tree.fit(X, y)
viz = dtreeviz(tree, X, y, target_name="y")
viz.save("./regression_tree.svg")
```

### 回帰木の分岐を可視化


```python
from IPython.display import SVG

SVG(filename="regression_tree.svg")
```




    
![svg](/images/basic/tree/Decision_Tree_Regressor_files/Decision_Tree_Regressor_10_0.svg)
    


