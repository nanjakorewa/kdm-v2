---
title: "Pohon keputusan (regresi) | Memvisualisasikan pohon keputusan (regresi) dalam python"
linkTitle: "Pohon keputusan (regresi)"
seo_title: "Pohon keputusan (regresi) | Memvisualisasikan pohon keputusan (regresi) dalam python"
pre: "2.3.2 "
weight: 2
searchtitle: "Memvisualisasikan pohon keputusan (regresi) dalam python"
---

<div class="pagetop-box">
    <p>Pohon keputusan (regresi) adalah jenis model yang menggunakan kombinasi aturan. Kumpulan aturan diwakili oleh grafik berbentuk pohon (struktur pohon), yang mudah diinterpretasikan. Halaman ini menjalankan regresi pohon keputusan dan selanjutnya memvisualisasikan pohon yang dihasilkan.</p>
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

## Membuat data sampel untuk pohon keputusan

```python
X, y = make_regression(n_samples=100, n_features=2, random_state=777)
plt.figure(figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```


    
![png](/images/basic/tree/Decision_Tree_Regressor_files/Decision_Tree_Regressor_6_0.png)
    


## Periksa bagaimana pohon regresi bercabang


```python
tree = DecisionTreeRegressor(max_depth=3, random_state=117117)
model = tree.fit(X, y)
viz = dtreeviz(tree, X, y, target_name="y")
viz.save("./regression_tree.svg")
```

### Memvisualisasikan percabangan pohon regresi


```python
from IPython.display import SVG

SVG(filename="regression_tree.svg")
```




    
![svg](/images/basic/tree/Decision_Tree_Regressor_files/Decision_Tree_Regressor_10_0.svg)
    


