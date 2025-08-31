---
title: "Linear Discriminant Analysis"
pre: "2.2.2 "
weight: 2
---

<div class="pagetop-box">
    <p>Linear Discriminant Analysis (LDA) is a method of drawing a boundary that can discriminate between classes based on the degree of data cohesion between classes and the degree of data variability between classes for two classes of data. It also allows for dimensionality reduction of the data based on the results obtained. On this page, we will visualize the decision boundaries obtained by LDA and visualize the results of dimensionality reduction.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

## Generate dataset


```python
n_samples = 200
X, y = make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=2)
X[:, 0] -= np.mean(X[:, 0])
X[:, 1] -= np.mean(X[:, 1])

fig = plt.figure(figsize=(7, 7))
plt.title("Scatter plots of data", fontsize=20)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
```


    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_5_0.png)
    


## Find decision boundaries by Linear discriminant analysis

{{% notice document %}}
[sklearn.discriminant_analysis.LinearDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html)
{{% /notice %}}


```python
# Find the decision boundary
clf = LinearDiscriminantAnalysis(store_covariance=True)
clf.fit(X, y)

# Check the decision boundary
w = clf.coef_[0]
wt = -1 / (w[1] / w[0])  ## Find the slope perpendicular to w
xs = np.linspace(-10, 10, 100)
ys_w = [(w[1] / w[0]) * xi for xi in xs]
ys_wt = [wt * xi for xi in xs]

fig = plt.figure(figsize=(7, 7))
plt.title("Visualize the slope of the decision boundary", fontsize=20)
plt.scatter(X[:, 0], X[:, 1], c=y)  # sample dataset
plt.plot(xs, ys_w, "-.", color="k", alpha=0.5)  # orientation of W
plt.plot(xs, ys_wt, "--", color="k")  # Orientation perpendicular to w

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()

# The result of transferring data to one dimension based on the vector w obtained
X_1d = clf.transform(X).reshape(1, -1)[0]
fig = plt.figure(figsize=(7, 7))
plt.title("Location of data when data is projected in one dimension", fontsize=15)
plt.scatter(X_1d, [0 for _ in range(n_samples)], c=y)
plt.show()
```


    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_7_0.png)
    



    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_7_1.png)
    


## Example with more than 2-dimensional data


```python
X_3d, y_3d = make_blobs(n_samples=200, centers=3, n_features=3, cluster_std=3)

# Distribution of sample data
fig = plt.figure(figsize=(7, 7))
plt.title("Scatter plots of data", fontsize=20)
ax = fig.add_subplot(projection="3d")
ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=y_3d)
plt.show()

# Apply LDA
clf_3d = LinearDiscriminantAnalysis()
clf_3d.fit(X_3d, y_3d)
X_2d = clf_3d.transform(X_3d)

# Location of data when data is projected in two dimensions
fig = plt.figure(figsize=(7, 7))

plt.title("Location of data when data is projected in two dimensions", fontsize=15)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_3d)
plt.show()
```


    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_9_0.png)
    



    
![png](/images/basic/classification/Linear_Discriminant_Analysis_files/Linear_Discriminant_Analysis_9_1.png)
    

