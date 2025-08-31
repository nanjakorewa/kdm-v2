---
title: "Decision Tree (Classification)"
pre: "2.3.1 "
weight: 1
searchtitle: "Running decision trees (classification) in python"
---

<div class="pagetop-box">
    <p>A decision tree (classification) is a type of model that uses a combination of rules to classify. The collection of rules is represented by a tree-shaped graph (tree structure), which is easy to interpret. This page performs decision tree classification and further visualizes the resulting tree.</p>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
```

## Generate sample data

Generate sample data for 2-class classification.


```python
n_classes = 2
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=2,
    n_classes=n_classes,
    n_clusters_per_class=1,
)
```

## Create a decision tree

Train the model with `DecisionTreeClassifier(criterion="gini").fit(X, y)` to visualize the decision boundaries of the created tree.
The `criterion="gini"` is an option to specify an indicator to determine the branching.

{{% notice document %}}
[sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier)
{{% /notice %}}


```python
# Train Decision Tree Classifier
clf = DecisionTreeClassifier(criterion="gini").fit(X, y)

# Dataset for color map of decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Visualize decision boundaries
plt.figure(figsize=(8, 8))
plt.tight_layout()
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1)
plt.xlabel("x1")
plt.ylabel("x2")

# Separate color plots for each label
for i, color, label_name in zip(range(n_classes), ["r", "b"], ["A", "B"]):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color, label=label_name, cmap=plt.cm.Pastel1)

plt.legend()
plt.show()
```


    
![png](/images/basic/tree/Decision_Tree_Classifier_files/Decision_Tree_Classifier_7_0.png)
    


## Outputs decision tree structure as an image

{{% notice document %}}
[sklearn.tree.plot_tree — scikit-learn 1.0.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html)
{{% /notice %}}


```python
plt.figure()
clf = DecisionTreeClassifier(criterion="gini").fit(X, y)
plt.figure(figsize=(12, 12))
plot_tree(clf, filled=True)
plt.show()
```

    
![png](/images/basic/tree/Decision_Tree_Classifier_files/Decision_Tree_Classifier_9_1.png)
    

