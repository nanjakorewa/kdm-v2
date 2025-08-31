---
title: "Ridge and Lasso regression"
pre: "2.1.2 "
weight: 2
searchtitle: "Ridge and Lasso regression in Python"
---

<div class="pagetop-box">
    <p>Ridge and Lasso regression are linear regression models that are designed to prevent models from overfitting by penalizing the coefficients of the model. Techniques such as these that attempt to prevent overfitting and increase generalizability are called regularization. On this page, we will use python to run least squares, ridge, and lasso regression and compare their coefficients.</p>
</div>

## Import required modules

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

## Generate regression data for experiments

{{% notice document %}}
[sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)
{{% /notice %}}

Artificially generate regression data using [sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html). Specify two features (`n_informative` = 2) as necessary for prediction. Otherwise, they are redundant features that are not useful for prediction.


```python
n_features = 5
n_informative = 2
X, y = make_regression(
    n_samples=500,
    n_features=n_features,
    n_informative=n_informative,
    noise=0.5,
    random_state=777,
)
X = np.concatenate([X, np.log(X + 100)], 1)  # Add redundant features
y_mean = y.mean(keepdims=True)
y_std = np.std(y, keepdims=True)
y = (y - y_mean) / y_std
```

## Compare least squares, ridge regression, and lasso regression models

Let's train each model with the same data and see the differences.
Normally, the coefficient alpha of the regularization term should be changed, but in this case, it is fixed.

{{% notice document %}}
- [sklearn.pipeline.make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)
- [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)
{{% /notice %}}


```python
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
rid_r = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2)).fit(X, y)
las_r = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.1)).fit(X, y)
```

## Compare the values of the coefficients for each model

Plot the absolute value of each coefficient. It can be seen from the graph that for linear regression, the coefficients are almost never zero. We can also see that Lasso regression has coefficients = 0 for all but the features needed for prediction.

Since there were two useful features (`n_informative = 2`), we can see that linear regression has unconditional coefficients even on features that are not useful, while Lasso-Ridge regression could not be narrowed down to two features. Still, it resulted in many unwanted features having coefficients of zero.

```python
feat_index = np.array([i for i in range(X.shape[1])])

plt.figure(figsize=(12, 4))
plt.bar(
    feat_index - 0.2,
    np.abs(lin_r.steps[1][1].coef_),
    width=0.2,
    label="LinearRegression",
)
plt.bar(feat_index, np.abs(rid_r.steps[1][1].coef_), width=0.2, label="RidgeRegression")
plt.bar(
    feat_index + 0.2,
    np.abs(las_r.steps[1][1].coef_),
    width=0.2,
    label="LassoRegression",
)

plt.xlabel(r"$\beta$")
plt.ylabel(r"$|\beta|$")
plt.grid()
plt.legend()
```

    
![png](/images/basic/regression/02_Ridge_and_Lasso_files/02_Ridge_and_Lasso_10_1.png)
    

