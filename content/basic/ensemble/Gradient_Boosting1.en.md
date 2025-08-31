---
title: "Gradient boosting"
pre: "2.4.5 "
weight: 5
searchtitle: "Visualize the learning process of gradient boosting"
---

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.ensemble import GradientBoostingRegressor
```

## Fit a regression model to training data

Create experimental data, waveform data with trigonometric functions added together.


```python
# training dataset
X = np.linspace(-10, 10, 500)[:, np.newaxis]
noise = np.random.rand(X.shape[0]) * 10

# target
y = (
    (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10
    + 10
    + np.linspace(-10, 10, 500)
    + noise
)

# train gradient boosting
reg = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.5,
)
reg.fit(X, y)
y_pred = reg.predict(X)

# Check the fitting to the training data
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", label="training data")
plt.plot(X, y_pred, c="r", label="Predictions for the final created model", linewidth=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Degree of fitting to training dataset")
plt.legend()
plt.show()
```


    
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_5_0.png)
    


### Effect of loss function on results

Check how fitting to the training data changes when loss is changed to ["squared_error", "absolute_error", "huber", "quantile"]." It is expected that "absolute_error" and "huber" will not go on to predict outliers since the penalty for outliers is not as large as the squared error.

{{% notice document %}}
- [sklearn/ensemble/_gb_losses.py](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_gb_losses.py)
- [sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
{{% /notice %}}


```python
# training data
X = np.linspace(-10, 10, 500)[:, np.newaxis]

# prepare outliers
noise = np.random.rand(X.shape[0]) * 10
for i, ni in enumerate(noise):
    if i % 80 == 0:
        noise[i] = 70 + np.random.randint(-10, 10)

# target
y = (
    (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10
    + 10
    + np.linspace(-10, 10, 500)
    + noise
)

for loss in ["squared_error", "absolute_error", "huber", "quantile"]:
    # train gradient boosting
    reg = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.5,
        loss=loss,
    )
    reg.fit(X, y)
    y_pred = reg.predict(X)

    # Check the fitting to the training data.
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, c="k", marker="x", label="training dataset")
    plt.plot(X, y_pred, c="r", label="Predictions for the final created model", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Degree of fitting to training data, loss={loss}", fontsize=18)
    plt.legend()
    plt.show()
```


    
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_0.png)
    



    
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_1.png)
    



    
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_2.png)
    



    
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_7_3.png)
    


## Effect of `n_estimators` on results

You can see how the degree of improvement comes to a head when you increase `n_estimators` to some extent.


```python
from sklearn.metrics import mean_squared_error as MSE

# trainig dataset
X = np.linspace(-10, 10, 500)[:, np.newaxis]
noise = np.random.rand(X.shape[0]) * 10

# target
y = (
    (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10
    + 10
    + np.linspace(-10, 10, 500)
    + noise
)


# Try to create a model with different n_estimators
n_estimators_list = [(i + 1) * 5 for i in range(20)]
mses = []
for n_estimators in n_estimators_list:
    reg = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.3,
    )
    reg.fit(X, y)
    y_pred = reg.predict(X)
    mses.append(MSE(y, y_pred))

# Plotting mean_squared_error for different n_estimators
plt.figure(figsize=(10, 5))
plt.plot(n_estimators_list, mses, "x")
plt.xlabel("n_estimators")
plt.ylabel("Mean Squared Error(training data)")
plt.grid()
plt.show()
```


    
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_9_0.png)
    


### Impact of learning_rate on results

If `learning_rate` is too small, the accuracy does not improve, and if `learning_rate` is too large, it does not converge.


```python
# Try to create a model with different n_estimators
learning_rate_list = [np.round(0.1 * (i + 1), 1) for i in range(20)]
mses = []
for learning_rate in learning_rate_list:
    reg = GradientBoostingRegressor(
        n_estimators=30,
        learning_rate=learning_rate,
    )
    reg.fit(X, y)
    y_pred = reg.predict(X)
    mses.append(np.log(MSE(y, y_pred)))

# Plotting mean_squared_error for different n_estimators
plt.figure(figsize=(10, 5))
plt_index = [i for i in range(len(learning_rate_list))]
plt.plot(plt_index, mses, "x")
plt.xticks(plt_index, learning_rate_list, rotation=90)
plt.xlabel("learning_rate", fontsize=15)
plt.ylabel("log(Mean Squared Error) (training data)", fontsize=15)
plt.grid()
plt.show()
```


    
![png](/images/basic/ensemble/Gradient_Boosting1_files/Gradient_Boosting1_11_0.png)
    

