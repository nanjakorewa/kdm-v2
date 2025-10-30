---
title: "Linear Regression and Ordinary Least Squares"
pre: "2.1.1 "
weight: 1
title_suffix: "Understand from first principles"
---

{{% summary %}}
- Linear regression models the linear relationship between inputs and outputs and provides a baseline that is both predictive and interpretable.
- Ordinary least squares estimates the coefficients by minimizing the sum of squared residuals, yielding a closed-form solution.
- The slope tells us how much the output changes when the input increases by one unit, while the intercept represents the expected value when the input is zero.
- When noise or outliers are large, consider standardization and robust variants so that preprocessing and evaluation remain reliable.
{{% /summary %}}

## Intuition
When the scatter plot of observations \((x_i, y_i)\) roughly forms a straight line, extending that line is a natural way to interpolate unknown inputs. Ordinary least squares draws a single straight line that passes as close as possible to all points, by making the overall deviation between the observations and the line as small as it can be.

## Mathematical formulation
A univariate linear model is written as

$$
y = w x + b.
$$

By minimizing the sum of squared residuals \(\epsilon_i = y_i - (w x_i + b)\)

$$
L(w, b) = \sum_{i=1}^{n} \big(y_i - (w x_i + b)\big)^2,
$$

we obtain the analytic solution

$$
w = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}, \qquad b = \bar{y} - w \bar{x},
$$

where \(\bar{x}\) and \(\bar{y}\) are the means of \(x\) and \(y\). The same idea extends to multivariate regression with vectors and matrices.

## Experiments with Python
The following code fits a simple regression line with `scikit-learn` and plots the result. The code is identical to the Japanese page so figures will match across languages.

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib  # optional; keeps Japanese labels if you run the notebook
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Generate data
n_samples = 100
X = np.linspace(-5, 5, n_samples)[:, np.newaxis]
epsilon = np.random.normal(scale=2, size=n_samples)
y = 2 * X.ravel() + 1 + epsilon  # true relationship y = 2x + 1 + noise

# Fit model with optional scaling (useful when extending to multiple features)
lin_reg = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
y_pred = lin_reg.predict(X)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="observations", c="orange")
plt.plot(X, y_pred, label="linear regression (OLS)")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

![linear-regression block 1](/images/basic/regression/linear-regression_block01.svg)

### Reading the results
- **Slope \(w\)**: indicates how much the output increases or decreases when the input grows by one unit. The estimate should be close to the true slope.
- **Intercept \(b\)**: shows the expected output when the input is 0, adjusting the vertical position of the line.
- Standardizing the features with `StandardScaler` stabilizes learning when inputs vary in scale.

## References
{{% references %}}
<li>Draper, N. R., &amp; Smith, H. (1998). <i>Applied Regression Analysis</i> (3rd ed.). John Wiley &amp; Sons.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
{{% /references %}}
