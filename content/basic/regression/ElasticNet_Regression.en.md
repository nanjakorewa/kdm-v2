---
title: "Elastic Net Regression"
pre: "2.1.5 "
weight: 5
title_suffix: "Blend the strengths of L1 and L2"
---

{{% summary %}}
- Elastic Net mixes the L1 (lasso) and L2 (ridge) penalties to balance sparsity and stability.
- It can retain groups of strongly correlated features while adjusting their importance collectively.
- Tuning both \(\alpha\) and `l1_ratio` with cross-validation makes it easy to strike the bias–variance balance.
- Standardizing features and allowing enough iterations improve numerical stability for the optimizer.
{{% /summary %}}

## Intuition
Lasso can zero out coefficients and perform feature selection, but when features are highly correlated it may choose only one and discard the rest. Ridge shrinks coefficients smoothly and remains stable, yet never sets them exactly to zero. Elastic Net combines both penalties so that correlated features can survive together while unimportant coefficients are driven toward zero.

## Mathematical formulation
Elastic Net minimizes

$$
\min_{\boldsymbol\beta, b} \sum_{i=1}^{n} \left( y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b) \right)^2 + \alpha \left( \rho \lVert \boldsymbol\beta \rVert_1 + (1 - \rho) \lVert \boldsymbol\beta \rVert_2^2 \right),
$$

where \(\alpha > 0\) is the regularization strength and \(\rho \in [0,1]\) (`l1_ratio`) controls the L1/L2 trade-off. Moving \(\rho\) between 0 and 1 lets you explore the spectrum between ridge and lasso.

## Experiments with Python
Below we use `ElasticNetCV` to choose \(\alpha\) and `l1_ratio` simultaneously, then examine coefficients and performance.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Generate data with strong feature correlations
X, y, coef = make_regression(
    n_samples=500,
    n_features=30,
    n_informative=10,
    noise=15.0,
    coef=True,
    random_state=123,
)

# Duplicate several features to amplify correlations
X = np.hstack([X, X[:, :5] + np.random.normal(scale=0.1, size=(X.shape[0], 5))])
feature_names = [f"x{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Tune alpha and l1_ratio jointly
enet_cv = ElasticNetCV(
    l1_ratio=[0.2, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=np.logspace(-3, 1, 30),
    cv=5,
    random_state=42,
    max_iter=5000,
)
enet_cv.fit(X_train, y_train)

print("Best alpha:", enet_cv.alpha_)
print("Best l1_ratio:", enet_cv.l1_ratio_)

# Fit with the best hyperparameters and evaluate
enet = ElasticNet(alpha=enet_cv.alpha_, l1_ratio=enet_cv.l1_ratio_, max_iter=5000)
enet.fit(X_train, y_train)

train_pred = enet.predict(X_train)
test_pred = enet.predict(X_test)

print("Train R^2:", r2_score(y_train, train_pred))
print("Test R^2:", r2_score(y_test, test_pred))
print("Test RMSE:", mean_squared_error(y_test, test_pred, squared=False))

# Inspect the largest coefficients
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": enet.coef_,
}).sort_values("coef", key=lambda s: s.abs(), ascending=False)
print(coef_df.head(10))
```

### Reading the results
- `ElasticNetCV` evaluates multiple L1/L2 combinations automatically and picks a good balance.
- When correlated features survive together, their coefficients tend to align in magnitude, which simplifies interpretation.
- If convergence is slow, standardize the inputs or raise `max_iter`.

## References
{{% references %}}
<li>Zou, H., &amp; Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net. <i>Journal of the Royal Statistical Society: Series B</i>, 67(2), 301–320.</li>
<li>Friedman, J., Hastie, T., &amp; Tibshirani, R. (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. <i>Journal of Statistical Software</i>, 33(1), 1–22.</li>
{{% /references %}}
