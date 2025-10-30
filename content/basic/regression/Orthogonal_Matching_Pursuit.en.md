---
title: "Orthogonal Matching Pursuit (OMP)"
pre: "2.1.12 "
weight: 12
title_suffix: "Greedy selection for sparse coefficients"
---

{{% summary %}}
- Orthogonal Matching Pursuit (OMP) selects the feature most correlated with the current residual at each step to build a sparse linear model.
- The algorithm solves a least-squares problem restricted to the selected features, keeping the coefficients interpretable.
- Instead of tuning a regularization strength, you directly control sparsity by specifying the number of features to keep.
- Standardizing features and checking for multicollinearity help the method remain stable when recovering sparse signals.
{{% /summary %}}

## Intuition
When only a handful of features truly matter, OMP adds the one that most reduces the residual error at each iteration. It is a cornerstone of dictionary learning and sparse coding, producing compact coefficient vectors that highlight the most relevant predictors.

## Mathematical formulation
Initialize the residual as \(\mathbf{r}^{(0)} = \mathbf{y}\). For each iteration \(t\):

1. Compute the inner product between every feature \(\mathbf{x}_j\) and the residual \(\mathbf{r}^{(t-1)}\), and pick the feature \(j\) with the largest absolute value.
2. Add \(j\) to the active set \(\mathcal{A}_t\).
3. Solve the least-squares problem restricted to \(\mathcal{A}_t\) to obtain \(\hat{\boldsymbol\beta}_{\mathcal{A}_t}\).
4. Update the residual \(\mathbf{r}^{(t)} = \mathbf{y} - \mathbf{X}_{\mathcal{A}_t} \hat{\boldsymbol\beta}_{\mathcal{A}_t}\).

Stop once you reach the desired sparsity or the residual is sufficiently small.

## Experiments with Python
The snippet below compares OMP and lasso on data with a sparse ground-truth coefficient vector.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)
n_samples, n_features = 200, 40
X = rng.normal(size=(n_samples, n_features))
true_coef = np.zeros(n_features)
true_support = [1, 5, 12, 33]
true_coef[true_support] = [2.5, -1.5, 3.0, 1.0]
y = X @ true_coef + rng.normal(scale=0.5, size=n_samples)

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=4).fit(X, y)
lasso = Lasso(alpha=0.05).fit(X, y)

print("OMP non-zero indices:", np.flatnonzero(omp.coef_))
print("Lasso non-zero indices:", np.flatnonzero(np.abs(lasso.coef_) > 1e-6))

print("OMP MSE:", mean_squared_error(y, omp.predict(X)))
print("Lasso MSE:", mean_squared_error(y, lasso.predict(X)))

coef_df = pd.DataFrame({
    "true": true_coef,
    "omp": omp.coef_,
    "lasso": lasso.coef_,
})
print(coef_df.head(10))
```

### Reading the results
- Setting `n_nonzero_coefs` to the true sparsity helps OMP recover the relevant features reliably.
- Compared with lasso, OMP produces exactly zero coefficients for unselected features.
- When features are highly correlated, the selection order can fluctuate, so preprocessing and feature engineering remain important.

## References
{{% references %}}
<li>Pati, Y. C., Rezaiifar, R., &amp; Krishnaprasad, P. S. (1993). Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition. In <i>Conference Record of the Twenty-Seventh Asilomar Conference on Signals, Systems and Computers</i>.</li>
<li>Tropp, J. A. (2004). Greed is Good: Algorithmic Results for Sparse Approximation. <i>IEEE Transactions on Information Theory</i>, 50(10), 2231–2242.</li>
{{% /references %}}
