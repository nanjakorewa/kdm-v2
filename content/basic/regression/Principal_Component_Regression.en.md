---
title: "Principal Component Regression (PCR)"
pre: "2.1.8 "
weight: 8
title_suffix: "Mitigate multicollinearity via dimensionality reduction"
---

{{% summary %}}
- Principal Component Regression (PCR) applies PCA to compress features before fitting linear regression, reducing instability from multicollinearity.
- Principal components prioritize directions with large variance, filtering noisy axes while preserving informative structure.
- Choosing how many components to keep balances overfitting risk and computational cost.
- Proper preprocessing—standardization and handling missing values—lays the groundwork for accuracy and interpretability.
{{% /summary %}}

## Intuition
When predictors are highly correlated, least squares can yield wildly varying coefficients. PCR first performs PCA to combine correlated directions, then regresses on the leading principal components ordered by explained variance. By focusing on dominant variation, the resulting coefficients become more stable.

## Mathematical formulation
Apply PCA to the standardized design matrix \(\mathbf{X}\) and retain the top \(k\) eigenvectors. With principal component scores \(\mathbf{Z} = \mathbf{X} \mathbf{W}_k\), the regression model

$$
y = \boldsymbol{\gamma}^\top \mathbf{Z} + b
$$

is learned. Coefficients in the original feature space are recovered via \(\boldsymbol{\beta} = \mathbf{W}_k \boldsymbol{\gamma}\). The number of components \(k\) is selected using cumulative explained variance or cross-validation.

## Experiments with Python
We evaluate cross-validation scores of PCR on the diabetes dataset as we vary the number of components.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import japanize_matplotlib

X, y = load_diabetes(return_X_y=True)

def build_pcr(n_components):
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=0)),
        ("reg", LinearRegression()),
    ])

components = range(1, X.shape[1] + 1)
cv_scores = []
for k in components:
    model = build_pcr(k)
    score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    cv_scores.append(score.mean())

best_k = components[int(np.argmax(cv_scores))]
print("Best number of components:", best_k)

best_model = build_pcr(best_k).fit(X, y)
print("Explained variance ratio:", best_model["pca"].explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(components, [-s for s in cv_scores], marker="o")
plt.axvline(best_k, color="red", linestyle="--", label=f"best={best_k}")
plt.xlabel("Number of components k")
plt.ylabel("CV MSE (lower is better)")
plt.legend()
plt.tight_layout()
plt.show()
```

![principal-component-regression block 1](/images/basic/regression/principal-component-regression_block01_en.png)

### Reading the results
- As the number of components increases, the training fit improves, but cross-validated MSE reaches a minimum at an intermediate value.
- Inspecting the explained variance ratio reveals how much of the overall variability each component captures.
- Component loadings indicate which original features contribute most to each principal direction.

## References
{{% references %}}
<li>Jolliffe, I. T. (2002). <i>Principal Component Analysis</i> (2nd ed.). Springer.</li>
<li>Massy, W. F. (1965). Principal Components Regression in Exploratory Statistical Research. <i>Journal of the American Statistical Association</i>, 60(309), 234–256.</li>
{{% /references %}}
