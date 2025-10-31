---
title: "Partial Least Squares Regression (PLS)"
pre: "2.1.9 "
weight: 9
title_suffix: "Learn latent factors aligned with the target"
---

{{% summary %}}
- Partial Least Squares (PLS) extracts latent factors that maximize the covariance between predictors and the target before performing regression.
- Unlike PCA, the learned axes incorporate target information, preserving predictive performance while reducing dimensionality.
- Tuning the number of latent factors stabilizes models in the presence of strong multicollinearity.
- Inspecting loadings reveals which combinations of features are most related to the target.
{{% /summary %}}

## Intuition
Principal component regression relies only on feature variance, so directions that matter for the target might be discarded. PLS builds latent factors while simultaneously considering predictors and response, retaining the information most useful for prediction. As a result you can work with a compact representation without sacrificing accuracy.

## Mathematical formulation
Given a predictor matrix \(\mathbf{X}\) and response vector \(\mathbf{y}\), PLS alternates updates of latent scores \(\mathbf{t} = \mathbf{X} \mathbf{w}\) and \(\mathbf{u} = \mathbf{y} c\) so that their covariance \(\mathbf{t}^\top \mathbf{u}\) is maximized. Repeating this procedure yields a set of latent factors on which a linear regression model

$$
\hat{y} = \mathbf{t} \boldsymbol{b} + b_0
$$

is fitted. The number of factors \(k\) is typically chosen via cross-validation.

## Experiments with Python
We compare PLS performance for different numbers of latent factors on the Linnerud fitness dataset.

```python
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_linnerud
import matplotlib.pyplot as plt
import japanize_matplotlib

data = load_linnerud()
X = data["data"]          # Workout routines and body measurements
y = data["target"][:, 0]  # Predict pull-up count only

components = range(1, min(X.shape[1], 6) + 1)
cv = KFold(n_splits=5, shuffle=True, random_state=0)

scores = []
for k in components:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("pls", PLSRegression(n_components=k)),
    ])
    score = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error").mean()
    scores.append(score)

best_k = components[int(np.argmax(scores))]
print("Best number of latent factors:", best_k)

best_model = Pipeline([
    ("scale", StandardScaler()),
    ("pls", PLSRegression(n_components=best_k)),
]).fit(X, y)

print("X loadings:\n", best_model["pls"].x_loadings_)
print("Y loadings:\n", best_model["pls"].y_loadings_)

plt.figure(figsize=(8, 4))
plt.plot(components, [-s for s in scores], marker="o")
plt.axvline(best_k, color="red", linestyle="--")
plt.xlabel("Number of latent factors")
plt.ylabel("CV MSE (lower is better)")
plt.tight_layout()
plt.show()
```

![partial-least-squares block 1](/images/basic/regression/partial-least-squares_block01_en.png)

### Reading the results
- Cross-validated MSE decreases as factors are added, reaches a minimum, and then worsens if you keep adding more.
- Inspecting `x_loadings_` and `y_loadings_` shows which features contribute most to each latent factor.
- Standardizing inputs ensures features measured on different scales contribute evenly.

## References
{{% references %}}
<li>Wold, H. (1975). Soft Modelling by Latent Variables: The Non-Linear Iterative Partial Least Squares (NIPALS) Approach. In <i>Perspectives in Probability and Statistics</i>. Academic Press.</li>
<li>Geladi, P., &amp; Kowalski, B. R. (1986). Partial Least-Squares Regression: A Tutorial. <i>Analytica Chimica Acta</i>, 185, 1–17.</li>
{{% /references %}}
