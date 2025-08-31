---
title: "Ridge and Lasso Regression"
pre: "2.1.2 "
weight: 2
title_suffix: "Intuition and Practice"
---

{{% youtube "rhGYOBrxPXA" %}}


<div class="pagetop-box">
  <p><b>Ridge</b> (L2) and <b>Lasso</b> (L1) add penalties on coefficient size to <b>reduce overfitting</b> and improve <b>generalization</b>. This is called <b>regularization</b>. We’ll fit ordinary least squares, Ridge, and Lasso under the same conditions and compare their behavior.</p>
  </div>

---

## 1. Intuition and formulas

Ordinary least squares minimizes the residual sum of squares
$$
\text{RSS}(\boldsymbol\beta, b) = \sum_{i=1}^n \big(y_i - (\boldsymbol\beta^\top \mathbf{x}_i + b)\big)^2.
$$
With many features, strong collinearity, or noisy data, coefficients can grow too large and overfit. Regularization adds a penalty on coefficient size to <b>shrink</b> them.

- <b>Ridge (L2)</b>
  $$
  \min_{\boldsymbol\beta, b}\; \text{RSS}(\boldsymbol\beta,b) + \alpha \lVert \boldsymbol\beta \rVert_2^2
  $$
  Shrinks all coefficients smoothly (rarely sets exactly to zero).

- <b>Lasso (L1)</b>
  $$
  \min_{\boldsymbol\beta, b}\; \text{RSS}(\boldsymbol\beta,b) + \alpha \lVert \boldsymbol\beta \rVert_1
  $$
  Sets some coefficients exactly to zero (feature selection).

> Heuristics:
> - Ridge “shrinks everything a bit” → stable, not sparse.
> - Lasso “shrinks some to zero” → sparse/parsimonious models, can be unstable under strong collinearity.

---

## 2. Setup

{{% notice tip %}}
With regularization, <b>feature scales matter</b>. Use <b>StandardScaler</b> so the penalty acts fairly across features.
{{% /notice %}}

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

---

## 3. Data: only 2 informative features

{{% notice document %}}
- [sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)
{{% /notice %}}

We create data with exactly two informative features (<code>n_informative=2</code>) and then add redundant transforms to simulate lots of non-useful features.

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

# Add redundant nonlinear transforms
X = np.concatenate([X, np.log(X + 100)], axis=1)

# Standardize target for fair scaling
y = (y - y.mean()) / np.std(y)
```

---

## 4. Train three models under the same pipeline

{{% notice document %}}
- [make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)  
- [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)  
- [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)  
- [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
{{% /notice %}}

```python
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
rid_r = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2)).fit(X, y)
las_r = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.1, max_iter=10_000)).fit(X, y)
```

> Lasso can converge slowly; in practice, increase <code>max_iter</code> if needed.

---

## 5. Compare coefficient magnitudes (plot)

- OLS: coefficients are rarely near zero.  
- Ridge: coefficients shrink broadly.  
- Lasso: some coefficients become exactly zero (feature selection).

```python
feat_index = np.arange(X.shape[1])

plt.figure(figsize=(12, 4))
plt.bar(feat_index - 0.25, np.abs(lin_r.steps[1][1].coef_), width=0.25, label="Linear")
plt.bar(feat_index,         np.abs(rid_r.steps[1][1].coef_), width=0.25, label="Ridge")
plt.bar(feat_index + 0.25,  np.abs(las_r.steps[1][1].coef_), width=0.25, label="Lasso")

plt.xlabel(r"coefficient index $\beta$")
plt.ylabel(r"$|\beta|$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

![png](/images/basic/regression/02_Ridge_and_Lasso_files/02_Ridge_and_Lasso_10_1.png)

{{% notice tip %}}
Here only two features are informative (<code>n_informative=2</code>). Lasso tends to zero-out many useless features (sparse model), while Ridge shrinks all coefficients smoothly to stabilize the solution.
{{% /notice %}}

---

## 6. Rough generalization check (CV)

Fixed <code>alpha</code> can mislead; compare via cross-validation.

```python
from sklearn.metrics import make_scorer, mean_squared_error

scorer = make_scorer(mean_squared_error, greater_is_better=False)

models = {
    "Linear": lin_r,
    "Ridge (α=2)": rid_r,
    "Lasso (α=0.1)": las_r,
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"{name:12s}  R^2 (CV mean±std): {scores.mean():.3f} ± {scores.std():.3f}")
```

{{% notice tip %}}
There’s no one-size-fits-all. With strong collinearity or small samples, Ridge/Lasso often yield more stable scores than OLS. <b>Tune α</b> via CV.
{{% /notice %}}

---

## 7. Choosing α automatically

In practice, use <code>RidgeCV</code>/<code>LassoCV</code> (or <code>GridSearchCV</code>).

```python
from sklearn.linear_model import RidgeCV, LassoCV

ridge_cv = make_pipeline(
    StandardScaler(with_mean=False),
    RidgeCV(alphas=np.logspace(-3, 3, 13), cv=5)
).fit(X, y)

lasso_cv = make_pipeline(
    StandardScaler(with_mean=False),
    LassoCV(alphas=np.logspace(-3, 1, 9), cv=5, max_iter=50_000)
).fit(X, y)

print("Best α (RidgeCV):", ridge_cv.steps[1][1].alpha_)
print("Best α (LassoCV):", lasso_cv.steps[1][1].alpha_)
```

---

## 8. Which to use?

- Many features / strong collinearity → try <b>Ridge</b> first (stabilizes coefficients).
- Want feature selection / interpretability → <b>Lasso</b>.
- Strong collinearity and Lasso unstable → <b>Elastic Net</b> (L1+L2).

---

## 9. Summary

- Regularization adds <b>penalties on coefficient size</b> to reduce overfitting.
- <b>Ridge (L2)</b> shrinks coefficients smoothly; rarely exactly zero.
- <b>Lasso (L1)</b> drives some coefficients to zero; performs feature selection.
- <b>Standardization matters</b>; <b>tune α via CV</b>.

---

