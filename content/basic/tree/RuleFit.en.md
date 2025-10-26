---
title: "RuleFit"
pre: "2.3.4 "
weight: 4
title_suffix: "Rules + Linear with L1"
---

{{% notice ref %}}
Friedman, Jerome H., and Bogdan E. Popescu. “Predictive learning via rule ensembles.” The Annals of Applied Statistics (2008). ([pdf](https://jerryfriedman.su.domains/ftp/RuleFit.pdf))
{{% /notice %}}

<div class="pagetop-box">
  <p><b>RuleFit</b> combines <b>rules</b> extracted from tree ensembles (e.g., gradient boosting, random forests) with original features in a <b>linear model</b>, and fits with <b>L1 regularization</b> for sparsity and interpretability.</p>
</div>

---

## 1. Idea (with formulas)

1) <b>Extract rules from trees</b>. Each path to a leaf defines an if–then rule, converted to a binary feature $r_j(x)\in\{0,1\}).  
2) <b>Add (scaled) linear terms</b> for continuous features \\(z_k(x)$.  
3) <b>L1-regularized linear fit</b>:

$$
\hat y(x) = \beta_0 + \sum_j \beta_j r_j(x) + \sum_k \gamma_k z_k(x)
$$

L1 promotes sparsity so only important rules/terms remain.

---

## 2. Dataset (OpenML: house_sales)

{{% notice info %}}
OpenML dataset “house_sales” (King County housing data). Numeric columns only for brevity.
{{% /notice %}}

{{% notice document %}}
- [fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
{{% /notice %}}

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset = fetch_openml(data_id=42092, as_frame=True)
X = dataset.data.select_dtypes("number").copy()
y = dataset.target.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 3. Fit RuleFit

{{% notice document %}}
- Python implementation: <a href="https://github.com/christophM/rulefit" target="_blank" rel="noopener">christophM/rulefit</a>
{{% /notice %}}

```python
from rulefit import RuleFit
import warnings
warnings.simplefilter("ignore")

rf = RuleFit(max_rules=200, tree_random_state=27)
rf.fit(X_train.values, y_train.values, feature_names=list(X_train.columns))

pred_tr = rf.predict(X_train.values)
pred_te = rf.predict(X_test.values)

def report(y_true, y_pred, name):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:8s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R2={r2:.3f}")

report(y_train, pred_tr, "Train")
report(y_test,  pred_te, "Test")
```

Tuning: `max_rules`, tree depth/number, `learning_rate` (if using GBDT), etc.

---

## 4. Inspect top rules

```python
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values(by="importance", ascending=False)
rules.head(10)
```

- <b>rule</b>: if–then condition (or `type=linear` for linear terms)  
- <b>coef</b>: regression coefficient (same units as target)  
- <b>support</b>: fraction of samples satisfying the rule  
- <b>importance</b>: rule importance (coef × support-like)

Reading tips: `type=linear` shows marginal linear effects; `type=rule` captures interactions and thresholds; prefer rules with sizable coef and reasonable support.

---

## 5. Validate rules by visualization

Example: check a simple linear effect for `sqft_above` and a specific rule with boxplots to compare distributions.

---

## 6. Practical tips

- Scaling and outlier handling (Winsorization) help stability  
- Categorical variables: clean/aggregate rare levels before one-hot  
- Target transforms (e.g., `log(y)`) for skewed targets  
- Control rule count/depth for readability vs. fit; tune with CV  
- Prevent leakage: fit preprocessors on train only  
- Produce a short report with top-N rules translated to business language

---

## 7. Summary

- RuleFit = tree-derived <b>rules</b> + <b>linear terms</b> + <b>L1</b>.  
- Balances nonlinearity/interactions with interpretability.  
- Tune rule granularity and validate with visual checks.

---

