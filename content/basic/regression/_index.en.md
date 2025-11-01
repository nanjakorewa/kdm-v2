---
title: "Regression Models | From Linear Baselines to Regularization"
linkTitle: "Regression"
seo_title: "Regression Models | From Linear Baselines to Regularization"
weight: 1
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.1 </b>"
---

{{% summary %}}
- Linear regression captures a linear relationship between inputs and outputs and forms the backbone for both prediction and interpretation.
- By combining extensions such as regularization, robust losses, or dimensionality reduction, it adapts to a wide variety of data sets.
- Each page in this chapter follows the same flow—overview, intuition, formulas, Python experiments, and references—to deepen understanding from fundamentals to applications.
{{% /summary %}}

# Linear Regression

## Intuition
Linear regression answers the simple question “If the input increases by 1, how much does the output change?” Because the coefficients are easy to interpret and the computations are fast, it is often the first model applied in machine learning projects.

## Mathematical formulation
Ordinary least squares estimates the coefficients by minimizing the sum of squared errors between observations and predictions. For multiple features with matrix \\(\mathbf{X}\\) and target vector \\(\mathbf{y}\\),

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
$$

The remaining pages in this chapter extend this framework with regularization, robustification, and other techniques.

## Experiments with Python
Throughout the chapter we provide executable `scikit-learn` examples that cover topics such as:

- Foundations: ordinary least squares, Ridge, Lasso, robust regression  
- Enhanced expressiveness: polynomial regression, Elastic Net, quantile regression, Bayesian linear regression  
- Dimensionality reduction and sparsity: principal component regression, PLS, weighted least squares, Orthogonal Matching Pursuit, SVR, and more

Run the code yourself to see how the models behave on synthetic and real data.

## References
{{% references %}}
<li>Draper, N. R., &amp; Smith, H. (1998). <i>Applied Regression Analysis</i> (3rd ed.). John Wiley &amp; Sons.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
