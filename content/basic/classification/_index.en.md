---
title: Linear Classification
weight: 3
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.2 </b>"
---

{{% summary %}}
- Linear classifiers separate classes with hyperplanes, offering interpretability and fast training.
- Perceptron, logistic regression, SVM, and related methods share many ideas, which makes comparing them insightful.
- Concepts learned here connect to regularisation, kernels, and distance-based learning.
{{% /summary %}}

# Linear Classification

## Intuition
Linear classifiers place a hyperplane in feature space and assign labels depending on which side of the plane a sample falls. From the simple perceptron to probabilistic logistic regression and margin-based SVM, the key difference lies in how the hyperplane is chosen.

## Mathematical formulation
A generic linear classifier predicts \(\hat{y} = \operatorname{sign}(\mathbf{w}^\top \mathbf{x} + b)\). Training methods differ in how they determine \(\mathbf{w}\); probabilistic variants pass the linear score through a sigmoid or softmax. Kernel tricks replace the dot product with a kernel function to obtain non-linear boundaries while keeping the same recipe.

## Experiments with Python
This chapter walks through hands-on examples in Python and scikit-learn:

- Perceptron: update rules and decision boundaries for the simplest linear classifier.
- Logistic regression: probabilistic binary classification with cross-entropy.
- Softmax regression: multinomial extension that outputs all class probabilities.
- Linear discriminant analysis (LDA): directions that maximise class separability.
- Support vector machines (SVM): margin maximisation and kernel tricks.
- Naive Bayes: fast classification under the conditional independence assumption.
- k-Nearest neighbours: lazy learning based on distances.

You can copy the code snippets and experiment to observe how decision boundaries and probabilities behave.

## References
{{% references %}}
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
<li>Bishop, C. M. (2006). <i>Pattern Recognition and Machine Learning</i>. Springer.</li>
{{% /references %}}
