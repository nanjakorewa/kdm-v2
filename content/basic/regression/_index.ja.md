---
title: 線形回帰
weight: 1
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.1 </b>"
---

{{% summary %}}
- 線形回帰は入力と出力の線形関係を捉える基本モデルで、予測と解釈の両面で土台となる。
- 正則化・ロバスト化・次元圧縮などの発展手法と組み合わせることで、多様なデータに適応できる。
- 本章では基礎から応用まで、概要→直感→数式→Python 実装→参考文献の流れで理解を深める。
{{% /summary %}}

# 線形回帰

## 直感
線形回帰は「入力が 1 増えたら出力はどれだけ変わるか」といった素朴な問いに答える最もシンプルな回帰モデルです。係数による解釈のしやすさと計算の速さから、あらゆる機械学習タスクの入口として扱われます。

## 具体的な数式
最小二乗法に基づく線形回帰は、観測値と予測値の差の二乗和を最小化することで係数を求めます。重回帰では行列 \\(\mathbf{X}\\) とベクトル \\(\mathbf{y}\\) を用いて

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

の形で解析解を得られます。以降の各ページでは、この枠組みを正則化やロバスト化などさまざまな方向に拡張します。

## Pythonを用いた実験や説明
章内の各ページでは `scikit-learn` を中心にした Python 実装例を掲載し、以下のようなテーマを扱います。

- 基礎：最小二乗法、リッジ回帰、ラッソ回帰、ロバスト回帰
- 表現力の拡張：多項式回帰、Elastic Net、分位点回帰、ベイズ線形回帰
- 次元圧縮と疎性：主成分回帰、PLS 回帰、加重最小二乗法、Orthogonal Matching Pursuit、SVR など

コードはそのまま実行できるよう整備しているので、手元で動かしながら挙動を確認してください。

## 参考文献
{{% references %}}
<li>Draper, N. R., &amp; Smith, H. (1998). <i>Applied Regression Analysis</i> (3rd ed.). John Wiley &amp; Sons.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
