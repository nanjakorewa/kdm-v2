---
title: 線形回帰
weight: 1
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.1 </b>"
---

{{% summary %}}
- 線形回帰は入力と出力の線形関係を捉える基本モデルで、予測と解釈の両面で土台になる。
- 正則化・ロバスト化・次元圧縮などの拡張と組み合わせることで、多様なデータにも適応できる。
- 各ページは「まとめ→直感→数式→Python実験→参考文献」の流れで学びを深められる構成になっている。
{{% /summary %}}

# 線形回帰

## 直感
線形回帰は「入力が 1 増えると出力はどれだけ変化するか」という素朴な問いに答える最もシンプルな回帰モデルです。係数の解釈が容易で計算も高速なため、機械学習プロジェクトで最初に試されることが多く、他の手法を理解する足掛かりにもなります。

## 具体的な数式
最小二乗法に基づく線形回帰は、観測値と予測値の差の二乗和を最小化することで係数を推定します。重回帰では行列 \\(\mathbf{X}\\) とベクトル \\(\mathbf{y}\\) を用いて

$$
\hat{\boldsymbol\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

という解析解を得られます。本章ではこの枠組みを基に、正則化やロバスト化、次元圧縮などさまざまな観点から発展させていきます。

## Pythonを用いた実験や説明
すべてのページで `scikit-learn` を用いた実行可能な Python コードを提供し、次のようなテーマを取り上げます。

- 基礎: 最小二乗法、リッジ回帰、ラッソ回帰、ロバスト回帰  
- 表現力の拡張: 多項式回帰、Elastic Net、分位点回帰、ベイズ線形回帰  
- 次元圧縮と疎性: 主成分回帰、PLS 回帰、加重最小二乗法、Orthogonal Matching Pursuit、SVR など

コードはそのまま実行できるよう整備してあるので、実際に動かしながらモデルの挙動を確認してみてください。

## 参考文献
{{% references %}}
<li>Draper, N. R., &amp; Smith, H. (1998). <i>Applied Regression Analysis</i> (3rd ed.). John Wiley &amp; Sons.</li>
<li>Hastie, T., Tibshirani, R., &amp; Friedman, J. (2009). <i>The Elements of Statistical Learning</i>. Springer.</li>
<li>Seber, G. A. F., &amp; Lee, A. J. (2012). <i>Linear Regression Analysis</i> (2nd ed.). Wiley.</li>
{{% /references %}}
