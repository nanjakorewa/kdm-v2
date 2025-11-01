---
title: "決定木 | 機械学習の基礎解説"
linkTitle: "決定木"
seo_title: "決定木 | 機械学習の基礎解説"
weight: 4
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.3 </b>"
---

{{% summary %}}
- 決定木は特徴量に基づく条件分岐で予測を行い、木構造として可視化できるため解釈性が高い。
- 分類・回帰のいずれにも利用でき、アンサンブル手法（ランダムフォレストや勾配ブースティング）の基礎となる。
- 深さや葉ノード数、剪定などのハイパーパラメータを調整することで、当てはまりと汎化性能のバランスを取れる。
{{% /summary %}}

# 決定木

## 直感
「もし◯◯なら左へ、そうでなければ右へ」というルールを繰り返し、最終的に葉ノードで予測値やクラスを出力します。木構造をそのまま図示できるため、予測根拠を関係者に説明しやすいのが特徴です。

## 具体的な数式
分割は不純度指標（ジニ不純度、エントロピー、平均二乗誤差など）を用い、親ノードと子ノードの不純度の差（情報利得）が最大になる特徴量としきい値を選びます。コスト複雑度剪定では訓練誤差と木のサイズを同時に最小化し、シンプルな木に整えます。

## Pythonを用いた実験や説明
章内の各ページでは次のトピックを扱います。

- 決定木（分類）：条件分岐と不純度の仕組み、決定境界と木構造の可視化
- 決定木（回帰）：平均二乗誤差による分割と区分定数関数としての振る舞い
- 決定木のパラメータ：`max_depth`、`min_samples_leaf`、`ccp_alpha` などの役割と調整方法
- RuleFit：木から抽出したルールと線形モデルの組み合わせによる解釈可能な学習

各ページの Python コードは実行可能な形で掲載しているので、手元で試しながら挙動を確認してください。

## 参考文献
{{% references %}}
<li>Breiman, L., Friedman, J. H., Olshen, R. A., &amp; Stone, C. J. (1984). <i>Classification and Regression Trees</i>. Wadsworth.</li>
<li>scikit-learn developers. (2024). <i>Decision Trees</i>. https://scikit-learn.org/stable/modules/tree.html</li>
{{% /references %}}
