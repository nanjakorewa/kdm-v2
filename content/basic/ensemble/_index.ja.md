title: アンサンブル
weight: 5
chapter: true
not_use_colab: true
not_use_twitter: true
pre: "<b>2.4 </b>"
---

# アンサンブル

<div class="pagetop-box">
  <p><b>アンサンブル学習</b>は、複数のモデル（弱学習器）を組み合わせて、単体モデルよりも高い精度・安定性を得る手法の総称です。代表例として <b>Bagging（例: RandomForest）</b>、<b>Boosting（例: AdaBoost, Gradient Boosting）</b>、<b>Stacking</b> があります。</p>
</div>

---

## ここで学ぶこと

1. <b>RandomForest</b>: 決定木を多数作り多数決（または平均）で予測。特徴のサブサンプリングとブートストラップで分散を下げる。
2. <b>AdaBoost</b>: 誤分類（誤差）の大きなサンプルに重みを置き、弱学習器を逐次強化。
3. <b>Gradient Boosting</b>: 損失関数の負の勾配（残差）を逐次近似し、段階的にモデルを改善。
4. <b>Stacking</b>: 複数モデルの出力をメタ学習器に入力して最終予測を行う。

---

## 使い分けの直感

- データが高分散・非線形 → <b>RandomForest</b> が堅牢。
- 弱学習器（浅い木）で精度を積み上げたい → <b>Boosting</b> 系。
- 複数系統のモデルを融合 → <b>Stacking</b>。

---
