---
title: "ブライアスコア（Brier Score）"
pre: "4.3.10 "
weight: 10
title_suffix: "確率予測のキャリブレーションを見る"
---

{{< lead >}}
ブライアスコア（Brier Score）は、予測確率と実際のラベルとの差の二乗平均です。確率が実際の事象頻度にどれだけ近いかを評価でき、気象予報や信用リスクなど確率そのものを提示する場面で重要な指標です。
{{< /lead >}}

---

## 1. 定義

二値分類では次のように定義されます。

$$
\mathrm{Brier} = \frac{1}{n} \sum_{i=1}^n (p_i - y_i)^2
$$

ここで \\(p_i\\) は陽性の予測確率、\\(y_i\\) は実際のラベル（0 または 1）です。多クラスの場合は各クラスの確率との差平方を合計します。

---

## 2. Python で計算

```python
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

prob = model.predict_proba(X_test)[:, 1]
brier = brier_score_loss(y_test, prob)
print("Brier Score:", round(brier, 4))

disp = CalibrationDisplay.from_predictions(y_test, prob, n_bins=10)
plt.show()
```

`brier_score_loss` は負の対数尤度よりも解釈しやすい値を返します。Calibration plot と組み合わせると、確率校正の状態が視覚的に把握できます。

---

## 3. スコアの解釈

- 0 に近いほど良い（完全一致で 0）。
- ランダムな 0.5 の確率を常に出すと 0.25 に近づく。
- 予測確率が 0 や 1 に極端で間違うとスコアが大きく悪化するため、自信過剰なモデルを検知できる。

---

## 4. 分解解析

ブライアスコアは信頼性、解像度、不確実性の 3 要素に分解できます（Murphy 分解）。

$$
\mathrm{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}
$$

- **Reliability**：予測確率と実現頻度の差。小さいほど良い。
- **Resolution**：予測のばらつき。大きいほど良い。
- **Uncertainty**：対象プロセスの固有の不確実性。

気象分野ではこの分解を用いて予報モデルの改善点を特定します。

---

## 5. 実務での活用

- **確率を提示するサービス**：ローン審査のデフォルト確率など、ユーザーに確率をそのまま示す場合に適している。
- **校正の比較**：Platt scaling や isotonic regression などの後処理を行った後、ブライアスコアが改善しているか確認。
- **スコア解析**：Calibration plot や Reliability diagram と併用すると、どの確率帯で過小・過大評価が起きているか分かりやすい。

---

## まとめ

- ブライアスコアは確率予測の精度とキャリブレーションを評価する二乗誤差指標。
- `brier_score_loss` で簡単に算出でき、校正プロットと併用すると改善の方向性が見える。
- 対数損失（Log Loss）と合わせて使うことで、確率の質と順位性能を両面からチェックできる。

---
