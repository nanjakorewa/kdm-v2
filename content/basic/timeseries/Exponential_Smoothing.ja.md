---
title: "指数平滑法（Simple Exponential Smoothing）"
pre: "2.8.5 "
weight: 5
title_suffix: "トレンドのない系列をなめらかに予測"
---

{{< lead >}}
指数平滑法は、直近の観測値を指数的に重み付けして平均を取る単純な時系列予測手法です。トレンドや季節性が目立たない系列で効果を発揮し、滑らかな予測を得られます。
{{< /lead >}}

---

## 1. アルゴリズム

予測値 \\(\hat{y}_{t+1}\\) は、観測値 \\(y_t\\) と前回の予測値 \\(\hat{y}_{t}\\) を滑らかに混合します。

$$
\hat{y}_{t+1} = \alpha y_t + (1 - \alpha) \hat{y}_t
$$

ここで \\(\alpha\\) は平滑化係数（0〜1）。大きくすると直近の観測値を強く反映し、小さくすると過去の値を重視します。

---

## 2. Python 実装

```python
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

model = SimpleExpSmoothing(series, initialization_method="estimated")
fitted = model.fit(smoothing_level=None, optimized=True)
forecast = fitted.forecast(12)

print("Alpha:", fitted.model.params["smoothing_level"])
```

`SimpleExpSmoothing` は `statsmodels` の Holt-Winters モジュールに含まれており、`optimized=True` で最適な \\(\alpha\\) を推定できます。

---

## 3. 特徴と適用範囲

- **トレンドなし**：データが定常に近く、顕著なトレンドや季節性がない場合に適する。
- **レスポンス調整**：\\(\alpha\\) を調整することで、最新データへの追従性をコントロールできる。
- **計算コストが低い**：更新式が単純なのでオンライン処理にも向いている。

---

## 4. ハイパーパラメータ

- `smoothing_level (α)`：大きいほど最新観測を重視。ノイズが多い場合は小さめに。
- `initial_level`：初期値。`initialization_method="estimated"` を使うと自動で推定。
- 学習期間：あまり古いデータを含めすぎるとトレンドが入ってしまうため、適切な期間でモデルを更新する。

---

## 5. 実務での利用例

- **在庫補充**：日々の出荷データでトレンドがほぼ一定の SKU。
- **センサー値**：短期的なノイズを平滑化し、滑らかな予測を得たい場合。
- **ダッシュボード**：本格モデルを導入する前のベースラインとして利用。

---

## まとめ

- 指数平滑法は、直近の観測値に重みを置いたシンプルな平滑化・予測手法。
- `SimpleExpSmoothing` で簡単に実装でき、平滑化係数 \\(\alpha\\) の調整で追従性をコントロールできる。
- トレンドや季節性がある場合は Holt-Winters 法などの拡張を検討しよう。

---
