---
title: "説明分散（Explained Variance）"
pre: "4.2.8 "
weight: 8
title_suffix: "予測がどこまで変動を説明できたか"
---

{{< lead >}}
説明分散スコア（Explained Variance Score）は、モデルが目的変数の分散をどれだけ説明できたかを測る指標です。決定係数 R² に似ていますが、バイアスに敏感で誤差の平均値を重視する点が異なります。
{{< /lead >}}

---

## 1. 定義

観測値の分散を \\(\mathrm{Var}(y)\\)、予測誤差の分散を \\(\mathrm{Var}(y - \hat{y})\\) とすると、

$$
\mathrm{Explained\ Variance} = 1 - \frac{\mathrm{Var}(y - \hat{y})}{\mathrm{Var}(y)}
$$

値は 1 に近いほど良く、0 なら平均的な予測と同等、負の値は平均予測よりも悪いことを意味します。

---

## 2. Python で計算

```python
from sklearn.metrics import explained_variance_score

ev = explained_variance_score(y_test, y_pred)
print("Explained Variance:", round(ev, 3))
```

Scikit-learn の `explained_variance_score` は複数出力（multi-output）にも対応しており、`multioutput="raw_values"` で各出力ごとのスコアを得られます。

---

## 3. R² との違い

- **バイアスへの感度**：R² は誤差の平均値にも敏感だが、説明分散は平均誤差（バイアス）には影響されず、分散だけを評価する。
- **用途**：予測の平均的なズレよりも、ばらつきをどれだけ説明できたかを見たいときに適する。
- **値の範囲**：実務では R² と一緒に報告し、モデルの「分散説明力」と「全体適合度」を両面から確認する。

---

## 4. 実務での活用

- **リスク予測**：目標値の振れ幅を捉えられているかを確認したいとき。
- **複数出力回帰**：出力ごとの説明分散を並べて、どの出力が難しいかを特定する。
- **バイアス補正**：予測が全体的にずれている場合でも説明分散は高くなるため、MAE や MBE と併用してバイアスの確認を忘れない。

---

## 5. 他指標との併用

| 指標 | 焦点 | 説明 |
| --- | --- | --- |
| R² | 全体適合度 | バイアスと分散の両方に敏感 |
| **Explained Variance** | 分散説明力 | 誤差の平均値には鈍感 |
| MAE / RMSE | 絶対誤差 | 実際のズレを直接把握 |

---

## まとめ

- 説明分散スコアは目的変数の変動をどれだけ捉えたかを示す指標で、R² と補い合う。
- `explained_variance_score` で簡単に計算でき、マルチ出力にも対応。
- MAE や MBE とも併用し、分散・平均の両側面からモデルを評価しよう。

---
