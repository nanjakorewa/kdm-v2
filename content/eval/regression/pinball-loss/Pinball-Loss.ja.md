---
title: "ピンボール損失（Pinball Loss）"
pre: "4.2.10 "
weight: 10
title_suffix: "分位点回帰の評価指標"
---

{{< lead >}}
ピンボール損失（Pinball Loss）は、指定した分位点（quantile）に対する予測がどれだけ偏っているかを測る指標です。予測区間や上下限の推定を行うモデルの評価に欠かせません。
{{< /lead >}}

---

## 1. 定義

分位点 \\(\tau\\) におけるピンボール損失は、

$$
L_\tau(y, \hat{y}) =
\begin{cases}
  \tau (y - \hat{y}) & \text{if } y \ge \hat{y} \\
  (1 - \tau)(\hat{y} - y) & \text{otherwise}
\end{cases}
$$

予測値が分位点より低すぎれば \\(\tau\\) 倍、逆に高すぎれば \\(1-\tau\\) 倍のペナルティが課されます。

---

## 2. Python で計算

```python
from sklearn.metrics import mean_pinball_loss

q = 0.9  # 90%分位点
loss = mean_pinball_loss(y_true, y_pred_quantile, alpha=q)
print(f"Pinball Loss (q={q}):", round(loss, 4))
```

`alpha` に分位点を指定し、分位点回帰モデルの出力（例：GradientBoostingRegressor の `quantile` 損失）と組み合わせて評価します。

---

## 3. 解釈

- **小さいほど良い**：分位点が目標とする位置に近づいている。
- **α=0.5** の場合は MAE と同じになる。
- **左右非対称のペナルティ**：上振れと下振れでペナルティが異なるため、リスク回避の度合いを調整できる。

---

## 4. 実務での活用

- **需要予測の予測区間**：安全在庫のために 90% 分位点を評価。
- **リスク管理**：Value at Risk (VaR) など金融リスク指標の評価。
- **エネルギー負荷予測**：上限・下限の予測ラインを別々に学習し、ピンボール損失で性能を確認。

---

## 5. 注意点

- 分位点ごとにモデルを訓練する必要があるため、複数の quantile を出力するモデル（LightGBM の quantile モードなど）が便利。
- 分位点が 0 や 1 に近いほど外れ値に敏感になるため、サンプル数が十分に必要。
- ピンボール損失だけでなく、PICP（Prediction Interval Coverage Probability）などと併用すると、区間の信頼性を総合的に評価できる。

---

## まとめ

- ピンボール損失は分位点回帰の基本指標で、上振れ・下振れの誤差を非対称に評価する。
- `mean_pinball_loss` で容易に計算でき、リスク回避の度合いに応じて分位点を設定可能。
- 予測区間を扱うタスクでは、ピンボール損失と PICP を組み合わせて性能をチェックしよう。

---
