---
title: "ホルト・ウィンター法（Holt-Winters）"
pre: "2.8.6 "
weight: 6
title_suffix: "トレンドと季節性を同時にモデリング"
---

{{< lead >}}
ホルト・ウィンター法は、指数平滑法を拡張し、トレンドと季節性を同時に扱える時系列モデルです。売上やアクセス数のように季節変動があるデータに適しています。
{{< /lead >}}

---

## 1. モデル構造

モデルは「レベル（平均）」「トレンド」「季節性」の 3 つを平滑化します。加法モデルの更新式は以下のとおりです。

$$
\begin{aligned}
\ell_t &= \alpha (y_t - s_{t-m}) + (1 - \alpha)(\ell_{t-1} + b_{t-1}) \\
b_t &= \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1} \\
s_t &= \gamma (y_t - \ell_t) + (1 - \gamma) s_{t-m} \\
\hat{y}_{t+h} &= \ell_t + h b_t + s_{t-m+h}
\end{aligned}
$$

ここで \\(m\\) は季節周期、\\(\alpha, \beta, \gamma\\) は平滑化係数です。乗法モデルでは加法の部分が乗法になります。

---

## 2. Python 実装

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    series,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
    initialization_method="estimated",
)
fitted = model.fit(optimized=True)
forecast = fitted.forecast(12)

print(fitted.params)
```

`trend` と `seasonal` は `"add"`（加法）か `"mul"`（乗法）を指定できます。トレンドや季節性が成長率に比例するときは乗法が向いています。

---

## 3. ハイパーパラメータ

- `seasonal_periods`：季節周期（例：月次データなら 12、週次なら 52）。
- `trend` / `seasonal`：加法 or 乗法。データの構造に合わせて選択。
- `damped_trend`：減衰トレンドを使うと、遠い将来の予測が過大になりにくい。
- `alpha, beta, gamma`：`optimized=True` で自動推定が一般的。

---

## 4. 実務での活用

- **小売売上予測**：月次・週次の季節性を捉えて短〜中期予測。
- **アクセス解析**：曜日や時間帯による周期を含むトラフィックデータ。
- **在庫管理**：季節商品の需要見通し。

---

## 5. 注意点

- 長期トレンドが非線形な場合は精度が落ちる。必要に応じて Box-Cox やローカルレベルモデルを検討。
- 欠損値が多い場合は補完してから適用する。
- 季節周期を誤ると性能が低下するため、事前に自己相関や季節分解で確認する。

---

## まとめ

- ホルト・ウィンター法は指数平滑法を拡張し、トレンドと季節性を同時に扱える実用的なモデル。
- `ExponentialSmoothing` で簡単に学習・予測でき、加法/乗法や減衰トレンドなど柔軟に設定可能。
- トレンド・季節性を確認しながら、ARIMA や Prophet と組み合わせて比較すると精度向上に役立つ。

---
