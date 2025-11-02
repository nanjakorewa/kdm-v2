---

title: "MAE・RMSE"

pre: "4.2.3 "

weight: 3

title_suffix: "誤差指標の特徴と使い分け"

---



{{< lead >}}

平均絶対誤差 (MAE) と二乗平均平方根誤差 (RMSE) は、回帰モデルの予測誤差を評価する基本指標です。外れ値への感度や単位の解釈が異なるため、用途に応じて使い分けましょう。

{{< /lead >}}



---



## 1. 定義と性質



観測値を \\(y_i\\)、予測値を \\(\hat{y}_i\\) とすると、



$$

\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|, \qquad

\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}

$$



- **MAE**：誤差の絶対値を平均。外れ値の影響が線形で、中央値に相当するロバスト性がある。

- **RMSE**：誤差を二乗して平均し平方根。大きな誤差を強く罰するため、外れ値が多いと増加しやすい。

- 単位はいずれも元データと同じ。RMSE は二乗の平方根で戻すが、感覚的には「大きな外れを嫌う」設定。



---



## 2. Python で計算する



```python

import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error



y_true = np.array([10, 12, 9, 11, 10])

y_pred = np.array([9.5, 13, 8, 11.5, 9])



mae = mean_absolute_error(y_true, y_pred)

rmse = mean_squared_error(y_true, y_pred, squared=False)

print(f"MAE = {mae:.3f}")

print(f"RMSE = {rmse:.3f}")

```



`mean_squared_error(..., squared=False)` とすれば平方根付きで RMSE を得られます。平方根を掛けない場合は MSE です。



---



## 3. 外れ値がある場合の比較



```python

import numpy as np



baseline = np.linspace(100, 110, 20)

prediction = baseline + np.random.normal(0, 1.0, size=baseline.size)



mae_clean = np.mean(np.abs(baseline - prediction))

rmse_clean = np.sqrt(np.mean((baseline - prediction) ** 2))



prediction_with_outlier = prediction.copy()

prediction_with_outlier[0] += 15  # 大きな外れ



mae_outlier = np.mean(np.abs(baseline - prediction_with_outlier))

rmse_outlier = np.sqrt(np.mean((baseline - prediction_with_outlier) ** 2))



print(f"MAE (clean, outlier) = {mae_clean:.2f}, {mae_outlier:.2f}")

print(f"RMSE (clean, outlier) = {rmse_clean:.2f}, {rmse_outlier:.2f}")

```



- MAE は外れ値が加わっても緩やかに増加する。

- RMSE は外れ値に大きく引っ張られ、誤差の跳ね上がりを敏感に捉える。



---



## 4. 指標の選び方



- **外れ値が少なく、平均的な誤差を重視** ➜ RMSE を採用すると微妙なずれも評価しやすい。

- **外れ値が多い、頑健な指標が欲しい** ➜ MAE や中央値絶対偏差 (MAD) を併用。

- **報酬・コストが二乗に比例** ➜ RMSE がしっくり来る (例: エネルギー損失、物理誤差)。

- **単位をそのまま伝えたい** ➜ MAE は直感的に解釈しやすい。「平均で ○○ 単位ずれる」。



---



## 5. 追加の指標と組み合わせ



- **MAPE**：百分率誤差。相対的な誤差感を共有したいときに便利。ただしゼロ付近では不安定。

- **RMSLE**：対数スケールで RMSE を取る。需要予測などで過小予測を強く罰したいケース。

- **分位点損失 (Pinball Loss)**：売上予測の上下限など、リスク区間を評価したい場面で有効。



---



## 6. まとめ



- MAE と RMSE はいずれも回帰誤差を正確に可視化する基礎指標だが、外れ値への感度が異なる。

- 実務では両方を並べて確認し、ビジネス上のコスト構造に合わせて重視する指標を決める。

- MAPE や RMSLE など関連指標も組み合わせ、モデル改善の方向性を多角的に把握しよう。



---

