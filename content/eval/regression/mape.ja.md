---

title: "MAPE と sMAPE"

pre: "4.2.4 "

weight: 4

title_suffix: "相対誤差で予測精度を捉える"

---



{{< lead >}}

MAPE（Mean Absolute Percentage Error）は予測誤差をパーセンテージで表す指標です。需要予測や売上予測のように「何％外したか」を共有したいときに便利ですが、ゼロ付近の値で不安定になる点には注意が必要です。

{{< /lead >}}



---



## 1. 定義



$$

\mathrm{MAPE} = \frac{100}{n} \sum_{i=1}^n \left| \frac{y_i - \hat{y}_i}{y_i} \right|

$$



- 実測値に対する相対誤差の平均。

- 値が小さいほど予測が実測に近い。

- 実測値 `y_i` が 0 に近いと発散しやすい。



---



## 2. Python での実装



```python

import numpy as np

from sklearn.metrics import mean_absolute_percentage_error



y_true = np.array([120, 150, 80, 200])

y_pred = np.array([110, 160, 75, 210])



mape = mean_absolute_percentage_error(y_true, y_pred)

print(f"MAPE = {mape * 100:.2f}%")

```



scikit-learn の `mean_absolute_percentage_error` は既定で 0–1 の値を返すため、パーセント表示には 100 を掛けます。



---



## 3. sMAPE（対称 MAPE）



MAPE は実測値が 0 に近いと値が暴走しやすいため、分母に予測値も加えた sMAPE がよく使われます。



$$

\mathrm{sMAPE} = \frac{100}{n} \sum_{i=1}^n \frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}

$$



分母に平均を取ることで、実測値が小さいときも安定しやすくなります。



```python

def smape(y_true, y_pred):

    numerator = np.abs(y_true - y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # 0 で割らないように微小値を足す

    return np.mean(numerator / np.maximum(denominator, 1e-8))



print(f"sMAPE = {smape(y_true, y_pred) * 100:.2f}%")

```



---



## 4. 使用時の注意点



- **ゼロや負の値**：MAPE はゼロに敏感で、負の値では意味が変わる。ゼロ除けや sMAPE の利用を検討。

- **平均の偏り**：大きな実測値よりも小さな実測値の誤差が重くなる。売上 1000 の誤差 10 より、売上 10 の誤差 5 の方が大きく影響する。

- **外れ値**：割合指標なので絶対値より外れ値の影響が小さいが、ゼロ除けを怠ると破綻する。

- **報酬の解釈**：ビジネスでは「平均で何％外したか」を伝えやすい一方、金額換算ができない点に注意。



---



## 5. 他指標との併用



- **MAE / RMSE**：絶対値の誤差も併用して、実際のダメージ量を把握。

- **RMSLE**：過小予測を重視したいときはログ誤差を取る。

- **分位点損失**：需要予測でリスクバッファを設けるなら、上下限の目標値も合わせて評価する。



---



## まとめ



- MAPE は予測誤差を割合で表すため、ビジネス側に説明しやすいが 0 付近では不安定。

- sMAPE を使うと極端な値でも安定しやすく、需要予測コンテストでも標準的に採用されている。

- 絶対値指標と組み合わせて、ビジネス影響を多角的に評価するのが実務では重要。



---

