---

title: "MASE（Mean Absolute Scaled Error）"

pre: "4.2.9 "

weight: 9

title_suffix: "季節性を考慮したスケーリング誤差"

---



{{< lead >}}

MASE（Mean Absolute Scaled Error）は、回帰・時系列モデルの誤差を単純な基準モデル（ナイーブ予測）と比較する指標です。季節性があるデータでも扱いやすく、モデルがベンチマークよりどれだけ良いかを数値化できます。

{{< /lead >}}



---



## 1. 定義



$$

\mathrm{MASE} = \frac{\frac{1}{n} \sum_{i=1}^{n} | y_i - \hat{y}_i |}{\frac{1}{n-m} \sum_{t=m+1}^{n} | y_t - y_{t-m} |}

$$



ここで \\(m\\) は季節周期（非季節の場合は 1）です。分母は季節ナイーブ予測の平均絶対誤差を表し、MASE < 1 なら基準モデルより良いことを意味します。



---



## 2. Python での計算例



scikit-learn には直接の実装がないため、カスタム関数を作成します。



```python

import numpy as np



def mase(y_true, y_pred, m=1):

    y_true = np.asarray(y_true)

    y_pred = np.asarray(y_pred)

    n = y_true.size

    scale = np.mean(np.abs(y_true[m:] - y_true[:-m]))

    if scale == 0:

        return np.nan

    return np.mean(np.abs(y_true - y_pred)) / scale

```



---



## 3. 解釈のポイント



- **MASE < 1**：季節ナイーブ予測より優れている。

- **MASE = 1**：ベンチマークと同等。

- **MASE > 1**：ベンチマークより悪い。



季節性を考慮したスケーリングのため、MAPE や RMSLE と違い、0 や負の値が含まれていても問題ありません。



---



## 4. 実務での活用



- **需要予測**：季節性が強い売上データのモデル比較に最適。

- **モデル選抜**：複数モデルを比較し、MASE が最も低いモデルを採用する。

- **業務レポート**：ベンチマークとの相対性能を示したいときに説得力が高い。



---



## 5. 注意点



- 分母（季節ナイーブの誤差）が 0 に近い場合は NaN が発生するため、適切な季節周期を選ぶ。

- 短い系列では分母の推定が不安定なので、十分なデータ長が必要。

- MASE が良くても MAE が大きい場合は、絶対誤差も併せて報告する。



---



## まとめ



- MASE は季節ナイーブ予測と比較してモデルの優位性を数字で示せる指標。

- `m` を設定することで季節性のあるデータにも適用でき、MAPE の弱点だった 0 問題も回避できる。

- MAE や RMSE と併用し、絶対誤差と相対的な改善度を両方確認しながらモデルを改善しよう。



---

