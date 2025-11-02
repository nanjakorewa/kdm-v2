---

title: "PICP（Prediction Interval Coverage Probability）"

pre: "4.2.11 "

weight: 11

title_suffix: "予測区間が目標確率を満たしているか"

---



{{< lead >}}

PICP（Prediction Interval Coverage Probability）は、予測区間が実測値をどれだけ含むかを測る指標です。確率的予測や予測区間を提供するモデルの信頼性チェックに利用します。

{{< /lead >}}



---



## 1. 定義



予測下限を \\(L_i\\)、上限を \\(U_i\\)、実測値を \\(y_i\\)、ターゲット信頼水準を \\(\gamma\\) とすると、



$$

\mathrm{PICP} = \frac{1}{n} \sum_{i=1}^n \mathbf{1}\{ L_i \le y_i \le U_i \}

$$



ターゲット値 \\(\gamma\\)（例：0.9）に対して PICP が同程度か、どれくらいズレているかを確認します。



---



## 2. Python で計算



```python

import numpy as np



def picp(y_true, lower, upper):

    inside = (y_true >= lower) & (y_true <= upper)

    return inside.mean()



coverage = picp(y_test, lower_bound, upper_bound)

print("PICP:", round(coverage, 3))

```



`lower_bound` と `upper_bound` はモデルが出力した予測区間です。LightGBM の quantile モードや NGBoost などの分布予測モデルで取得できます。



---



## 3. 理想的な状態



- PICP ≒ 目標信頼水準（例：0.9）のとき、区間が適切にキャリブレーションされている。

- PICP が低すぎる → 区間が狭すぎて過小評価。

- PICP が高すぎる → 区間が広すぎて保守的。



区間幅も併せて確認しないと、過度に広い区間で PICP を満たしてしまうことがあるため注意します。



---



## 4. PINAW（Normalized Average Width）との併用



区間の幅を評価する指標として PINAW（Prediction Interval Normalized Average Width）があります。



$$

\mathrm{PINAW} = \frac{1}{nR} \sum_{i=1}^n (U_i - L_i)

$$



ここで \\(R\\) はデータの範囲です。PICP と PINAW を併用し、十分な被覆率と適切な幅を両立できているかを確認します。



---



## 5. 実務での活用



- **在庫・需給管理**：欠品を避けるために 90% 区間で PICP を監視。

- **エネルギー予測**：需給調整のリスク管理として区間予測の信頼性を評価。

- **金融リスク**：Value at Risk (VaR) のバックテストに近い概念で使用。



---



## まとめ



- PICP は予測区間が目標信頼水準を満たしているかをチェックする指標。

- 過小/過大評価を見極めるために PINAW やピンボール損失と併用すると効果的。

- 区間予測を提供するモデルでは、PICP を定期的にモニタリングして信頼性を保とう。



---

