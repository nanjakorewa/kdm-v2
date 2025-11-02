---

title: "自由度調整済み決定係数（Adjusted R²）"

pre: "4.2.6 "

weight: 6

title_suffix: "説明変数の増加にペナルティを与える"

---



{{< lead >}}

自由度調整済み決定係数（Adjusted R²）は、決定係数 R² に説明変数の数を加味した指標です。変数を増やすと R² が必ず上がる問題を補正し、真にモデルが改善したかを判断できます。

{{< /lead >}}



---



## 1. 定義



$$

\mathrm{Adjusted\ } R^2 = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}

$$



ここで \\(n\\) はサンプル数、\\(p\\) は説明変数（特徴量）の数です。\\(p\\) が増えると分母が小さくなり、改善しない特徴量を追加すると値が下がります。



---



## 2. Python で計算



scikit-learn の `LinearRegression` などは決定係数を返すので、Adjusted R² は自分で計算します。



```python

import numpy as np

from sklearn.datasets import make_regression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X, y = make_regression(

    n_samples=1000,

    n_features=10,

    n_informative=6,

    noise=5.0,

    random_state=0,

)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=0

)



model = LinearRegression()

model.fit(X_train, y_train)



r2 = model.score(X_test, y_test)

n = X_test.shape[0]

p = X_test.shape[1]

adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)



print(f"R² = {r2:.3f}, Adjusted R² = {adj_r2:.3f}")

```



---



## 3. 直感と使いどころ



- **特徴量が増えると厳しくなる**：不要な特徴量を追加すると Adjusted R² は低下し、モデル複雑度に対するペナルティが働く。

- **サンプルが少ない場合に注意**：`n` が小さいと分母 `n - p - 1` が小さくなり値が不安定。サンプル数に余裕があるときに評価すると信頼性が高い。

- **モデル比較**：同じデータセット上で、より少ない特徴量で高い Adjusted R² を実現できるモデルが望ましい。



---



## 4. 他指標との使い分け



| 指標 | 特徴 | 注意点 |

| --- | --- | --- |

| R² | 直感的で広く知られる | 特徴量を増やすと必ず上昇 |

| **Adjusted R²** | 特徴量数を考慮し、公平に比較できる | サンプル数が少ないと不安定 |

| AIC/BIC | 損失関数 + ペナルティ | モデルが正規分布仮定を大きく外れると解釈が難しい |



---



## 5. 実務での活用



- **特徴量選択**：ステップワイズ法などで特徴量を増減させる際、Adjusted R² の向上を目安にする。

- **説明責任**：R² だけでなく Adjusted R² も提示すると、過学習を抑えていることを示しやすい。

- **比較基準**：正規化やスケーリング違いのモデルを比較する際、Adjusted R² が向上していれば実質的な改善があったと判断しやすい。



---



## まとめ



- Adjusted R² は R² に特徴量数のペナルティを掛けた指標で、モデルが本当に改善したかを確認できる。

- サンプル数と特徴量のバランスに注意しながら、R² と併せてモデル評価に活用しよう。

- AIC/BIC など他の情報量規準とも組み合わせると、より堅牢なモデル選択が可能になる。



---

