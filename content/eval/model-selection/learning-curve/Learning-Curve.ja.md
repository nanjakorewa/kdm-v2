---

title: "ラーニングカーブ"

pre: "4.1.5 "

weight: 5

title_suffix: "データ量と汎化性能の関係を可視化"

---



{{< lead >}}

ラーニングカーブは、学習データ量を増やしたときの訓練スコアと検証スコアの推移を描くチャートです。過学習・学習不足の状態を診断し、追加データの効果を見積もるのに役立ちます。

{{< /lead >}}



---



## 1. Python で描画



```python

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve

from sklearn.ensemble import RandomForestRegressor



train_sizes, train_scores, valid_scores = learning_curve(

    RandomForestRegressor(random_state=0),

    X, y,

    train_sizes=np.linspace(0.1, 1.0, 8),

    cv=5,

    scoring="neg_mean_absolute_error",

    n_jobs=-1,

)



train_mean = -train_scores.mean(axis=1)

valid_mean = -valid_scores.mean(axis=1)



plt.plot(train_sizes, train_mean, label="Train")

plt.plot(train_sizes, valid_mean, label="Validation")

plt.xlabel("学習サンプル数")

plt.ylabel("MAE")

plt.legend()

plt.show()

```



`learning_curve` は、指定したスコアリングで訓練スコアと検証スコア（交差検証）を計算してくれます。



---



## 2. 読み取り方



- **過学習**：訓練スコアが非常に良いが、検証スコアが離れている。モデルが複雑すぎるか、正則化が不足。

- **学習不足**：訓練スコアも検証スコアも悪く、追加データや特徴量が必要。

- **収束**：訓練・検証スコアが近づき、追加データの効果が小さい。別のモデルを試すタイミング。



---



## 3. ハイパーパラメータ選定に活用



- ラーニングカーブを確認してから、モデルの容量（深さ、正則化強度など）を調整する。

- 学習サンプル数を追加する前に、カーブが飽和していないか確認すると投資判断に役立つ。

- 交差検証の分割数 `cv` を増やすと滑らかになるが、計算コストが増えるのでバランスを取る。



---



## 4. 実務のヒント



- **小規模データ**：サンプルが少ない領域では、追加データを収集する価値をラーニングカーブで判断。

- **特徴量エンジニアリング**：カーブが高い位置で平行に走っている場合、特徴量の改善が必要であるサイン。

- **モデル比較**：複数モデルのラーニングカーブを重ねると、どのモデルがデータ量に敏感か比較できる。



---



## まとめ



- ラーニングカーブは訓練・検証スコアをデータ量に対してプロットし、過学習・学習不足を可視化する。

- `learning_curve` 関数で簡単に描画でき、追加データやモデル選択の意思決定をサポートする。

- 他の診断（検証曲線、バリデーションスコア）と併用し、効率的に改善サイクルを回そう。



---

