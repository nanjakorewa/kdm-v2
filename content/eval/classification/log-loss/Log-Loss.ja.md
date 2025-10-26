---
title: "対数損失（Log Loss）"
pre: "4.3.4 "
weight: 4
title_suffix: "確率予測の信頼度を測る"
---

{{< lead >}}
対数損失（Log Loss）は予測確率がどれだけ正解ラベルに寄り添っているかを測る指標です。確率を出力できるモデルのキャリブレーションを評価する際に欠かせません。
{{< /lead >}}

---

## 1. 定義

二値分類では次のように表されます。

$$
\mathrm{LogLoss} = -\frac{1}{n} \sum_{i=1}^n \bigl[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \bigr]
$$

ここで \\(p_i\\) はサンプル \\(i\\) が陽性であるとモデルが予測した確率。  
多クラスでは正解クラスの予測確率の対数を足し合わせます。

---

## 2. Python で計算

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)

ll = log_loss(y_test, proba)
print(f"Log Loss = {ll:.4f}")
```

`log_loss` の第 2 引数には確率行列を渡します。しきい値で二値化したラベルを渡すと意味がなくなるので注意してください。

---

## 3. 確率に対するペナルティ

- 正解クラスに高確率を割り当てれば損失は小さく、外れれば大きなペナルティ。
- 0 や 1 に極端に寄った確率で外すと、損失がほぼ無限大に発散します。数値安定化のために \\(10^{-15}\\) などの下限でクリップして計算することが一般的です。
- モデルが自信過剰かどうかを判断するのに有用。Accuracy が高くても Log Loss が悪い場合は、確率が過剰に偏っている可能性があります。

---

## 4. ハイパーパラメータの直感

指標そのものにハイパーパラメータはありませんが、scikit-learn の `log_loss` にはいくつかのオプションがあります。

| 引数 | 役割 | 調整したとき |
| --- | --- | --- |
| `labels` | 期待するクラスの順序 | 確率行列の列順と一致させる。ラベル欠落時の整合性確保に使う |
| `eps` | クリップ下限 | デフォルトは `1e-15`。数値エラーが出る場合に調整 |
| `normalize` | 平均を取るか | `True`（既定）で平均、`False` で合計損失。サンプル数を変えず比較したい場合は `True` のままが推奨 |

---

## 5. 実務での活用

- **確率校正の確認**：Platt scaling や isotonic regression で校正した後に Log Loss の改善をチェック。
- **ランキング評価**：Kaggle などのコンペでは Log Loss が主要評価軸として使われるケースが多い。
- **アンサンブル**：確率を平均する際、Log Loss を指標にすると適切な重みを決めやすい。重み付き平均で最小化するよう探索すると改善が見込めます。
- **しきい値選択**：Log Loss は確率の質を評価する指標なので、Accuracy だけでは分からない改善余地を把握できます。

---

## まとめ

- Log Loss は確率予測の信頼度を評価する指標で、自信過剰な予測には大きなペナルティを課す。
- 確率出力をそのまま渡すこと、数値安定化のためのクリッピングを行うことが重要。
- Accuracy や ROC-AUC と併用し、確率の質と判別能力の両面からモデルを評価しよう。

---
