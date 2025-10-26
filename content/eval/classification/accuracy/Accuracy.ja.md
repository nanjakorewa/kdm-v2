---
title: "正解率（Accuracy）"
pre: "4.3.1 "
weight: 1
title_suffix: "まず押さえておきたい基本指標"
---

{{< lead >}}
正解率（Accuracy）は「正しく分類できたサンプルの割合」を示す最も基本的な評価指標です。データセット全体でどれだけ当たったかを直感的に把握できますが、クラス不均衡では過大評価に注意が必要です。
{{< /lead >}}

---

## 1. 定義

二値分類では真陽性 (TP)・真陰性 (TN)・偽陽性 (FP)・偽陰性 (FN) を用いて次のように表されます。

$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

多クラスでも「正しく分類できたサンプル数 ÷ 全サンプル数」で計算します。

---

## 2. Python で計算

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

model = RandomForestClassifier(random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy = {acc:.3f}")
```

`accuracy_score` は `normalize=True` が既定で割合を返します。`normalize=False` にすると正解数そのものが返る点に注意しましょう。

---

## 3. クラス不均衡への対策

Accuracy は多数派クラスを予測するだけでも高くなるため、不均衡データでは過信禁物です。以下の対策が有効です。

- **混同行列を確認**：どのクラスで誤りが多いかを把握する。
- **適合率・再現率・F1 を併用**：少数派クラスに焦点を当てた指標とセットで評価。
- **Balanced Accuracy**：各クラスの再現率を平均することで、不均衡でも公平に評価できます。

```python
from sklearn.metrics import balanced_accuracy_score

bal_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy = {bal_acc:.3f}")
```

Balanced Accuracy は各クラスの再現率を同じ重みで平均するため、クラス比が歪んでいるときのベースライン指標として有効です。

---

## 4. しきい値調整との関係

確率を出力できるモデルでは、しきい値を変更すると Accuracy が変化します。ROC 曲線や Precision-Recall 曲線を確認しながら、どのしきい値で Accuracy と他指標のバランスが取れるか検討しましょう。

---

## まとめ

- Accuracy は最もシンプルな評価指標で、全体的な当たりやすさを素早く把握できる。
- クラス不均衡では単独で判断せず、混同行列や Balanced Accuracy などと組み合わせて評価する。
- しきい値の調整や重み付き学習を取り入れると、不均衡データでもより信頼できる指標になる。

---
