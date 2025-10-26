---
title: "バランスド正解率（Balanced Accuracy）"
pre: "4.3.6 "
weight: 6
title_suffix: "クラス不均衡でも公平に評価"
---

{{< lead >}}
バランスド正解率（Balanced Accuracy）は、各クラスの再現率を単純平均した指標です。多数派クラスだけが正しく分類されても高くならないため、不均衡データのベースラインとして有効です。
{{< /lead >}}

---

## 1. 定義

二値分類では感度（True Positive Rate）と特異度（True Negative Rate）の平均として定義されます。

$$
\mathrm{Balanced\ Accuracy} = \frac{1}{2} \left( \frac{TP}{TP + FN} + \frac{TN}{TN + FP} \right)
$$

多クラスの場合は、各クラスの再現率を計算して単純平均します。

---

## 2. Python で計算

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

X, y = make_classification(
    n_samples=5000,
    n_features=20,
    weights=[0.9, 0.1],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

clf = RandomForestClassifier(class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

bal_acc = balanced_accuracy_score(y_test, pred)
print("Balanced Accuracy =", round(bal_acc, 3))
print(confusion_matrix(y_test, pred))
```

`class_weight="balanced"` を指定すると、少数派クラスの重みが自動調整され、Balanced Accuracy が改善することがあります。

---

## 3. 直感と活用

- **Accuracy との差**：多数派の正解率ばかり高いモデルは Balanced Accuracy で減点されるため、誤った安心感を防げます。
- **再現率の平均**：各クラスの再現率が同じなら Accuracy と一致します。極端な不均衡では Balanced Accuracy の方が実態を表すことが多いです。
- **しきい値調整の指標**：予測確率のしきい値を変えて Balanced Accuracy を最大化すると、各クラスの見落としを均等に抑えられます。

---

## 4. 他指標との比較

| 指標 | 特徴 | 注意点 |
| --- | --- | --- |
| Accuracy | 全体の正解率 | 不均衡だと多数派の影響が大きい |
| Recall / Sensitivity | 特定クラスに着目 | 負例は無視される |
| **Balanced Accuracy** | クラスごと再現率を平均 | 各クラスの重要度が同じという前提 |
| Macro F1 | Precision と Recall の平均 | 適合率も含めたいときに有効 |

---

## まとめ

- Balanced Accuracy は各クラスの再現率を平等に扱うため、不均衡データの評価に適している。
- `balanced_accuracy_score` で簡単に算出でき、Accuracy と並べて報告すると誤認を防げる。
- しきい値やクラス重みの調整と組み合わせ、モデルがどのクラスも適切に扱えているか確認しよう。

---
