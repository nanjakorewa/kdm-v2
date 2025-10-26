---
title: "マシューズ相関係数（MCC）"
pre: "4.3.5 "
weight: 5
title_suffix: "不均衡データに強い総合指標"
---

{{< lead >}}
マシューズ相関係数（Matthews Correlation Coefficient, MCC）は混同行列を全要素使って計算され、クラス不均衡でもバランス良く性能を評価できます。-1〜1 の範囲を取るため、分類器の良し悪しを直感的に把握しやすい指標です。
{{< /lead >}}

---

## 1. 定義

二値分類における MCC は次の式で表されます。

$$
\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{
  \sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}
}
$$

- 1 に近いほど性能が良い  
- 0 の場合はランダム予測と同程度  
- -1 は完全に逆の予測（すべて外れ）

多クラス版では混同行列全体を対象に計算する一般化式が用意されています。

---

## 2. Python で計算

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, confusion_matrix

X, y = make_classification(
    n_samples=2000,
    n_features=20,
    weights=[0.9, 0.1],
    random_state=0,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

clf = SVC(kernel="rbf", class_weight="balanced")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

mcc = matthews_corrcoef(y_test, y_pred)
print("MCC:", round(mcc, 3))
print(confusion_matrix(y_test, y_pred))
```

`class_weight="balanced"` を指定すると少数派クラスの重みが自動調整され、MCC の改善につながることが多いです。

---

## 3. 特徴と直感

- **不均衡に強い**：Accuracy や F1 が高いのに実は負例を大量に誤分類している、といったケースを検知しやすい。
- **相関係数として解釈できる**：1 に近いほど「正しく相関」、-1 に近いほど「逆相関」、0 付近は意味のある関係がないと考えられる。
- **全要素を利用**：TP/TN/FP/FN すべてが式に現れるため、どれかがゼロでも他の値で補正される。ただし分母が 0 になる場合は scikit-learn が 0 を返す。

---

## 4. 他指標との比較

| 指標 | 強み | 注意点 |
| --- | --- | --- |
| Accuracy | シンプルで共有しやすい | 不均衡データで過大評価しやすい |
| F1 | 陽性クラスのバランス確認 | TN を考慮しない |
| **MCC** | TN を含む全体評価、-1〜1 で直感的 | 計算がやや複雑、値の変化が小さめ |

MCC が高いと Precision/Recall なども一定水準以上になることが多く、総合的な指標として便利です。

---

## 5. 実務での活用

- **モデルの最終比較**：Accuracy が近い複数モデルの中から、MCC が高いものを選ぶとよりバランスの良いモデルを得やすい。
- **しきい値のチューニング**：予測確率のしきい値を変えながら MCC を最大化する方法もある。Sklearn の `make_scorer(matthews_corrcoef)` と GridSearchCV を組み合わせれば自動化可能。
- **報告資料での補助指標**：ROC-AUC や PR-AUC に加えて MCC を提示すると、クラス不均衡の影響に配慮していることを示せる。

---

## まとめ

- MCC は混同行列の全要素を利用し、クラス不均衡でも安定した評価ができる。
- 値が -1〜1 の範囲に収まるため直感的に解釈しやすく、Accuracy や F1 を補完する指標として有用。
- しきい値調整やクラス重みの最適化と組み合わせて活用し、バランスの取れた分類モデルを選定しよう。

---
