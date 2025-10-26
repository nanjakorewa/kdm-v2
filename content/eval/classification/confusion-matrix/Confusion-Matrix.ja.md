---
title: "混同行列の読み方"
pre: "4.3.0 "
weight: 0
title_suffix: "分類モデルの基本を押さえる"
---

{{< lead >}}
混同行列は分類モデルがどのクラスを取り違えたのかを一覧できる基本指標です。適合率・再現率・F1 を計算する前に、まず混同行列で誤分類の傾向を直感的に把握しましょう。
{{< /lead >}}

---

## 1. 混同行列とは

二値分類では以下の 2×2 行列で表現します。

|            | 予測:陽性 | 予測:陰性 |
| ---------- | --------- | --------- |
| **実際:陽性** | 真陽性 (TP) | 偽陰性 (FN) |
| **実際:陰性** | 偽陽性 (FP) | 真陰性 (TN) |

- 行は「実際のクラス」、列は「モデルの予測」。  
- どのクラスで誤りが多いかを直接確認できるので、閾値調整や学習データの再設計に役立ちます。

---

## 2. Python で計算

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.show()
```

`ConfusionMatrixDisplay` を使うとヒートマップで視覚化できます。左上と右下が大きいほど正しい予測が多いことを意味します。

---

## 3. 正規化して割合を見る

クラス分布が偏っている場合は割合で見ると分かりやすくなります。

```python
cm_normalized = confusion_matrix(y_test, y_pred, normalize="true")
print(cm_normalized)
```

- `normalize="true"`：各行で割り、実際のクラスごとの再現率を表す。  
- `normalize="pred"`：各列で割り、予測クラスごとの適合率を表す。  
- `normalize="all"`：全体で割り、割合表示だけ欲しいときに使う。

正規化すると値が 0〜1 になり、モデルの弱点を客観的に比較しやすくなります。

---

## 4. 多クラス分類での見方

多クラスの場合は行と列がクラス数だけ増えます。  
- 対角線上に並ぶ値が正解数。  
- 特定の列や行に大きな値が集中していれば、そのクラスで誤りが偏っているサイン。  
- ヒートマップを並べて可視化すると、どのラベルを取り違えているか一目で判断できます。

```python
ConfusionMatrixDisplay.from_predictions(
    y_test_multi, y_pred_multi, normalize="true", values_format=".2f"
)
```

---

## 5. 実務での活用ヒント

- **閾値調整**：偽陰性が許されないタスクでは、混同行列で FN をチェックしながら予測確率の閾値を下げる。
- **データ再収集**：誤分類が多いクラスに追加データを集めたり、サンプルウェイトを設定する根拠にできる。
- **評価指標との連携**：ROC-AUC や PR 曲線と合わせると、量的（スコア）・質的（誤りの種類）両面から分析できる。
- **説明資料に載せる**：ビジネスサイドに誤分類の影響を説明する際に直感的で伝わりやすい。

---

## まとめ

- 混同行列は分類モデルの誤りの種類と量を可視化する基礎指標。
- `normalize` オプションで割合を確認すると、クラス不均衡でも比較しやすい。
- 閾値調整やデータ改善の方針を決めるために、他の評価指標と併用すると効果的。

---
