---
title: "平均化戦略（Averaging Strategies）を使い分ける"
linkTitle: "平均化戦略"
seo_title: "平均化戦略（Averaging Strategies）| マルチクラス評価の基本"
pre: "4.3.14 "
weight: 14
---

{{< lead >}}
マルチクラス／マルチラベル分類で Precision・Recall・F1 などを集計するときは、`average` 引数で「どのように平均を取るか」を指定します。Python 3.13 と scikit-learn の実例とともに、代表的な 4 つの平均化方式を整理しましょう。
{{< /lead >}}

---

## 1. 主な平均化の種類
| average    | 計算方法                                        | 特徴・主な利用場面                                        |
| ---------- | ----------------------------------------------- | --------------------------------------------------------- |
| `micro`    | すべてのサンプルの TP/FP/FN を合算して指標を算出 | クラスごとの重み付けは行わず、全体の正解率を重視する場合 |
| `macro`    | クラスごとに指標を計算し、単純平均               | 少数クラスも同じ重みで扱える。医療・不正検知などで有効     |
| `weighted` | クラスごとに指標を計算し、サンプル数で加重平均   | 実データのクラス比を保ちながら平均を取りたい場合          |
| `samples`  | マルチラベル用。サンプルごとに平均               | 1 件に複数ラベルが付く画像タグ分類などで標準的            |

---

## 2. Python 3.13 での比較
```bash
python --version  # 例: Python 3.13.0
pip install scikit-learn matplotlib
```

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=30_000,
    n_features=20,
    n_informative=6,
    weights=[0.85, 0.1, 0.05],  # クラス不均衡
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, multi_class="ovr"),
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, digits=3))
for avg in ["micro", "macro", "weighted"]:
    print(f"F1 ({avg}):", f1_score(y_test, y_pred, average=avg))
```

`classification_report` は各クラスの指標と `macro avg` / `weighted avg` / `micro avg` を同時に表示してくれるため、平均化方式の違いをすぐ比較できます。

---

## 3. 使い分けのヒント
- **micro** … モデルの総合的な正解率を重視したいとき。Kaggle などで全サンプルの正答率を確認するときに便利。
- **macro** … 少数クラスも同じ重みで扱いたいとき。医療や不正検知など取りこぼしが許されない場面に向く。
- **weighted** … 現実のクラス比を保ったまま評価したいとき。Accuracy に近い感覚で Precision/Recall/F1 を報告できる。
- **samples** … マルチラベル分類で 1 サンプル複数ラベルの性能を測りたいときの標準的な選択。

---

## まとめ
- `average` の選択で同じモデルでも指標の意味合いが大きく変わる。タスクとビジネス要件に合わせて使い分けることが重要。
- `macro` はクラス間を公平に、`micro` は全体の比率を重視、`weighted` はクラス比を保った平均、`samples` はマルチラベル専用と覚えておこう。
- scikit-learn の `f1_score` などで複数の平均化方式を簡単に計算できるので、併記すると指標の読み違いを防げる。
