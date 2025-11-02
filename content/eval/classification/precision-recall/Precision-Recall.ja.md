---

title: "適合率・再現率・F1スコア | Python 3.13で理解する閾値設計"

linkTitle: "Precision-Recall"

seo_title: "適合率・再現率・F1スコア | Python 3.13で理解する閾値設計"

pre: "4.3.2 "

weight: 2

---



{{< lead >}}

適合率（Precision）と再現率（Recall）は、陽性と予測したサンプルの正しさ、真の陽性を取りこぼさない力をそれぞれ測る指標です。Python 3.13 環境で計算・可視化し、F1 スコアや平均適合率（AP）と合わせて閾値調整に役立てましょう。

{{< /lead >}}



---



## 1. 定義と直感



混同行列の記号を \(TP\)（真陽性）、\(FP\)（偽陽性）、\(FN\)（偽陰性）、\(TN\)（真陰性）とすると、





\text{Precision} = \frac{TP}{TP + FP}, \qquad

\text{Recall} = \frac{TP}{TP + FN}





- **適合率** … 陽性と判定したうち、本当に陽性だった割合。誤報（FP）を抑えたいときに重視。

- **再現率** … 実際の陽性のうち、正しく検出できた割合。見逃し（FN）が問題になるときに重視。

- **F1 スコア** … 適合率と再現率の調和平均。双方を同程度に重視したいときのまとめ指標。





F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}





---



## 2. Python 3.13 での実装と PR 曲線



環境を確認し、必要なパッケージを導入します。



```bash

python --version        # 例: Python 3.13.0

pip install scikit-learn matplotlib

```



次のコードでは少数クラスが 5 % のデータセットを生成し、class_weight="balanced" 付きロジスティック回帰で PR 曲線と平均適合率（AP）を描画します。図は static/images/eval/classification/precision-recall/pr_curve.png に保存してあり、generate_eval_assets.py から再生成できます。



```python

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    ConfusionMatrixDisplay,

    classification_report,

    precision_recall_curve,

    precision_score,

    recall_score,

    f1_score,

    average_precision_score,

)

from sklearn.model_selection import train_test_split



X, y = make_classification(

    n_samples=40_000,

    n_features=20,

    n_informative=6,

    weights=[0.95, 0.05],

    random_state=42,

)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, stratify=y, random_state=42

)



clf = LogisticRegression(max_iter=2000, class_weight="balanced")

clf.fit(X_train, y_train)



proba = clf.predict_proba(X_test)[:, 1]

y_pred = (proba >= 0.5).astype(int)

print(classification_report(y_test, y_pred, digits=3))



precision, recall, thresholds = precision_recall_curve(y_test, proba)

ap = average_precision_score(y_test, proba)



fig, ax = plt.subplots(figsize=(5, 4))

ax.step(recall, precision, where="post", label=f"PR curve (AP={ap:.3f})")

ax.set_xlabel("Recall")

ax.set_ylabel("Precision")

ax.set_xlim(0, 1)

ax.set_ylim(0, 1.05)

ax.grid(alpha=0.3)

ax.legend()

fig.tight_layout()

output_dir = Path("static/images/eval/classification/precision-recall")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "pr_curve.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/precision-recall/pr_curve.png" alt="Precision-Recall 曲線" caption="平均適合率（AP）は PR 曲線下の面積。閾値の移動で適合率と再現率のバランスが変わる。" >}}



---



## 3. 閾値調整の考え方



- 閾値 0.5 固定では、再現率を高くしたいケースに十分対応できないことがある。

- PR 曲線上の任意の点に相当する閾値は precision_recall_curve の 	hresholds から取得可能。

- False Negative を減らしたい場合は閾値を下げて再現率を稼ぎ、Precision の低下を許容範囲内に収める。



```python

threshold = 0.3

custom_pred = (proba >= threshold).astype(int)

print(

    "threshold=0.30",

    "Precision=", precision_score(y_test, custom_pred),

    "Recall=", recall_score(y_test, custom_pred),

    "F1=", f1_score(y_test, custom_pred),

)

```



---



## 4. 多クラス分類での平均化



scikit-learn では verage パラメータで重み付けを切り替えられます。



- macro … クラスごとの Precision / Recall の単純平均。クラス数が少なくても比較しやすい。

- weighted … 各クラスにサンプル数の重みを付ける。データ全体のバランスを維持。

- micro … 全サンプルをまとめて再計算。クラス不均衡が強いときは少数クラスが埋もれる点に注意。



```python

precision_score(y_test, y_pred, average="macro")

recall_score(y_test, y_pred, average="weighted")

f1_score(y_test, y_pred, average="micro")

```



---



## 5. まとめ



- Precision は誤警報の少なさ、Recall は見逃しの少なさを測る。両方を高めるには PR 曲線で閾値を確認する。

- F1 や平均適合率（AP）を併用すると指標を 1 つに集約でき、モデル比較や閾値調整の議論がしやすい。

- Python 3.13 では scikit-learn の precision_recall_curve / verage_precision_score を使えば数行で計算・可視化できる。

---

