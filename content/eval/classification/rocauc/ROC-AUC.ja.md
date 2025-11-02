---

title: "ROC-AUC | 閾値設計とモデル比較を支える指標"

linkTitle: "ROC-AUC"

seo_title: "ROC-AUC | 閾値設計とモデル比較を支える指標"

pre: "4.3.3 "

weight: 1

---



{{< lead >}}

ROC-AUC は ROC 曲線（受信者動作特性曲線）の下側面積を指し、クラス分類モデルの確信度を評価する代表指標です。Python 3.13 環境での再現コードとともに、閾値調整にどう使うかを整理します。

{{< /lead >}}



---



## 1. ROC 曲線と AUC の定義



ROC 曲線は **False Positive Rate (FPR)** を横軸、**True Positive Rate (TPR)** を縦軸に取った曲線です。分類器の閾値を 0〜1 の間で動かすことで得られ、AUC はその面積を 0〜1 の範囲で表します。



- AUC = 1.0: 完全に理想的な分類

- AUC = 0.5: 完全なランダム予測（斜め 45 度の線）

- AUC < 0.5: 予測がほぼ逆転している可能性（閾値や符号を反転すれば改善余地あり）



---



## 2. Python 3.13 での実装と可視化



環境を確認し、必要なライブラリを導入します。



```bash

python --version        # 例: Python 3.13.0

pip install scikit-learn matplotlib

```



下記コードは乳がん診断データセットにロジスティック回帰を適用し、ROC 曲線と AUC を描画します。generate_eval_assets.py から自動生成できるよう、図版は static/images/eval/classification/rocauc に保存しています。



```python

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import RocCurveDisplay, roc_auc_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, random_state=42, stratify=y

)



pipeline = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, solver="lbfgs"),

)

pipeline.fit(X_train, y_train)

proba = pipeline.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, proba)

print(f"ROC-AUC: {auc:.3f}")



fig, ax = plt.subplots(figsize=(5, 5))

roc_display = RocCurveDisplay.from_predictions(

    y_test,

    proba,

    name="Logistic Regression",

    ax=ax,

)

ax.plot([0, 1], [0, 1], "--", color="grey", alpha=0.5, label="Random")

ax.set_xlabel("False Positive Rate")

ax.set_ylabel("True Positive Rate")

ax.set_title("ROC Curve (Breast Cancer Dataset)")

ax.legend(loc="lower right")

fig.tight_layout()

output_dir = Path("static/images/eval/classification/rocauc")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "roc_curve.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/rocauc/roc_curve.png" alt="ROC 曲線" caption="ROC 曲線の面積 (AUC) がモデルの判別性能を表す" >}}



---



## 3. 閾値調整にどう使うか



- **医療・不正検知などで Recall を重視** → ROC 曲線で TPR を高く保ちつつ許容可能な FPR を探る。

- **精度と再現率のバランス** → AUC が高いモデルは、閾値0.5以外でも性能が安定していることが多い。

- **複数モデルの比較** → AUC が高いモデルほど、全体的な判別能力が高いと期待できる。



閾値を変更すると Precision-Recall のバランスも変化するため、ROC-AUC と PR-AUC を合わせて確認すると意思決定がスムーズになります。



---



## 4. 実運用でのチェックリスト



1. **データが大きく偏っていないか** – 0.5 付近の AUC でも別の閾値で救える場合がある。

2. **クラス重みを調整した場合の AUC** – サンプルウェイトやクラス重みを変えても ROC-AUC が改善するかを確認。

3. **可視化をダッシュボード化** – 人間が閾値を調整しやすいよう、ROC 曲線を共有する。

4. **Python 3.13 ノートブックで再現** – モデルの更新時にも同じ手順で計算・比較できるようにしておく。



---



## まとめ



- ROC-AUC は閾値を横断的に評価できる指標で、0.5〜1.0 の範囲で性能を把握できる。

- Python 3.13 + scikit-learn では RocCurveDisplay と 

oc_auc_score を組み合わせると簡潔に可視化と評価が可能。

- 閾値チューニングやモデル比較とセットで活用し、Precision-Recall と合わせて総合的に判断しよう。

---

