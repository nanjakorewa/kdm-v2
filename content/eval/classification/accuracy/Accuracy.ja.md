---

title: "正解率（Accuracy）| Python 3.13 での基礎と落とし穴"

linkTitle: "Accuracy"

seo_title: "正解率（Accuracy）| Python 3.13 での基礎と落とし穴"

pre: "4.3.1 "

weight: 1

---



{{< lead >}}

Accuracy は「正しく当てたサンプルの割合」を示す最も基本的な指標です。しかしクラスが偏っているデータでは過大評価につながるため、Python 3.13 の環境で再現可能なコードとともに、補助指標の使い分けを整理します。

{{< /lead >}}



---



## 1. 定義



混同行列の各成分（真陽性 TP / 偽陽性 FP / 偽陰性 FN / 真陰性 TN）を用いて、Accuracy は次式で定義されます。



$$

\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}

$$



**全サンプルのうち何件正解したか** を一目で把握できますが、クラス不均衡を含むデータでは補助指標とセットで評価する必要があります。



---



## 2. Python 3.13 での実装と可視化



まず Python のバージョンを確認し、必要なライブラリをインストールします。



```bash

python --version        # 例: Python 3.13.0

pip install scikit-learn matplotlib

```



以下のコードでは乳がん診断データセットにランダムフォレストを適用し、Accuracy と Balanced Accuracy を比較する棒グラフを生成します。`Pipeline` と `StandardScaler` を使ってスケーリングを自動化し、図版は `static/images/eval/...` に保存されます（`generate_eval_assets.py` から実行可能）。



```python

import matplotlib.pyplot as plt

import numpy as np

from pathlib import Path

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.25, random_state=42, stratify=y

)



pipeline = make_pipeline(

    StandardScaler(),

    RandomForestClassifier(random_state=42, n_estimators=300),

)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



acc = accuracy_score(y_test, y_pred)

bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc:.3f}, Balanced Accuracy: {bal_acc:.3f}")



fig, ax = plt.subplots(figsize=(5, 4))

scores = np.array([acc, bal_acc])

labels = ["Accuracy", "Balanced Accuracy"]

colors = ["#2563eb", "#f97316"]

ax.bar(labels, scores, color=colors)

ax.set_ylim(0, 1.05)

for label, score in zip(labels, scores):

    ax.text(label, score + 0.02, f"{score:.3f}", ha="center", va="bottom", fontsize=11)

ax.set_ylabel("Score")

ax.set_title("Accuracy vs. Balanced Accuracy (Breast Cancer Dataset)")

ax.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()

output_dir = Path("static/images/eval/classification/accuracy")

output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "accuracy_vs_balanced.png", dpi=150)

plt.close(fig)

```



{{< figure src="/images/eval/classification/accuracy/accuracy_vs_balanced.png" alt="Accuracy と Balanced Accuracy の比較" caption="クラスが偏ると Balanced Accuracy での再評価が有効になる例" >}}



---



## 3. クラス不均衡への対処



Accuracy だけでは **少数クラスの識別性能が隠れてしまう** ため、以下の補助指標を併用しましょう。



- **Precision / Recall / F1**: 誤警報と見逃しのバランスを確認。

- **Balanced Accuracy**: 各クラスの再現率を平均化し、偏りを補正。

- **Confusion Matrix**: どのクラスで間違えたかを視覚的に把握。

- **ROC-AUC / PR 曲線**: 予測確率を評価し、閾値調整の余地を探る。



Balanced Accuracy は `accuracy_score(..., normalize=False)` で件数を取得し、クラスごとに平均を取るイメージです。少数クラスを重視するケースではこちらを報告値に採用することが推奨されます。



---



## 4. 現場でのチェックリスト



1. **ビジネス要件に合うか？** 不均衡データで「 Accuracy 99% 」と言われても、重要クラスを取りこぼしていないか必ず混同行列で確認する。

2. **閾値の調整余地は？** ROC-AUC や PR 曲線を併用し、閾値を変えたときに Accuracy がどう動くかを検証する。

3. **メトリクスを複数報告する**: Accuracy と合わせて Precision / Recall / F1 / Balanced Accuracy を揃えると、偏りが可視化され意思決定がしやすい。

4. **再現可能なノートブックを用意する**: Python 3.13 環境で実行可能な Notebook を残し、モデル再訓練時にも同じ評価を素早く回せるようにする。



---



## まとめ



- Accuracy は全体的な当て方を手早く把握できる指標だが、クラス不均衡では過大評価に注意。

- Python 3.13 + scikit-learn での実装では、`Pipeline` と `StandardScaler` を使えば安定した結果が得やすい。

- Balanced Accuracy や Precision / Recall などの補助指標を合わせて確認し、モデルの癖やビジネスリスクに即した評価を行おう。

