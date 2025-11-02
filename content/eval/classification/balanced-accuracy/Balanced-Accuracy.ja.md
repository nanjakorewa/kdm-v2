---

title: "バランスド正解率（Balanced Accuracy）| 不均衡データでの評価"

linkTitle: "Balanced Accuracy"

seo_title: "バランスド正解率（Balanced Accuracy）| 不均衡データでの評価"

pre: "4.3.6 "

weight: 6

---



{{< lead >}}

Balanced Accuracy はクラスごとの再現率を平均した指標で、陽性・陰性のバランスが崩れたデータでも過大評価を避けられます。Python 3.13 のコード例で Accuracy との違いを確認し、どのようなケースで採用すべきか整理します。

{{< /lead >}}



---



## 1. 定義



陽性クラスの再現率（TPR）と陰性クラスの再現率（TNR）を平均したものが Balanced Accuracy です。混同行列の記号を使うと次式になります。





\mathrm{Balanced\ Accuracy} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)





多クラス分類の場合も、各クラスの再現率を平均すれば同様の考え方で拡張できます。



---



## 2. Python 3.13 での実装



```bash

python --version        # 例: Python 3.13.0

pip install scikit-learn matplotlib

```



以下のコードは Accuracy との比較に使ったものと同じで、ランダムフォレストを学習させたあと alanced_accuracy_score と ccuracy_score を並べています。図は Accuracy の記事でも使っている static/images/eval/classification/accuracy/accuracy_vs_balanced.png を共有します。



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

```



{{< figure src="/images/eval/classification/accuracy/accuracy_vs_balanced.png" alt="Accuracy と Balanced Accuracy の比較" caption="Balanced Accuracy はクラスごとの再現率を平均するため、不均衡データでも過大評価を避けられる。" >}}



---



## 3. いつ Balanced Accuracy を使うか



- **クラス不均衡が大きいとき** … Accuracy だけでは多い方のクラスを当ててしまうだけで高得点になる。Balanced Accuracy なら少数クラスの再現率も同じ重みで評価できる。

- **モデル間の比較** … 不均衡データ上で複数モデルを比較する際、Balanced Accuracy を併用すると能力差が浮き彫りになる。

- **閾値調整の目安** … Precision / Recall と組み合わせれば、どの閾値で両クラスをバランスよく拾えているかが分かる。



---



## 4. 他指標との併用



| 指標 | 何を測るか | 不均衡データでの注意点 |

| --- | --- | --- |

| Accuracy | 全体の正解割合 | 多数派クラスだけを当てても高得点になる |

| Recall / Sensitivity | 特定クラスの検出率 | クラスごとに値が異なるので別々に報告が必要 |

| **Balanced Accuracy** | クラスごとの再現率の平均 | 各クラスを同じ重みで評価できる |

| Macro F1 | Precision と Recall の調和平均（各クラス同等） | Precision も考慮したい場合に有効 |



---



## まとめ



- Balanced Accuracy は「クラスごとの再現率の平均」。不均衡データでモデルを評価する際の基準に適している。

- Python 3.13 では alanced_accuracy_score 一行で計算でき、Accuracy との違いを数値で比較できる。

- Precision / Recall / F1 など他指標と併せて、どのクラスを重視するかを明確にしながら評価を進めよう。

---

