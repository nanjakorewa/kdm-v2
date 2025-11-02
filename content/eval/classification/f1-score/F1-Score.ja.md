---

title: "F1スコア | 適合率と再現率の調和平均"

linkTitle: "F1 Score"

seo_title: "F1スコア | 適合率と再現率の調和平均"

pre: "4.3.8 "

weight: 8

---



{{< lead >}}

F1 スコアは適合率（Precision）と再現率（Recall）の調和平均で、誤警報と見逃しをバランスよく抑えたいときに有効です。Python 3.13 のコードで計算・可視化し、閾値調整や Fβ スコアの使い分けまで整理します。

{{< /lead >}}



---



## 1. 定義



適合率を \(P\)、再現率を \(R\) とすると F1 スコアは





F_1 = 2 \cdot \frac{P \cdot R}{P + R}





で表せます。より一般的な Fβ スコアは





F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}





となり、\(\beta > 1\) で再現率を、\(\beta < 1\) で適合率をより重視します。



---



## 2. Python 3.13 での計算



```bash

python --version        # 例: Python 3.13.0

pip install scikit-learn matplotlib

```



```python

import numpy as np

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (

    classification_report,

    f1_score,

    fbeta_score,

)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



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



model = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, class_weight="balanced"),

)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)



print(classification_report(y_test, y_pred, digits=3))

print("F1:", f1_score(y_test, y_pred))

print("F0.5:", fbeta_score(y_test, y_pred, beta=0.5))

print("F2:", fbeta_score(y_test, y_pred, beta=2.0))

```



classification_report ではクラス別の Precision / Recall / F1 がまとめて確認できます。



---



## 3. 閾値と F1 スコアの関係



確率出力 predict_proba を使えば、閾値を動かしたときに F1 がどう変わるかを可視化できます。



```python

from sklearn.metrics import precision_recall_curve, f1_score



proba = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, proba)

thresholds = np.append(thresholds, 1.0)

f1_scores = [

    f1_score(y_test, (proba >= t).astype(int))

    for t in thresholds

]

```



{{< figure src="/images/eval/classification/f1-score/f1_vs_threshold.png" alt="F1 スコアと閾値" caption="閾値によって F1 は変動する。Recall を優先したい場合は閾値を下げて再計算する。" >}}



- F1 が最大になる閾値を探せば、Precision / Recall のバランスが最も良い点が分かります。

- Recall をさらに重視したい場合は F2 や F0.5 といった Fβ スコアで閾値を評価し直すと良いでしょう。



---



## 4. 多クラスでの平均化



scikit-learn の verage 引数を切り替えることで、多クラス問題でも F1 を集計できます。



- macro … クラスごとに F1 を計算し単純平均。クラス数が少なくても公平。

- weighted … クラスごとの F1 をサンプル数で加重平均。全体のバランスを保つ。

- micro … 混同行列を合算してから計算。クラス不均衡の影響を受けやすい点に注意。



```python

from sklearn.metrics import f1_score



f1_macro = f1_score(y_test, y_pred, average="macro")

f1_weighted = f1_score(y_test, y_pred, average="weighted")

```



サンプル単位のマルチラベルでは verage="samples" も利用できます。



---



## まとめ



- F1 スコアは適合率と再現率を同じ比重で評価する調和平均。閾値により大きく変動するので可視化して確認する。

- Fβ スコアを使えば、見逃し重視／誤検知重視など要件に応じて重み付けを変えられる。

- 多クラスやマルチラベルでは平均化方式を明示し、Precision / Recall / PR 曲線と併せて総合的にモデルの挙動を把握しよう。

---

