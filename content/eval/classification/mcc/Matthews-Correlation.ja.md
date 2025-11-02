---

title: "マシューズ相関係数（MCC）| 全要素を使って判定する"

linkTitle: "MCC"

seo_title: "マシューズ相関係数（MCC）| 全要素を使って判定する"

pre: "4.3.5 "

weight: 5

---



{{< lead >}}

マシューズ相関係数（Matthews Correlation Coefficient, MCC）は TP / FP / FN / TN の全てを考慮した相関指標です。-1 〜 1 のスケールで分類性能を評価でき、クラス不均衡にも強いため、Accuracy や F1 スコアの補助として有用です。

{{< /lead >}}



---



## 1. 定義



二値分類において MCC は次式で表されます。





\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}





- **1** … 完全に正しい分類

- **0** … ランダム予測と同程度

- **−1** … 完全に逆の予測



多クラスでも同様に、クラスごとの混同行列から拡張できます。



---



## 2. Python 3.13 での計算



```bash

python --version        # 例: Python 3.13.0

pip install scikit-learn matplotlib

```



```python

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import matthews_corrcoef, confusion_matrix

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



print(confusion_matrix(y_test, y_pred))

print("MCC:", matthews_corrcoef(y_test, y_pred))

```



class_weight="balanced" を指定すると少数クラスの影響も MCC に反映しやすくなります。



---



## 3. 閾値と MCC の関係



確率出力から MCC を閾値ごとに算出すると、バランスの良いポイントが分かります。



{{< figure src="/images/eval/classification/mcc/mcc_vs_threshold.png" alt="MCC と閾値" caption="閾値ごとの MCC。クラス不均衡でも、最大値付近の閾値を使えば相関が高まりやすい。" >}}



F1 スコアと異なり TN も評価に含まれるため、クラス比率が極端でも安定して比較できます。



---



## 4. 実務での活用



- **Accuracy の補助** – Accuracy が高くても MCC が低い場合、どちらかのクラスが無視されている可能性があります。

- **モデル比較** – Grid Search のスコアリングとして make_scorer(matthews_corrcoef) を指定すれば MCC を最適化できます。

- **閾値チューニング** – ROC や PR 曲線と併用し、MCC が最大になる閾値を候補にすると全体バランスが把握しやすいです。



---



## まとめ



- MCC は TP/FP/FN/TN を同時に考慮する相関指標で、−1〜1 のスケールでわかりやすく評価できる。

- Python 3.13 では matthews_corrcoef で簡単に算出でき、閾値別に可視化すると最適な動作点が見える。

- Accuracy や F1 と併用して、クラス不均衡下でも偏りのない評価を行おう。

---

