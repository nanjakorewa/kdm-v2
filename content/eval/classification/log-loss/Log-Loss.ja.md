---

title: "対数損失（Log Loss）| 予測確率の信頼性を測る"

linkTitle: "Log Loss"

seo_title: "対数損失（Log Loss）| 予測確率の信頼性を測る"

pre: "4.3.4 "

weight: 4

---



{{< lead >}}

Log Loss（Cross Entropy）は、予測確率と実際の結果のずれを評価する指標です。確率を間違えるほどペナルティが急速に大きくなるため、キャリブレーションが重要なシーンで役立ちます。Python 3.13 での計算手順と理解に役立つグラフを紹介します。

{{< /lead >}}



---



## 1. 定義



二値分類の Log Loss は次式で表されます。





\mathrm{LogLoss} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]





ここで \(p_i\) は陽性クラスの予測確率、\(y_i\) は正解ラベル（0 または 1）です。多クラスでは各クラスの確率とワンホットラベルの積を同様に加算します。



---



## 2. Python 3.13 での計算



```bash

python --version        # 例: Python 3.13.0

pip install scikit-learn matplotlib

```



```python

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss, classification_report

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.3, stratify=y, random_state=42

)



model = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=2000, solver="lbfgs"),

)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)



print(classification_report(y_test, model.predict(X_test), digits=3))

print("Log Loss:", log_loss(y_test, proba))

```



predict_proba が返す確率を log_loss に渡すだけで評価できます。



---



## 3. ペナルティのイメージ



推定確率が真のラベルとズレるほど Log Loss は急激に増えます。



{{< figure src="/images/eval/classification/log-loss/log_loss_curves.png" alt="Log Loss のペナルティ" caption="実際のラベルが 1 のときは −log(p)、0 のときは −log(1−p) がペナルティになる。確率を外すほど急増する。" >}}



- 陽性サンプルに低い確率（例: 0.1）を付与すると大きく罰せられます。

- 0.5 のような無難な確率ばかりでは、検出性能が低いと判断されます。



---



## 4. 実務での使いどころ



- **キャリブレーションの確認** — Platt scaling や isotonic regression で補正したあとの効果測定に最適。

- **コンペティション** — Kaggle など確率を評価するコンテストで採用されることが多い指標。

- **閾値に依存しない比較** — Accuracy のように閾値固定ではないため、ランキング性能や確率の信頼性をまとめて評価できます。



scikit-learn の log_loss には labels, eps, 

ormalize などの引数があり、欠損ラベルや数値の安定性に配慮できます。



---



## まとめ



- Log Loss は確率予測の「どれだけ自信を外したか」を測る指標。小さいほど良い。

- Python 3.13 では log_loss と predict_proba で簡単に算出できる。

- Accuracy や ROC-AUC だけでなく、Log Loss で確率の品質を併せて確認しよう。

---

