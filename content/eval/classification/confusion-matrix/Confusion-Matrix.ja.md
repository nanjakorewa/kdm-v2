---

title: "混同行列 | クラス分類の評価指標を読み解く"

linkTitle: "混同行列"

seo_title: "混同行列 | クラス分類の評価指標を読み解く"

pre: "4.3.0 "

weight: 0

---



{{< lead >}}

混同行列は、分類モデルが各クラスをどのように判定したかを表形式で可視化する基本指標です。正解との突合せを通じて、精度・再現率・F1 スコアなど派生メトリクスを理解しやすくなります。

{{< /lead >}}



---



## 1. 混同行列の構造



二値分類の混同行列は次の 2×2 テーブルで表現できます。



|              | 予測:陰性 | 予測:陽性 |

| ------------ | --------- | --------- |

| **実際:陰性** | 真陰性 (TN) | 偽陽性 (FP) |

| **実際:陽性** | 偽陰性 (FN) | 真陽性 (TP) |



- 行は「実測値」、列は「モデルが予測した値」を意味します。

- TP・FP・FN・TN のバランスを見ることで、特定クラスに偏った判断をしていないかを把握できます。



---



## 2. Python 3.13 + scikit-learn での計算例



ローカル環境が **Python 3.13** であることを前提に、以下のようにバージョンを確認しておきましょう。



```bash

python --version  # 例: Python 3.13.0

pip install scikit-learn matplotlib

```



次のスクリプトは乳がん診断データセットにロジスティック回帰を適用し、混同行列を算出・描画します。`StandardScaler` を併用することで、収束警告を避けつつ安定した結果を得られます。



```python

from pathlib import Path



import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y

)



pipeline = make_pipeline(

    StandardScaler(),

    LogisticRegression(max_iter=1000, solver="lbfgs"),

)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)



cm = confusion_matrix(y_test, y_pred)

print(cm)



disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot(cmap="Blues", colorbar=False)

plt.tight_layout()

plt.show()

```



{{< figure src="/images/eval/confusion-matrix/binary_matrix.png" alt="乳がん診断データセットでの混同行列" caption="scikit-learn Pipeline（Python 3.13）を用いて描画した混同行列" >}}



---



## 3. 正規化して割合を確認する



データにクラス不均衡がある場合は、行（実測値）ごとに割合を出すと読みやすくなります。



```python

cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

print(cm_norm)



disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)

disp_norm.plot(cmap="Blues", values_format=".2f", colorbar=False)

plt.tight_layout()

plt.show()

```



- `normalize="true"`: 行ごとに正規化（実測クラスに対する割合）

- `normalize="pred"`: 列ごとに正規化（予測クラスに対する割合）

- `normalize="all"`: 全要素で正規化



---



## 4. 多クラス分類への拡張



多クラス分類でも同じ要領で混同行列を描画できます。`from_predictions` を使うとラベルを自動で設定できます。



```python

ConfusionMatrixDisplay.from_predictions(

    y_true=ground_truth_labels,

    y_pred=model_outputs,

    normalize="true",

    values_format=".2f",

    cmap="Blues",

)

plt.tight_layout()

plt.show()

```



---



## 5. 現場でのチェックポイント



- **偽陰性 vs. 偽陽性**: 医療や不正検知では、どちらを最小化したいかを明確にして混同行列を確認します。

- **ヒートマップと併用**: グラフ化すると偏りが直感的にわかり、チーム内の議論もスムーズになります。

- **派生指標との連携**: 混同行列から精度・適合率・再現率・F1 を算出し、ROC-AUC や PR 曲線とあわせて評価しましょう。

- **スクリプトの再利用**: Python 3.13 環境で再現可能なノートブックを用意しておくと、モデル改善サイクルが高速化します。



---



## まとめ



- 混同行列は TP・FP・FN・TN を整理し、モデルの癖を素早く把握できる評価表です。

- `normalize` オプションで割合を出すと、クラス不均衡でも比較がしやすくなります。

- 可視化したヒートマップと派生メトリクスを併用し、ビジネス要件に沿った評価基準を設計しましょう。

