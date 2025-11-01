---
title: "平均適合率（Average Precision, AP）| PR 曲線で見るモデル品質"
linkTitle: "Average Precision"
seo_title: "平均適合率（Average Precision, AP）| PR 曲線で見るモデル品質"
pre: "4.3.9 "
weight: 9
---

{{< lead >}}
Average Precision（AP）は Precision–Recall 曲線の下側面積を、再現率の変化量を重みとして積み上げた指標です。Python 3.13 のコードで算出し、F1 や ROC-AUC と並べてモデル品質を評価しましょう。
{{< /lead >}}

---

## 1. 定義

PR 曲線上の各点を \((R_n, P_n)\) とすると、Average Precision は次のように表されます。


\mathrm{AP} = \sum_{n}(R_n - R_{n-1}) P_n


再現率（Recall）の増加分を重みとして適合率（Precision）を積み上げるため、閾値を下げていくときの平均的な精度を表現できます。

---

## 2. Python 3.13 での計算

`ash
python --version        # 例: Python 3.13.0
pip install scikit-learn matplotlib
`

Precision-Recall の記事で生成した確率出力 proba を再利用し、precision_recall_curve と verage_precision_score で AP を計算します。

`python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, proba)
ap = average_precision_score(y_test, proba)
print(f"Average Precision: {ap:.3f}")
`

PR 曲線の描画は先ほどのスクリプトで保存した pr_curve.png を参照できます。

{{< figure src="/images/eval/classification/precision-recall/pr_curve.png" alt="Precision-Recall 曲線" caption="AP は PR 曲線の下側面積を再現率の増分で重み付けした指標。" >}}

---

## 3. AP と PR-AUC の違い

scikit-learn では verage_precision_score が AP、sklearn.metrics.auc(recall, precision) が単純な台形公式による PR-AUC を返します。AP はステップ状の曲線を想定して再現率の変化量を重視するため、クラス不均衡なデータでより安定した評価が得られることが多いです。

---

## 4. 実務での活用ポイント

- **閾値選択の指針** … AP が高いモデルほど、広い閾値範囲で高い Precision を維持しやすい。
- **ランキング課題の評価** … レコメンドや情報検索では、MAP（平均 AP）として各クエリの AP を平均するのが一般的。
- **F1 との比較** … F1 は特定の閾値に依存する一方、AP は閾値全体を通した性能を可視化できる。

---

## まとめ

- Average Precision は Precision–Recall 曲線全体の品質を数値化した指標。少数クラスの挙動も反映される。
- Python 3.13 + scikit-learn では verage_precision_score を使えば数行で算出できる。
- F1、ROC-AUC、精度曲線と合わせて利用し、モデル選定や閾値調整の議論をスムーズに進めよう。
---
