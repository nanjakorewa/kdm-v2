---
title: "ブライアスコア（Brier Score）| 予測確率の校正を評価する"
linkTitle: "Brier Score"
seo_title: "ブライアスコア（Brier Score）| 予測確率の校正を評価する"
pre: "4.3.10 "
weight: 10
---

{{< lead >}}
ブライアスコアは、「予測した確率」と「実際のラベル（0/1）」の二乗誤差です。確率予測がどれだけキャリブレーション（校正）されているかを測る指標として、天気予報や需要予測などで広く使われます。Python 3.13 で計算し、リライアビリティカーブ（信頼度曲線）と併せてチェックしましょう。
{{< /lead >}}

---

## 1. 定義

二値分類におけるブライアスコアは次式で表されます。


\mathrm{Brier} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)^2


ここで \(p_i\) はモデルが出力した陽性確率、\(y_i\) は実際のラベル（0 または 1）です。多クラスでは各クラスに対して同様の誤差を求め、平均します。

---

## 2. Python 3.13 での計算と可視化

`ash
python --version        # 例: Python 3.13.0
pip install scikit-learn matplotlib
`

以下のコードでは、乳がん診断データセットにロジスティック回帰を適用し、ブライアスコアと信頼度曲線を描画します。生成された図は static/images/eval/classification/brier-score/reliability_curve.png に保存し、generate_eval_assets.py から再生成できるようにしています。

`python
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import CalibrationDisplay
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, solver="lbfgs"),
)
pipeline.fit(X_train, y_train)
proba = pipeline.predict_proba(X_test)[:, 1]

score = brier_score_loss(y_test, proba)
print(f"Brier Score: {score:.3f}")

fig, ax = plt.subplots(figsize=(5, 5))
CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, ax=ax)
ax.set_title("Reliability Diagram (Breast Cancer Dataset)")
fig.tight_layout()
output_dir = Path("static/images/eval/classification/brier-score")
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "reliability_curve.png", dpi=150)
plt.close(fig)
`

{{< figure src="/images/eval/classification/brier-score/reliability_curve.png" alt="信頼度曲線 (Reliability Diagram)" caption="45 度線から外れているほど確率が過大・過小評価されていると分かる。" >}}

---

## 3. スコアの読み取り方

- 完全に正しい確率予測ではスコアが **0** になります。
- 常に 0.5 を返すモデルは、二値バランスで **0.25** を取るのが上限です。
- 値が小さいほど良く、**確率を外し過ぎているモデルほどスコアが大きくなる**と覚えておきましょう。

---

## 4. キャリブレーション診断との併用

リライアビリティカーブ（信頼度曲線）は、予測確率をビンごとに平均し、実際の陽性率と比較した図です。

- 曲線が 45 度線より上 → 確率を控えめに出している（アンダーコンフィデント）。
- 曲線が 45 度線より下 → 確率を盛り過ぎている（オーバーコンフィデント）。
- CalibrationDisplay.from_predictions は等幅ビンでの可視化をサポートしており、ブライアスコアの変化と合わせて調整の効果を観察できます。

確率校正（Platt scaling や isotonic regression など）を適用した後に再度スコアと図を確認すると、キャリブレーション改善の有無が把握できます。

---

## まとめ

- ブライアスコアは確率予測の二乗誤差で、キャリブレーション評価に適した指標（小さいほど良い）。
- Python 3.13 + scikit-learn では rier_score_loss と信頼度曲線で簡単に診断可能。
- ROC-AUC や Precision/Recall などの閾値ベース評価と併用し、**確率の正確さ**と**ランキング性能**の両面からモデルを分析しよう。
---
