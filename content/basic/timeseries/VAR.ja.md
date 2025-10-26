---
title: "ベクトル自己回帰（VAR）"
pre: "2.8.7 "
weight: 7
title_suffix: "多変量時系列の相互依存をモデル化"
---

{{< lead >}}
VAR（Vector Autoregression）は、複数系列が互いに影響し合う多変量時系列を扱うための拡張自己回帰モデルです。景気指標やセンサーネットワークなど、系列間の相互作用を明示的に捉えられます。
{{< /lead >}}

---

## 1. モデル概要

2 変量の VAR(1) モデルを例にとると、

$$
\begin{bmatrix}
y_{1,t} \\
y_{2,t}
\end{bmatrix}
=
\mathbf{c} +
\mathbf{A}_1
\begin{bmatrix}
y_{1,t-1} \\
y_{2,t-1}
\end{bmatrix}
 +
\begin{bmatrix}
\varepsilon_{1,t} \\
\varepsilon_{2,t}
\end{bmatrix}
$$

ここで \\(\mathbf{A}_1\\) は各系列のラグ項に対する係数行列です。ラグ次数を増やした VAR(p) モデルも同様に定義できます。

---

## 2. Python での実装

```python
import pandas as pd
from statsmodels.tsa.api import VAR

data = pd.DataFrame({"y1": series1, "y2": series2})
model = VAR(data)
results = model.fit(maxlags=4, ic="aic")

print(results.summary())
forecast = results.forecast(data.values[-results.k_ar:], steps=12)
```

`fit` の `ic` に `"aic"` や `"bic"` を指定すると、情報量基準で最適ラグを選択できます。

---

## 3. 前処理と仮定

- **定常性**：一度差分を取り、ADF 検定で単位根を確認。非定常なら差分 VAR（DVAR）や VECM へ。
- **共分散の安定性**：自己相関行列が安定していること。必要に応じてホワイトノイズ化を確認。
- **スケーリング**：系列のスケールが大きく異なる場合は標準化すると安定。

---

## 4. 応用例

- **マクロ経済分析**：GDP、失業率、金利などの相互作用を分析。
- **エネルギー需要**：地域別需要や気象要因を同時に扱う。
- **多センサー信号**：工場の複数センサー値から故障兆候を抽出。

---

## 5. 拡張機能

- **グレンジャー因果性**：`results.test_causality` で系列間の因果関係を検定。
- **インパルス応答分析**：ショックが他系列に及ぼす影響を可視化。
- **分散分解**：予測誤差分散に各系列がどれだけ寄与したかを評価。

---

## まとめ

- VAR は多変量時系列の相互依存をモデル化する基本手法で、ラグ行列を用いて予測を行う。
- `statsmodels` の VAR 実装で簡単に適用でき、情報量基準でラグ選択や因果性検定も行える。
- 定常性を満たすよう前処理を行い、インパルス応答などの解析と組み合わせて洞察を深めよう。

---
