---
title: "AdaBoost（回帰）"
pre: "2.4.4 "
weight: 4
title_suffix: "の直感・数式・実装"
---

{{< katex />}}
{{% youtube "1K-h4YzrnsY" %}}

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
```

# AdaBoost（回帰）

<div class="pagetop-box">
  <p><b>AdaBoost（Adaptive Boosting）</b>は、弱学習器（ふつうは浅い決定木）を<b>順番に</b>学習していき、前の器が苦手だった点（誤差の大きい点）に<b>重みを置き直す</b>ことで、全体として強い回帰モデルを作る手法です。</p>
</div>

---

## 1. まずは直感（何をしている？）

1. 最初は全データを<b>同じ重み</b>で学習  
2. 間違い（誤差）の大きかった点の<b>重みを増やす</b>  
3. 重み付きデータで次の弱学習器を学習  
4. これを繰り返し、最後に<b>多数の弱学習器の出力を重み付きで平均</b>して予測を出す

> 要するに「むずかしい点を後の学習で重点的に扱う」→ 徐々に誤差を縮めていくイメージです。

---

## 2. 数式でみる AdaBoost.R2（回帰）

サンプル \((x_i, y_i)\)（\(i=1,\dots,N\)）に重み \(w_i^{(m)}\) を持たせ、反復 \(m=1,\dots,M\) で弱学習器 \(h^{(m)}(x)\) を学習します。

### 2.1 誤差の定義（相対誤差）
AdaBoost.R2 では予測誤差を 0〜1 に正規化して使うのが特徴です（以下は代表的な定義の一例）：

$$
e_i^{(m)} \;=\; \frac{\bigl|\,y_i - h^{(m)}(x_i)\,\bigr|}{E_{\max}^{(m)}}
\quad,\quad
E_{\max}^{(m)} \;=\; \max_j \bigl|\,y_j - h^{(m)}(x_j)\,\bigr|
$$

（`loss="linear"` の場合。`square` なら \((e_i^{(m)})^2\)、`exponential` なら \(\exp(e_i^{(m)})-1\) を使うイメージ。）

### 2.2 学習器の誤差率と重み
重み付き誤差率：
$$
\varepsilon^{(m)} \;=\; \sum_{i=1}^N w_i^{(m)}\, e_i^{(m)}
\quad\text{（通常 } 0 \le \varepsilon^{(m)} < 0.5 \text{ を目標）}
$$

弱学習器の重み（信頼度）：
$$
\beta^{(m)} \;=\; \frac{\varepsilon^{(m)}}{1-\varepsilon^{(m)}}
\quad \Rightarrow \quad
\alpha^{(m)} \;=\; \log\frac{1}{\beta^{(m)}} \;=\; \log\frac{1-\varepsilon^{(m)}}{\varepsilon^{(m)}}
$$

> \(\varepsilon^{(m)}\) が小さい（= よく当たる）ほど \(\alpha^{(m)}\) は大きく、寄与が増えます。

### 2.3 サンプル重みの更新
$$
w_i^{(m+1)} \;\propto\; w_i^{(m)} \, \bigl(\beta^{(m)}\bigr)^{\,1 - e_i^{(m)}}
\quad\text{（最後に正規化して } \sum_i w_i^{(m+1)}=1\text{）}
$$

誤差 \(e_i^{(m)}\) が大きいほど指数の \(1-e_i^{(m)}\) が小さくなり、\(\beta^{(m)} < 1\) なら重みは<b>相対的に増える</b>直感です。

### 2.4 最終予測
連続値予測は弱学習器の重み付き結合で出します（scikit-learn 実装は <i>weighted median</i> 相当の集約を使うことがあります）：

$$
\hat{y}(x) \;=\; \sum_{m=1}^M \alpha^{(m)} \, h^{(m)}(x) \;\bigg/ \sum_{m=1}^M \alpha^{(m)}
$$

---

## 3. 実装：合成データで学習してみる

```python
# 合成データ（非線形 + 緩やかなトレンド）
X = np.linspace(-10, 10, 500)[:, np.newaxis]
y = (np.sin(X).ravel() + np.cos(4 * X).ravel()) * 10 + 10 + np.linspace(-2, 2, 500)

# AdaBoost 回帰モデル（浅い木 × たくさん）
reg = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=5, random_state=0),
    n_estimators=100,
    learning_rate=0.8,
    loss="linear",     # "linear" / "square" / "exponential"
    random_state=100,
)
reg.fit(X, y)
y_pred = reg.predict(X)

# 可視化
plt.figure(figsize=(10, 5))
plt.scatter(X, y, c="k", marker="x", s=15, label="訓練データ")
plt.plot(X, y_pred, c="r", lw=1.5, label="モデル予測")
plt.xlabel("x"); plt.ylabel("y"); plt.title("AdaBoost 回帰：学習結果")
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

![png](/images/basic/ensemble/Adaboost_Regression_files/Adaboost_Regression_6_0.png)

**読み方**  
- 弱学習器は「段差状」の予測をしますが、多数を重ねると<b>なめらかな曲線に近づく</b>傾向があります。  
- `max_depth` を深く／`n_estimators` を多くすると表現力は上がる一方、過学習しやすくなります。

---

## 4. 損失関数で外れ値への感度を調整

{{% notice document %}}
- [AdaBoostRegressor — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
{{% /notice %}}

- `loss="linear"`（既定）：誤差に比例して重み増→ **バランス型**  
  \[
  L(y,\hat{y})=|y-\hat{y}|
  \]
- `loss="square"`：二乗で強調→ **外れ値に敏感**（当てるのが難しい点に引っ張られやすい）  
  \[
  L(y,\hat{y})=(y-\hat{y})^2
  \]
- `loss="exponential"`：指数でさらに強調→ **かなり敏感**（ノイズが多いと不安定になりがち）  
  \[
  L(y,\hat{y})=\exp(|y-\hat{y}|)-1
  \]

> ノイズが多い実データでは、まずは `linear` を試し、必要に応じて変更するのが無難です。

---

## 5. 「重み付け」の直感を掴む（簡単な可視化）

> scikit-learn では内部の重みベクトルはそのまま見えません。  
> ここでは「誤差の大きい点が次の学習でより重要になる」直感を簡易に可視化するアイデアを示します。

```python
# 例：学習の前後で誤差が大きい点を色で目立たせる（直感用）
y_pred_once = DecisionTreeRegressor(max_depth=5, random_state=0).fit(X, y).predict(X)
err = np.abs(y - y_pred_once)
thr = np.percentile(err, 80)  # 上位20%を「むずかしい点」とする

plt.figure(figsize=(10, 4))
plt.scatter(X[err<=thr], y[err<=thr], s=15, c="gray", label="誤差が小さい点")
plt.scatter(X[err>thr],  y[err>thr],  s=25, c="crimson", label="誤差が大きい点（重み↑）")
plt.xlabel("x"); plt.ylabel("y"); plt.title("次の学習で重みが増えやすい点の直感")
plt.legend(); plt.grid(alpha=0.3); plt.show()
```

**解釈**  
- 赤い点（誤差が大きい点）ほど次の器で重点的に学習され、順次改善が試みられます。  
- 実データでは外れ値やラベルノイズが混ざるため、損失の選択や学習率で安定性を担保しましょう。

---

## 6. ハイパーパラメータの考え方

- **`n_estimators`**：弱学習器の数。多いほど表現力↑だが過学習・計算コスト↑  
- **`learning_rate`**：各ステップの寄与。小さいと一歩ずつ（安定）・大きいと速いが過学習しやすい  
- **`base_estimator`**（`estimator` 名称のバージョンもあり）：  
  - ふつうは `DecisionTreeRegressor(max_depth=3〜6)` 程度の浅い木  
  - 深すぎると各器が「強くなり過ぎ」、Boosting の積み重ね効果が薄れることがある  
- **`loss`**：外れ値感度（上の節）

> 目安：`learning_rate × n_estimators` は概ねトレードオフ。小さめの学習率 + やや多めの器で始め、検証データで調整。

---

## 7. 実務のコツ

- **評価指標**：RMSE と MAE を併用（外れ値が多いときは MAE も重視）  
- **外れ値/ノイズ対策**：`loss="linear"` から開始、必要ならロバスト前処理（Winsorize, クリッピング）  
- **スケーリング**：木ベースなので必須ではないが、極端なスケール差は特徴選択に影響しうる  
- **早期停止（擬似）**：`n_estimators` を段階的に増やし、検証スコアが頭打ちなら打ち切る  
- **再現性**：`random_state` を固定  
- **比較**：勾配ブースティング（`HistGradientBoostingRegressor` など）と性能・速度・安定性を比較する

---

## 8. まとめ

- AdaBoost（回帰）は、誤差の大きい点の<b>重みを増やしながら</b>弱学習器を重ねるブースティング手法  
- AdaBoost.R2 では誤差を正規化して使い、外れ値への感度は `loss` で調整可能  
- `learning_rate × n_estimators × base_estimator` のバランス調整がカギ  
- 実務では指標の複数併用・外れ値対策・早期停止・他手法との比較が有効

---
