---
title: "RuleFit"
pre: "2.3.4 "
weight: 4
---

{{< katex />}}

{{% notice ref %}}
Friedman, Jerome H and Bogdan E Popescu. “Predictive learning via rule ensembles.” The Annals of Applied Statistics (2008).  
([pdf](https://jerryfriedman.su.domains/ftp/RuleFit.pdf))
{{% /notice %}}

<div class="pagetop-box">
  <p><b>RuleFit</b> は、木モデル（勾配ブースティング木やランダムフォレスト）から抽出した<b>ルール</b>と、元の特徴量の<b>線形項</b>を組み合わせて、最後に <b>L1正則化（Lasso）</b> で疎な（=解釈しやすい）線形モデルとして学習する手法です。</p>
  <p>非線形性や相互作用（if-thenルール）を取り込みつつ、係数で解釈できるのが最大の魅力です。</p>
</div>

---

## 1. RuleFit の考え方（直感と数式）

1. **木からルールを抽出**：  
   勾配ブースティング木（GBDT）やランダムフォレストの各終端葉までの経路は、  
   例：<code>x1 ≤ 3.2 AND 5.0 &lt; x2 ≤ 7.5</code> のような **if-then ルール** になります。  
   それぞれをバイナリ特徴（満たすと1/満たさないと0）に変換します：  
   $$ r_j(x) \in \{0,1\} $$

2. **線形項も合わせる**：  
   元の連続特徴（必要なら標準化・Winsor化）を線形項として加えます：  
   $$ z_k(x) $$

3. **L1正則化回帰（または分類）**：  
   これらをまとめて係数を推定します（回帰の例）：  
   $$
   \hat{y}(x)=\beta_0 + \sum_j \beta_j r_j(x) + \sum_k \gamma_k z_k(x)
   $$  
   L1正則化により、**重要なルールや線形項だけが残る**ため、<b>簡潔で解釈しやすいモデル</b>になります。

> **利点**：非線形・相互作用を取り込みつつ、最終形は「スパースな線形モデル」なので説明が簡単。  
> **注意**：生成できるルールは膨大。正則化（L1）とルール数・木の深さの制御が重要です。

---

## 2. 実験用のデータを取得する（OpenML: house_sales）

openml公開の CC0 データセット **house_sales** を使います（キング郡の住宅データ）。

{{% notice info %}}
OpenML 上の出典表記は曖昧ですが、調査時点では  
[King County Open Data](https://gis-kingcounty.opendata.arcgis.com/datasets/zipcodes-for-king-county-and-surrounding-area-shorelines-zipcode-shore-area/explore?location=47.482924%2C-121.477600%2C8.00&showTable=true) が元データ提供元と推測されます。
{{% /notice %}}

{{% notice document %}}
- [sklearn.datasets.fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
{{% /notice %}}

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

dataset = fetch_openml(data_id=42092, as_frame=True)
X = dataset.data.select_dtypes("number").copy()  # 数値列のみ採用（カテゴリは簡略化のため除外）
y = dataset.target.astype(float)                 # 価格（price）

# 学習/検証に分割（データリーク防止）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## 3. RuleFit を実行する

{{% notice document %}}
- 実装: [Python implementation of the rulefit algorithm (GitHub)](https://github.com/christophM/rulefit)  
- 依存関係により <code>scikit-learn</code> の木モデル（既定は GBDT）が内部で使われます
{{% /notice %}}

> **インストール例**（ローカル環境）：  
> `pip install git+https://github.com/christophM/rulefit.git`

```python
from rulefit import RuleFit
import warnings
warnings.simplefilter("ignore")  # チュートリアルではWarningを抑制（本番では外す）

# ルール数や木側のパラメータは過学習と可読性のトレードオフ
rf = RuleFit(max_rules=200, tree_random_state=27)
rf.fit(X_train.values, y_train.values, feature_names=list(X_train.columns))

# 学習済みでの基本評価
pred_tr = rf.predict(X_train.values)
pred_te = rf.predict(X_test.values)

def report(y_true, y_pred, name):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name:8s}  RMSE={rmse:,.0f}  MAE={mae:,.0f}  R2={r2:.3f}")

report(y_train, pred_tr, "Train")
report(y_test,  pred_te, "Test")
```

> **Tip**：`max_rules`、木の深さ（`max_depth`）、木の本数（`n_estimators`）、学習率（`learning_rate`）などで**ルールの粒度**が変わります。  
> ・粒度を細かくすると当てはまりは上がるが過学習・可読性低下  
> ・粗くすると汎化しやすいが表現力低下  
> → 交差検証でチューニングしましょう。

---

## 4. 作成されたルールを確認する（重要度の高い順）

```python
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values(by="importance", ascending=False)
rules.head(10)
```

- **rule**：ルールそのもの（if-thenの条件列）または線形項（`type=linear`）  
- **coef**：回帰係数（単位は目的変数と同じスケール。解釈に重要）  
- **support**：ルールを満たすサンプルの割合（0〜1）。高すぎると汎用的、低すぎるとデータ依存  
- **importance**：論文由来の指標（係数とサポートを組み合わせた重要度）

> **読み方のコツ**  
> - `type=linear` は元特徴の線形効果。係数の符号・大きさで解釈。  
> - `type=rule` は相互作用や非線形を含むルール。<b>support × coef</b> のバランスで実用性を判断。  
> - 「係数が大きい & supportが適度」なルールは、ビジネス説明に向くことが多いです。

---

## 5. ルールの妥当性を可視化で検証する

### 5.1 単一特徴の線形効果をざっくり確認

例：`sqft_above` の線形項がプラスなら、広いほど価格が上がるはず。

```python
plt.figure(figsize=(6, 5))
plt.scatter(X_train["sqft_above"], y_train, s=10, alpha=0.5)
plt.xlabel("sqft_above")
plt.ylabel("price")
plt.title("sqft_above と価格（散布図）")
plt.grid(alpha=0.3)
plt.show()
```

### 5.2 具体的なルールで分布比較（箱ひげ図）

例：`sqft_living <= 3935.0 & lat <= 47.5315` のような「価格が下がりやすい」ルール。

```python
rule_mask = X["sqft_living"].le(3935.0) & X["lat"].le(47.5315)

applicable_data = np.log(y[rule_mask])      # 対数にして分布の歪みを緩和
not_applicable  = np.log(y[~rule_mask])

plt.figure(figsize=(8, 5))
plt.boxplot([applicable_data, not_applicable],
            labels=["ルール該当", "ルール非該当"])
plt.ylabel("log(price)")
plt.title("ルール該当 vs 非該当 の価格分布（対数）")
plt.grid(alpha=0.3)
plt.show()
```

> **解釈**：ルール該当群の分布が下がっていれば、係数の符号（マイナス）と整合的。

---

## 6. 実務でのコツ（前処理・チューニング・解釈）

- **スケーリング／ウィンズor外れ値処理**：線形項の安定性と解釈性が向上。  
- **カテゴリ変数**：One-Hot化の前にカテゴリを整理（希少カテゴリはまとめる等）。  
- **ターゲット変換**：価格のように歪度が強い場合、`log(y)` で安定。  
- **ルール数の制御**：`max_rules`／木の深さ・本数を控えめに → 可読性と汎化のバランス。  
- **交差検証**：ハイパラは CV チューニング（`max_rules`, `tree_size`, `learning_rate`, `n_estimators` など）。  
- **リーク対策**：train/test split したあとに学習（前処理も学習側で fit → 変換）。  
- **説明レポート**：  
  - 重要なルール Top N を日本語に直して添える（例：「面積 3400 以上 かつ 緯度が北側 → 価格が上がりやすい」）  
  - **support × business sense** で現実妥当性をチェック。  
  - 類似ルールが並ぶときは統合・簡素化を検討。

---

## 7. まとめ

- RuleFit は **木由来のルール** と **線形項** を **L1 正則化**で統合した、<b>解釈性と非線形性のバランスが取れた手法</b>。  
- ルールは if-then として説明しやすく、線形係数は量的な影響を示せる。  
- ルール数や木の深さは過学習と可読性のトレードオフ。**CV でチューニング**しよう。  
- 可視化（散布図・箱ひげ図等）でルールの妥当性を確認すると、納得感のあるモデル運用につながる。

---
