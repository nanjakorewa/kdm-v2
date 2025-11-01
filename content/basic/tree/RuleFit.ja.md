---
title: "RuleFit | 機械学習の基礎解説"
linkTitle: "RuleFit"
seo_title: "RuleFit | 機械学習の基礎解説"
pre: "2.3.4 "
weight: 4
---

{{% summary %}}
- RuleFit は決定木系モデルから抽出した if-then ルールと元の特徴量の線形項を組み合わせ、L1 正則化で疎な線形モデルを得る手法。
- ルールは非線形や相互作用を表現し、線形項は全体的な傾向を補うことで、解釈性と表現力の両立を狙う。
- 生成されるルール数は多くなりがちなので、`max_rules` や木の深さを制御し、交差検証で正則化強度を調整する。
- 重要なルールを抽出して可視化すると、ビジネスサイドへの説明資料として活用しやすい。
{{% /summary %}}

## 直感
ランダムフォレストや勾配ブースティング木を学習すると、多数の if-then ルール（葉ノードに至るまでの条件）が得られます。RuleFit はこれらのルールを二値特徴として取り出し、元の連続特徴と合わせて線形回帰（あるいは分類）モデルを学習します。L1 正則化（Lasso）により重要でないルールの係数が 0 になり、少数のルールだけで予測と説明が可能になります。

## 具体的な数式
抽出したルール群を \\(r_j(x) \in \{0, 1\}\\)、元の連続特徴を前処理したものを \\(z_k(x)\\) とすると、回帰問題では

$$
\hat{y}(x) = \beta_0 + \sum_j \beta_j r_j(x) + \sum_k \gamma_k z_k(x)
$$

の形でモデル化します。学習は L1 正則化付き回帰：

$$
\min_{\beta, \gamma} \left\{
\frac{1}{2n} \sum_{i=1}^{n} \bigl(y_i - \hat{y}(x_i)\bigr)^2
 + \lambda \left( \sum_j |\beta_j| + \sum_k |\gamma_k| \right)
\right\}
$$

を解くことで行われます。ルールは木モデルの葉ノードに対応し、`max_rules` や木の深さで数を調整します。

## Pythonを用いた実験や説明
OpenML の住宅価格データを用いて RuleFit を適用し、ルールの重要度を確認します。

```python
import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

```python
dataset = fetch_openml(data_id=42092, as_frame=True)
X = dataset.data.select_dtypes("number").copy()  # 数値列のみ採用
y = dataset.target.astype(float)                 # 価格列

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

```python
from rulefit import RuleFit
import warnings
warnings.simplefilter("ignore")  # チュートリアルでは Warning を抑制

rf = RuleFit(max_rules=200, tree_random_state=27)
rf.fit(X_train.values, y_train.values, feature_names=list(X_train.columns))

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

```python
rules = rf.get_rules()
rules = rules[rules.coef != 0].sort_values(by="importance", ascending=False)
rules.head(10)
```

```python
plt.figure(figsize=(6, 5))
plt.scatter(X_train["sqft_above"], y_train, s=10, alpha=0.5)
plt.xlabel("sqft_above")
plt.ylabel("price")
plt.title("sqft_above と価格の散布図")
plt.grid(alpha=0.3)
plt.show()
```

![rulefit block 4](/images/basic/tree/rulefit_block04.svg)

```python
rule_mask = X["sqft_living"].le(3935.0) & X["lat"].le(47.5315)

applicable_data = np.log(y[rule_mask])
not_applicable  = np.log(y[~rule_mask])

plt.figure(figsize=(8, 5))
plt.boxplot([applicable_data, not_applicable],
            labels=["ルール該当", "ルール非該当"])
plt.ylabel("log(price)")
plt.title("ルール該当 vs 非該当の価格分布（対数）")
plt.grid(alpha=0.3)
plt.show()
```

![rulefit block 5](/images/basic/tree/rulefit_block05.svg)

## 参考文献
{{% references %}}
<li>Friedman, J. H., &amp; Popescu, B. E. (2008). Predictive Learning via Rule Ensembles. <i>The Annals of Applied Statistics</i>, 2(3), 916–954.</li>
<li>Christoph Molnar. (2020). <i>Interpretable Machine Learning</i>. https://christophm.github.io/interpretable-ml-book/rulefit.html</li>
{{% /references %}}
