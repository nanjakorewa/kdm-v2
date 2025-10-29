---
title: "決定木のパラメータ"
pre: "2.3.3 "
weight: 3
title_suffix: "複雑さを制御して汎化させる"
---

{{% youtube "AOEtom_l3Wk" %}}

{{% summary %}}
- 決定木は分割を深くすれば訓練データへ高い当てはまりを得られるが、同時に過学習リスクが高まる。
- `max_depth` や `min_samples_leaf` などのハイパーパラメータは木の複雑さを抑え、汎化性能を保つためのブレーキとして働く。
- コスト複雑度剪定（`ccp_alpha`）は訓練誤差と木のサイズのトレードオフを明示的に最適化するアプローチである。
- ハイパーパラメータを調整したときの決定境界や予測曲面を可視化すると、モデル挙動とパラメータの関係を直感的に理解できる。
{{% /summary %}}

## 直感
決定木は「もし◯◯なら左へ、そうでなければ右へ」という条件を繰り返し、最終的に葉ノードで値を返します。分割を制限しなければ訓練データをほぼ完全に近似できますが、その分だけデータのわずかな揺らぎにも敏感になります。  
`max_depth` や `min_samples_leaf` などのパラメータを調整することで、木の成長を適度に抑え、過学習を回避することが重要です。

## 具体的な数式
親ノード \\(P\\) を左右の子ノード \\(L, R\\) に分割したときの不純度減少量は

$$
\Delta I = I(P) - \frac{|L|}{|P|} I(L) - \frac{|R|}{|P|} I(R)
$$

で表されます。回帰木の場合、\\(I(t)\\) を平均二乗誤差や平均絶対誤差として定義し、\\(\Delta I\\) が最大になる特徴量としきい値を選びます。  
剪定では、木 \\(T\\) のコスト複雑度

$$
R_\alpha(T) = R(T) + \alpha |T|
$$

を最小化する部分木を探します。ここで \\(R(T)\\) は訓練誤差、\\(|T|\\) は葉ノード数、\\(\alpha \ge 0\\) は複雑さに対するペナルティを示します。

## Pythonを用いた実験や説明
以下のスニペットは 2 次元の回帰データに対して異なるパラメータ設定を試すシンプルな例です。`max_depth` や `ccp_alpha`、`min_samples_leaf` を切り替えると、同じデータでも予測曲面が大きく変化することが確認できます。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score

X, y = make_regression(
    n_samples=500,
    n_features=2,
    noise=0.2,
    random_state=42,
)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

def evaluate(params):
    model = DecisionTreeRegressor(random_state=0, **params).fit(Xtr, ytr)
    r2_train = r2_score(ytr, model.predict(Xtr))
    r2_test = r2_score(yte, model.predict(Xte))
    print(f"{params}: train R2={r2_train:.3f}, test R2={r2_test:.3f}")

evaluate({"max_depth": 3})
evaluate({"max_depth": 10})
evaluate({"max_depth": 5, "min_samples_leaf": 5})
evaluate({"max_depth": 5, "ccp_alpha": 0.01})
```

### 可視化ギャラリー
決定木のパラメータを変えたときの予測面や分割の違いを、以前のノートブックで出力していた画像として再掲します。

![png](/images/basic/tree/Parameter_files/Parameter_5_0.png)
![png](/images/basic/tree/Parameter_files/Parameter_7_0.png)
![png](/images/basic/tree/Parameter_files/Parameter_7_1.png)
![png](/images/basic/tree/Parameter_files/Parameter_9_0.png)
![png](/images/basic/tree/Parameter_files/Parameter_11_0.png)
![png](/images/basic/tree/Parameter_files/Parameter_13_0.png)
![png](/images/basic/tree/Parameter_files/Parameter_15_0.png)
![png](/images/basic/tree/Parameter_files/Parameter_17_0.png)
![png](/images/basic/tree/Parameter_files/Parameter_20_0.png)
![png](/images/basic/tree/Parameter_files/Parameter_21_0.png)

## 参考文献
{{% references %}}
<li>Breiman, L., Friedman, J. H., Olshen, R. A., &amp; Stone, C. J. (1984). <i>Classification and Regression Trees</i>. Wadsworth.</li>
<li>Breiman, L., &amp; Friedman, J. H. (1991). <i>Cost-Complexity Pruning</i>. In <i>Classification and Regression Trees</i>. Chapman &amp; Hall.</li>
<li>scikit-learn developers. (2024). <i>Decision Trees</i>. https://scikit-learn.org/stable/modules/tree.html</li>
{{% /references %}}
