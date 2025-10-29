---
title: "サポートベクター回帰（SVR）"
pre: "2.1.10 "
weight: 10
title_suffix: "ε不感帯でロバストに予測する"
---

{{% summary %}}
- サポートベクター回帰（SVR）は SVM を回帰に拡張し、ε 不感帯により小さな誤差をゼロとみなして外れ値の影響を抑える。
- カーネル法を利用して非線形な関係も柔軟に学習できるため、複雑なパターンを少数のサポートベクターで表現できる。
- ハイパーパラメータ `C`・`epsilon`・`gamma` の調整で汎化性能とモデルの滑らかさを制御する。
- 特徴量のスケーリングは必須であり、パイプライン化して前処理と学習を一体化すると安定する。
{{% /summary %}}

## 直感
SVR は「一定幅の誤差なら許容する」という考え方で、得られた近似関数とデータ点の間に ε 幅のチューブを敷き、チューブの外に出た点にのみペナルティを課します。サポートベクターと呼ばれるチューブの境界に触れる点だけがモデルに影響し、その他の点は誤差に含まれても無視されます。

## 具体的な数式
最適化問題は以下のようになります。

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^{n} (\xi_i + \xi_i^*)
$$

制約条件は

$$
\begin{aligned}
y_i - (\mathbf{w}^\top \phi(\mathbf{x}_i) + b) &\le \epsilon + \xi_i, \\
(\mathbf{w}^\top \phi(\mathbf{x}_i) + b) - y_i &\le \epsilon + {\xi}_i^*, \\
\xi_i, \xi_i^* &\ge 0
\end{aligned}
$$

で、\(\phi\) はカーネルによる特徴空間への写像です。双対問題を解くことでサポートベクターと係数が得られます。

## Pythonを用いた実験や説明
`StandardScaler` と組み合わせた SVR の基本的な使い方です。

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
)

svr.fit(X_train, y_train)
pred = svr.predict(X_test)
```

### 実行結果の読み方
- パイプライン化すると訓練データの平均と分散でスケーリングされ、テストデータにも同じ変換が適用される。
- `pred` はテストデータに対する予測値で、`epsilon` や `C` を調整すると過学習・過小適合のバランスが変わる。
- RBF カーネルの `gamma` を大きくすると局所的なフィットが強まり、小さくすると滑らかな関数になる。

## 参考文献
{{% references %}}
<li>Smola, A. J., &amp; Schölkopf, B. (2004). A Tutorial on Support Vector Regression. <i>Statistics and Computing</i>, 14(3), 199–222.</li>
<li>Vapnik, V. (1995). <i>The Nature of Statistical Learning Theory</i>. Springer.</li>
{{% /references %}}
