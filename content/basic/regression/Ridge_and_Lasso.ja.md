---
title: "リッジ回帰・ラッソ回帰"
pre: "2.1.2 "
weight: 2
title_suffix: "の仕組みの説明"
---

{{% youtube "rhGYOBrxPXA" %}}

<div class="pagetop-box">
    <p><b>リッジ回帰・ラッソ回帰</b>とはモデルの係数にペナルティを付けることでモデルが過学習しないように工夫した線形回帰モデルです。これらの手法のように、過学習を防いで汎化性を高めようとする技術を正則化と呼びます。このページでは、pythonを使って最小二乗法・リッジ回帰・ラッソ回帰を実行して、それらの係数を比較します。</p>
</div>

## 必要なモジュールをインポート

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

## 実験用の回帰データを作成する
{{% notice document %}}
[sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)
{{% /notice %}}

[sklearn.datasets.make_regression](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html)を用いて人工的に回帰データを作成します。予測に必要な特徴量は2つ(`n_informative` = 2)と指定します。それ以外は予測の役に立たない冗長な特徴量です。


```python
n_features = 5
n_informative = 2
X, y = make_regression(
    n_samples=500,
    n_features=n_features,
    n_informative=n_informative,
    noise=0.5,
    random_state=777,
)
X = np.concatenate([X, np.log(X + 100)], 1)  # 冗長な特徴量を追加する
y_mean = y.mean(keepdims=True)
y_std = np.std(y, keepdims=True)
y = (y - y_mean) / y_std
```

## 最小二乗法・リッジ回帰・ラッソ回帰モデルを比較する
それぞれのモデルについて同じデータをつかって訓練して違いを見てみます。
本来は正則化項の係数alphaをいろいろ変えて試すべきですが、ここでは固定しています。

{{% notice document %}}
- [sklearn.pipeline.make_pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)
- [sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)
- [sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)
{{% /notice %}}


```python
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
rid_r = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha=2)).fit(X, y)
las_r = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha=0.1)).fit(X, y)
```

## 各モデルの係数の値を比較する

各係数の絶対値をプロットします。線形回帰は係数がほとんど０にならないことがグラフから見て取れます。 また、Lasso回帰は予測に必要な特徴量以外で係数＝０になっていることが確認できます。

有用な特徴は２個（`n_informative = 2`）だったので、線形回帰は有用でない特徴でも無条件に係数がついていることがわかります。Lasso・Ridge回帰は２個まで絞り込むことはできませんでしたが、それでも多くの不要な特徴で係数が０になるという結果になりました。

```python
feat_index = np.array([i for i in range(X.shape[1])])

plt.figure(figsize=(12, 4))
plt.bar(
    feat_index - 0.2,
    np.abs(lin_r.steps[1][1].coef_),
    width=0.2,
    label="LinearRegression",
)
plt.bar(feat_index, np.abs(rid_r.steps[1][1].coef_), width=0.2, label="RidgeRegression")
plt.bar(
    feat_index + 0.2,
    np.abs(las_r.steps[1][1].coef_),
    width=0.2,
    label="LassoRegression",
)

plt.xlabel(r"$\beta$のインデックス")
plt.ylabel(r"$|\beta|$")
plt.grid()
plt.legend()
```

    
![png](/images/basic/regression/02_Ridge_and_Lasso_files/02_Ridge_and_Lasso_10_1.png)
    

