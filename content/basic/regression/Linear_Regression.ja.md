---
title: "線形回帰（最小二乗法）"
pre: "2.1.2 "
weight: 2
title_suffix: "理論と実装"
---

{{< katex />}}

{{% youtube "KKuAxQbuJpk" %}}

<div class="pagetop-box">
<p>
<b>線形回帰（Linear Regression）</b>は、最小二乗法を用いてデータに直線を当てはめる基本的な手法です。  
観測されたデータ $(x_i, y_i)$ の関係を「できるだけよく説明する直線」を求めることを目的としています。  
</p>
</div>

---

## 1. 線形回帰の数式モデル

### 単回帰モデル
入力 $x$ と出力 $y$ の関係を直線で表すと、次のようになります。

$$
y = w x + b
$$

- $w$: 傾き（係数）  
- $b$: 切片（バイアス項）  

これにより「$x$ が1増えると $y$ が $w$ 増える」という解釈ができます。

---

## 2. 損失関数（残差平方和）

直線がどれだけデータにフィットしているかを測るために、**残差**（観測値と予測値の差）を考えます。

$$
\epsilon_i = y_i - (w x_i + b)
$$

この残差を二乗して全データで合計したものを **損失関数**（残差平方和）とします。

$$
L(w, b) = \sum_{i=1}^n \big(y_i - (w x_i + b)\big)^2
$$

最小二乗法とは、この $L(w, b)$ を最小化する $w, b$ を求める方法です。

---

## 3. 解析解（解の導出）

損失関数を $w, b$ で偏微分しゼロと置くと、次の解析解が得られます。

$$
w = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}, \quad
b = \bar{y} - w \bar{x}
$$

ここで、

- $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ （$x$ の平均）  
- $\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i$ （$y$ の平均）  

です。

つまり、**回帰直線は平均値 $(\bar{x}, \bar{y})$ を通る**ことがわかります。

---

## 4. Pythonによる実装例

scikit-learn を使うと、線形回帰を簡単に実装できます。

```python
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# データ作成
n_samples = 100
X = np.linspace(-5, 5, n_samples)[:, np.newaxis]
epsolon = np.random.normal(scale=2, size=n_samples)
y = 2 * X.ravel() + 1 + epsolon  # 真の関係: y = 2x + 1 + ノイズ

# モデル学習
lin_r = make_pipeline(StandardScaler(with_mean=False), LinearRegression()).fit(X, y)
y_pred = lin_r.predict(X)

# 可視化
plt.figure(figsize=(10, 5))
plt.scatter(X, y, marker="x", label="観測データ", c="orange")
plt.plot(X, y_pred, label="回帰直線（最小二乗法）")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.show()
```

---

## 5. 学習結果の解釈

- **傾き $w$**  
  - 入力 $x$ が1増えたとき、$y$ がどの程度変化するかを表す。  
- **切片 $b$**  
  - $x=0$ のときの $y$ の予測値。  

この $w, b$ を求めることで「データの傾向」を数式で表すことができます。

---

## 6. まとめ

- 線形回帰は「データを直線で説明する」基本的な手法。  
- 最小二乗法で「観測値と予測値のズレ（二乗和）」を最小にする。  
- 解析的に解を導けるだけでなく、ライブラリを使えば簡単に実装できる。  
- 今後は多変量回帰や正則化（リッジ回帰・ラッソ回帰）などに拡張可能。

---

