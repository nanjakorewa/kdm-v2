---
title: "ロジスティック回帰"
pre: "2.2.1 "
weight: 1
title_suffix: "をPythonで実行する"
---



<div class="pagetop-box">
  <p><b>ロジスティック回帰</b>は、データを<b>2つのクラス</b>に分類するための代表的な手法です。  
  線形回帰は連続値を出力しますが、ロジスティック回帰はその出力を<b>0〜1の確率</b>に変換することで分類に利用できます。</p>
</div>

---

## 1. ロジスティック回帰の基本アイデア

線形モデルは
$$
z = w^\top x + b
$$
の形をしています。  
これをそのまま使うと値が \\((-\infty, +\infty)\\) の範囲をとるため、「確率」とはみなせません。

そこで **シグモイド関数** を使って \\(z\\) を確率に変換します。

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- \\(z \to -\infty\\) のとき \\(\sigma(z) \to 0\\)  
- \\(z \to +\infty\\) のとき \\(\sigma(z) \to 1\\)  
- \\(z=0\\) のとき \\(\sigma(z)=0.5\\)

この「0.5を境に0か1を判定する」というのがロジスティック回帰の仕組みです。

---

## 2. ロジット関数（確率を直線に変換）

シグモイドの逆関数が **ロジット関数** です。

$$
\text{logit}(p) = \log\frac{p}{1-p}
$$

- \\(p \to 0\\) のとき \\(\text{logit}(p) \to -\infty\\)  
- \\(p \to 1\\) のとき \\(\text{logit}(p) \to +\infty\\)  

つまり「確率を線形の世界に引き戻す」役割を持ちます。

### ロジット関数を可視化

```python
import matplotlib.pyplot as plt
import numpy as np

p = np.linspace(0.01, 0.999, 200)
y = np.log(p / (1 - p))

plt.figure(figsize=(6, 4))
plt.plot(p, y)
plt.axhline(y=0, color="k", linewidth=0.8)
plt.xlabel("確率 p")
plt.ylabel("logit(p)")
plt.title("ロジット関数の形")
plt.grid(True)
plt.show()
```

![logistic-regression block 1](/images/basic/classification/logistic-regression_block01.svg)

---

## 3. 数式で表すロジスティック回帰モデル

入力 \\(x\\) に対してクラス1に属する確率は

$$
P(y=1 \mid x) = \sigma(w^\top x + b) = \frac{1}{1+e^{-(w^\top x + b)}}
$$

クラス0の確率はその補数で

$$
P(y=0 \mid x) = 1 - P(y=1 \mid x)
$$

と表されます。

> 予測が0.5を超えたらクラス1、それ以下ならクラス0と判定します。

---

## 4. Pythonで実行してみる

scikit-learnの `LogisticRegression` を使えば、すぐに学習と予測が可能です。

{{% notice document %}}
- [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
{{% /notice %}}

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 2次元の分類データを生成
X, Y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=1,
    random_state=2,
    n_clusters_per_class=1,
)

# モデル学習
clf = LogisticRegression()
clf.fit(X, Y)

# 決定境界を求める
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
slope = -w1 / w2
intercept = -b / w2

xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
xd = np.array([xmin, xmax])
yd = slope * xd + intercept

# 可視化
plt.figure(figsize=(8, 8))
plt.plot(xd, yd, "k-", lw=1, label="決定境界")
plt.scatter(*X[Y == 0].T, marker="o", label="クラス0")
plt.scatter(*X[Y == 1].T, marker="x", label="クラス1")
plt.legend()
plt.title("ロジスティック回帰による分類")
plt.show()
```

![logistic-regression block 2](/images/basic/classification/logistic-regression_block02.svg)

---

## 5. ロジスティック回帰の特徴と注意点

- 出力が「確率」なので解釈しやすい  
- 決定境界は **直線（または超平面）**  
- 学習は「最尤法（尤度最大化）」で行う  
- デフォルトで **L2正則化（リッジ）** が効いており、過学習を防ぐ  

---

## 6. 応用・よくある使い方

- **医療分野**：「病気を持つ確率」を推定  
- **金融分野**：「融資を返済できる確率」を予測  
- **広告・マーケティング**：「クリック率予測」など  

### 拡張
- 多クラス分類：`multi_class="multinomial"` を指定  
- 正則化：L1（ラッソ）で特徴選択も可能  
- 係数解釈：係数の符号・大きさで「どの特徴が分類に寄与しているか」がわかる  

---

## まとめ

- ロジスティック回帰は「線形モデル＋シグモイド」で分類を実現  
- 出力は0〜1の確率で解釈可能  
- scikit-learnで簡単に実行＆可視化できる  
- 多クラス分類や正則化拡張にも対応でき、実務での利用も広い  

---
