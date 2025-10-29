---
title: "ソフトマックス回帰"
pre: "2.2.2 "
weight: 2
title_suffix: "（多クラス分類をPythonで実行）"
---



<div class="pagetop-box">
  <p><b>ソフトマックス回帰</b>は、ロジスティック回帰を多クラス分類に拡張した手法です。  
  クラス数が2つのときはロジスティック回帰と同じになりますが、3クラス以上でも自然に「確率」を出力できるようになります。</p>
</div>

---

## 1. ソフトマックス関数とは？

入力ベクトル \\(z=(z_1, z_2, \dots, z_K)\\) を「確率分布」に変換する関数です。

$$
\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^K \exp(z_j)} \quad (i=1,\dots,K)
$$

- 出力は **0〜1の範囲**  
- 各クラスの確率に解釈できる  
- 総和は必ず1になる

---

## 2. ソフトマックス回帰モデル

特徴量 \\(x\\) に対して、各クラス \\(k\\) のスコアは

$$
z_k = w_k^\top x + b_k
$$

これをソフトマックス関数で変換し、

$$
P(y=k \mid x) = \frac{\exp(w_k^\top x + b_k)}{\sum_{j=1}^K \exp(w_j^\top x + b_j)}
$$

で確率を得ます。  
最も確率が高いクラスが予測ラベルになります。

---

## 3. Pythonで実行してみる

scikit-learn の `LogisticRegression` に `multi_class="multinomial"` を指定すると、多クラスロジスティック回帰（ソフトマックス回帰）が実行できます。

{{% notice document %}}
- [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [sklearn.datasets.make_classification](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html)
{{% /notice %}}

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# データ生成（3クラス分類）
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_classes=3,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# ソフトマックス回帰
clf = LogisticRegression(multi_class="multinomial", solver="lbfgs")
clf.fit(X, y)

# 可視化のためのメッシュグリッドを作成
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# 描画
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:,0], X[:,1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
plt.title("ソフトマックス回帰による多クラス分類")
plt.show()
```

![softmax block 1](/images/basic/classification/softmax_block01.svg)

---

## 4. 特徴と利点

- 出力が確率分布なので「モデルの確信度」を解釈できる  
- ロジスティック回帰の自然な拡張で、実装が容易  
- 線形モデルなので解釈性が高い（係数で特徴の影響を説明できる）

---

## 5. 実務での使い方

- **自然言語処理**：テキスト分類（例：感情3クラス、ニュース記事カテゴリ分類など）  
- **画像認識（小規模タスク）**：手書き文字（0〜9の10クラス分類など）  
- **多値ラベル分類**：ユーザー行動予測（「購入」「離脱」「継続」など）

---

## まとめ

- ソフトマックス回帰は、ロジスティック回帰を多クラス分類に拡張したもの。  
- 各クラスの確率を計算でき、解釈性が高い。  
- scikit-learn で `multi_class="multinomial"` を指定するだけで簡単に使える。  
- シンプルかつ強力なので、まず試すべき多クラス分類モデルの一つです。

---
