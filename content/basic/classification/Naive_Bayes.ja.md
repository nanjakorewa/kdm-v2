---
title: "ナイーブベイズ"
pre: "2.2.6 "
weight: 6
title_suffix: "条件付き独立性で高速に推論する"
---

{{% summary %}}
- ナイーブベイズは特徴量が条件付き独立であると仮定し、ベイズの定理で事前確率と尤度を組み合わせて分類する。
- 学習・推論ともに高速で、テキスト分類やスパム判定など高次元で疎なデータの強力なベースラインになる。
- ラプラス平滑化や TF-IDF と組み合わせると未知語や頻度差にうまく対応できる。
- 独立性の仮定が強すぎる場合は、特徴選択や他モデルとのアンサンブルを検討する。
{{% /summary %}}

## 直感
ベイズの定理は「事前確率 × 尤度 ∝ 事後確率」という関係を与えます。特徴量が条件付き独立であれば、尤度は各特徴量ごとの確率の積として簡単に表せます。ナイーブベイズはこの近似を利用し、少ない学習データでも安定した推定を行います。

## 具体的な数式
クラス \(y\) と特徴ベクトル \(\mathbf{x} = (x_1, \ldots, x_d)\) について

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^{d} P(x_j \mid y)
$$

と近似します。テキスト分類では頻度を扱う多項型、出現/非出現を扱うベルヌーイ型、連続値にはガウス型など、尤度の形は課題に合わせて選択できます。

## Pythonを用いた実験や説明
以下はニュース記事を対象に TF-IDF 特徴量と多項ナイーブベイズを組み合わせた例です。高次元でも高速に学習でき、分類レポートで性能を確認できます。

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

categories = ["rec.autos", "sci.space", "talk.politics.guns"]
train = fetch_20newsgroups(subset="train", categories=categories, remove=("headers", "footers", "quotes"))
test = fetch_20newsgroups(subset="test", categories=categories, remove=("headers", "footers", "quotes"))

model = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB(alpha=0.5))
model.fit(train.data, train.target)
pred = model.predict(test.data)

print(classification_report(test.target, pred, target_names=categories))
```

## 参考文献
{{% references %}}
<li>Manning, C. D., Raghavan, P., &amp; Schütze, H. (2008). <i>Introduction to Information Retrieval</i>. Cambridge University Press.</li>
<li>Murphy, K. P. (2012). <i>Machine Learning: A Probabilistic Perspective</i>. MIT Press.</li>
{{% /references %}}
