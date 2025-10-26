---
title: "ナイーブベイズ分類"
pre: "2.2.5 "
weight: 5
title_suffix: "条件付き確率を効率的に推定する"
---

{{< lead >}}
ナイーブベイズは「特徴量が条件付き独立である」という大胆な仮定を置きつつも、計算が速くテキスト分類などで今も実用的なベースラインとして使われます。
{{< /lead >}}

---

## 1. ベイズの定理から導く

クラス \\(y\\) と特徴ベクトル \\(\mathbf{x} = (x_1, \dots, x_d)\\) に対し、

$$
P(y \mid \mathbf{x}) \propto P(y) \prod_{j=1}^d P(x_j \mid y)
$$

と近似します。分母 \\(P(\mathbf{x})\\) は全クラス共通なので、省略して比較できます。

---

## 2. バリアント

- **多項分布型**: 単語頻度など離散データ向け
- **ベルヌーイ型**: 出現/非出現など二値特徴
- **ガウス型**: 実数値特徴を正規分布とみなす

---

## 3. Python での実装例

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

![ダミー図: ナイーブベイズの混同行列](/images/placeholder_regression.png)

---

## 4. 平滑化と特徴選択

- **ラプラス平滑化**: 未観測単語で確率が 0 にならないよう \\(lpha > 0\\) を加える
- **TF-IDF**: 生の頻度よりも情報量の高い単語を強調
- **特徴量選択**: chi-square, mutual information などで次元削減すると精度が安定

---

## 5. 長所と短所

| 長所 | 短所 |
| ---- | ---- |
| 学習・推論が高速 | 独立仮定が現実と合わないと精度低下 |
| 少量データでも安定 | 連続値は正規分布仮定に依存 |
| 実装が簡単 | 相関の強い特徴が多いとダブルカウント |

---

## 6. まとめ

- ベイズの定理を使って事前分布と条件付き確率を組み合わせるシンプルな分類器
- テキストやログのような高次元疎データでも高速
- 強い独立仮定の影響を理解し、必要なら特徴量前処理や他モデルと組み合わせましょう

---
