---

title: "MAP（Mean Average Precision）"

pre: "4.5.2 "

weight: 2

title_suffix: "複数正解を持つランキングの定番指標"

---



{{< lead >}}

MAP（Mean Average Precision）は、各クエリの平均適合率（AP）を平均した指標です。検索やレコメンドで、複数の正解アイテムが存在するランキングを評価する際に広く使われます。

{{< /lead >}}



---



## 1. AP と MAP の定義



1 クエリに対する Average Precision (AP) は、正解アイテムを見つけたタイミングごとの適合率を平均します。



$$

\mathrm{AP} = \frac{1}{|G|} \sum_{k \in G} P(k)

$$



ここで \\(G\\) は正解アイテムのランク位置集合、\\(P(k)\\) は位置 \\(k\\) までの適合率。MAP は複数クエリの AP を平均します。



$$

\mathrm{MAP} = \frac{1}{Q} \sum_{q=1}^Q \mathrm{AP}_q

$$



---



## 2. Python で計算



Scikit-learn には直接の MAP 実装はありませんが、`average_precision_score` をクエリごとに計算して平均できます。



```python

from sklearn.metrics import average_precision_score

import numpy as np



aps = []

for q in queries:

    aps.append(average_precision_score(y_true[q], y_score[q]))



map_score = np.mean(aps)

print("MAP:", round(map_score, 4))

```



`y_true[q]` は該当クエリのラベル（0/1）、`y_score[q]` はモデルスコアです。



---



## 3. 特徴と利点



- 正解が複数あるランキングで有効。

- 上位で正解を見つけた方が高いスコアになる。

- Recall の影響も含むため、単純な Precision@k より包括的。



---



## 4. 実務での活用



- **検索システム**：ユーザーが求める複数結果を提示できているかを評価。

- **推薦システム**：購買履歴や視聴履歴など、正解が複数存在する場面で KPI として採用。

- **ランキング学習（LTR）**：LambdaMART や XGBoost などのランキングモデルをオフライン評価する指標。



---



## 5. 注意点



- クエリごとの正解アイテム数が極端に異なる場合、MAP の平均が偏ることがある。Weighted MAP を検討する。

- 正解なし（ラベルが 0 のみ）クエリは AP が定義できないため除外するか、0 として扱うかを決めておく。

- NDCG や Recall@k と併用し、ランキングの多面的な性能を評価する。



---



## まとめ



- MAP は平均適合率の平均で、複数正解を持つランキング評価に最適。

- クエリごとの AP を計算し、平均するだけで算出できる。

- NDCG や Recall@k など他指標と併用し、ランキングモデルの改善を進めよう。



---

