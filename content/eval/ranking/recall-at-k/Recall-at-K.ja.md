---

title: "Recall@k と Precision@k"

pre: "4.5.3 "

weight: 3

title_suffix: "候補リストの網羅性と純度"

---



{{< lead >}}

Recall@k（Top-k Recall）は、上位 k 件の候補の中で正解アイテムをどれだけ取り込めたかを測る指標です。Precision@k と組み合わせることで、候補リストの網羅性と純度を同時に評価できます。

{{< /lead >}}



---



## 1. 定義



クエリ \\(q\\) に対する正解アイテム集合を \\(G_q\\)、上位 k の候補集合を \\(S_{q,k}\\) とすると、



$$

\mathrm{Recall@k} = \frac{|G_q \cap S_{q,k}|}{|G_q|}

$$



$$

\mathrm{Precision@k} = \frac{|G_q \cap S_{q,k}|}{k}

$$



- Recall@k：正解をどれだけ拾えたか。

- Precision@k：提示した候補のうち、どれだけ正解だったか。



---



## 2. Python で計算



```python

import numpy as np



def recall_at_k(y_true, y_score, k):

    idx = np.argsort(-y_score)[:k]

    return y_true[idx].sum() / y_true.sum()



def precision_at_k(y_true, y_score, k):

    idx = np.argsort(-y_score)[:k]

    return y_true[idx].sum() / k

```



`y_true` は 0/1 のラベル、`y_score` はモデルのスコアです。複数クエリがある場合は平均を取ります。



---



## 3. k の選択



- UI や配信枠に合わせて k を決定（例：おすすめ 5 件 → Recall@5）。

- 複数の k を設定し、Recall@5 / Recall@10 などで段階的に評価すると改善ポイントが掴みやすい。



---



## 4. 実務での活用



- **レコメンド**：ユーザーが実際に選ぶアイテムが候補リストに含まれているかを確認。

- **広告配信**：インプレッション枠の中で、クリックやコンバージョンが発生した広告をどれだけ含められたかを評価。

- **A/B テスト**：オンライン実験と合わせて Recall@k をウォッチし、指標の改善がユーザー行動に直結しているかを確認。



---



## 5. Precision@k とのトレードオフ



- Recall@k を上げるには候補を増やす必要があり、Precision@k は下がりやすい。

- k を固定した上で、Recall@k と Precision@k を両立できるようモデルを改善することが理想。

- F1@k や MAP を併用し、バランス良く改善できているかをチェック。



---



## まとめ



- Recall@k は候補リストの網羅性、Precision@k は純度を示す基本指標。

- k の設定を明確にし、複数の指標を併用してランキングモデルの性能を把握しよう。

- オンライン指標（CTR、CVR など）と関連づけて分析することで、ビジネスインパクトを説明しやすくなる。



---

