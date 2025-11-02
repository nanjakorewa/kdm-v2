---

title: "NDCG（Normalized Discounted Cumulative Gain）"

pre: "4.5.1 "

weight: 1

title_suffix: "順位と関連度を同時に評価する"

---



{{< lead >}}

NDCG は、ランキングの上位にどれだけ関連度の高いアイテムを配置できたかを評価する指標です。検索エンジンやレコメンドで、関連度スコアが大きいアイテムを上位に並べる能力を測ります。

{{< /lead >}}



---



## 1. 定義



ランク \\(i\\) の関連度スコアを \\(rel_i\\) とすると、DCG（Discounted Cumulative Gain）は



$$

\mathrm{DCG@k} = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}

$$



理想的な並び（IDCG）での DCG を求め、正規化することで NDCG を得ます。



$$

\mathrm{NDCG@k} = \frac{\mathrm{DCG@k}}{\mathrm{IDCG@k}}

$$



---



## 2. Python で計算



```python

from sklearn.metrics import ndcg_score



# y_true: shape (n_samples, n_labels) に関連度を格納

# y_score: モデルが出力したスコア

score = ndcg_score(y_true, y_score, k=10)

print("NDCG@10:", round(score, 4))

```



`ndcg_score` は関連度を 0/1 だけでなく、段階的な整数スコアでも扱えます。Ground Truth の関連度配列を用意するのがポイントです。



---



## 3. ハイパーパラメータ



- **k の設定**：ユーザーに提示する件数に合わせて @5, @10 などを選択。

- **関連度スコア**：0/1 の二値でも良いが、段階的なラベル（非常に関連、やや関連など）を使うと精度が高い評価になる。

- **log の底**：定義によっては log2 以外を使う場合もあるが、スケーリングの違いに過ぎない。



---



## 4. 実務での活用



- **検索結果評価**：人手で付けた関連度ラベルを使い、NDCG@10 を主指標にする例が多い。

- **レコメンド**：Implicit Feedback（閲覧や購買）を関連度として扱い、ランキング改善の KPI とする。

- **A/B テスト**：オンライン評価と合わせ、オフラインで NDCG の向上がオンライン指標にどう影響するかを分析。



---



## 5. 注意点



- Ground Truth の関連度を整備するコストが高い。Implicit Feedback を使う場合はノイズの影響に注意。

- 候補生成とランキングの二段階構成では、それぞれのフェーズに適した指標を設定する。

- NDCG だけでなく Recall@k や MAP も併用し、ユーザー体験を多面的に評価する。



---



## まとめ



- NDCG は関連度スコアを対数減衰で加重し、上位に高関連度が並ぶほど高くなる指標。

- `ndcg_score` で容易に計算でき、k の設定や関連度スコアの設計が重要。

- 他のランキング指標と併用し、ランキング品質を総合的に向上させよう。



---

