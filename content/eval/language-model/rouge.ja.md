---

title: ROUGE 指標

pre: "4.6.2 "

weight: 2

---



ROUGE（Recall-Oriented Understudy for Gisting Evaluation）は、候補要約と参照要約の重なり具合を測る指標群です。ROUGE-1/ROUGE-2 は n-gram の再現率（recall）、ROUGE-L は最長共通部分列（LCS）の長さに基づきます。



## 主なバリエーション



- **ROUGE-1 / ROUGE-2**: 1-gram・2-gram の再現率を計算。抽出型要約や重要語の取りこぼし確認に向く。

- **ROUGE-L**: 最長共通部分列（LCS）の長さを参照長で割ってスコア化。語順の情報を活かせる。

- **ROUGE-Lsum**: 文単位の LCS を平均化。長文要約の評価で使われることが多い。



```python

from rouge_score import rouge_scorer



scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)

candidate = "今日はとても良い天気だ"

reference = "今日は良い天気です"

print(scorer.score(reference, candidate))

```python

# 長所



- 計算が軽く、複数の派生指標を同時に算出できる。

- 人手評価（recall）との相関が比較的高い。

- 抽出型要約や文章の情報量を確認する用途で定番。



## 注意点



- 意味的な類似性を考慮しないため、言い換えで品質が変わってもスコアが低下する。

- 参照要約が 1 つだけだと、候補の多様性を正しく評価できないことがある。

- 日本語では形態素解析やサブワード分割を行ってから計算するとスコアが安定しやすい。



