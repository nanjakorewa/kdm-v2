---
title: ROUGE 指標
pre: "4.6.2 "
weight: 2
---

ROUGE（Recall-Oriented Understudy for Gisting Evaluation）は、候補要約と参照要約の重なり具合を測る指標群です。特に ROUGE-1／ROUGE-2（1-gram・2-gram の再現率）、ROUGE-L（最長共通部分列 LCS に基づく指標）が要約タスクで広く利用されています。

## 主なバリエーション

- **ROUGE-1 / ROUGE-2**: 1-gram・2-gram の **再現率（recall）** を計算。抽出型要約や情報保持の確認に有効。
- **ROUGE-L**: 最長共通部分列（Longest Common Subsequence）の長さを、参照文の長さで割った値を利用。語順を考慮できる。
- **ROUGE-Lsum**: 文単位で LCS を求め、それぞれのスコアを平均化。長文要約の評価でよく使われる。

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
candidate = "今日はとても良い天気だ"
reference = "今日は良い天気です"
print(scorer.score(reference, candidate))
```

## 長所
- 参照要約と候補要約の重なりを直感的に把握できる。
- 計算コストが小さく、複数の派生指標を同時に算出しやすい。
- 人手評価（回収率）との相関が高いとされ、研究や競技会での採用実績が多い。

## 注意点
- 意味が同じでも語彙や言い回しが異なるとスコアが下がる。埋め込みベース指標（BERTScore など）と併用すると安心。
- 参照要約が 1 つだけの場合は、言い換えへの耐性が低くなる。可能であれば複数参照を用意する。
- 日本語では形態素解析やサブワード分割を行ってから計算すると、語彙の揺れに強くなる。
