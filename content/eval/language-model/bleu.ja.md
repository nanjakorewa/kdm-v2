---
title: BLEU 指標
pre: "4.6.1 "
weight: 1
---

BLEU（Bilingual Evaluation Understudy）は、候補文と参照文の n-gram 一致率を基に翻訳・要約などの品質を評価する自動指標です。単語列の一致数を precision として集計し、出力が短すぎる場合は短縮ペナルティ（brevity penalty）を掛けてスコアを算出します。

## 計算手順

1. 1〜4-gram など所定の n について、候補文と参照文の一致数（modified precision）を計算する。
2. 各 precision の対数平均を取り、幾何平均を求める（通常は等重み）。
3. 候補文が参照文より短い場合は brevity penalty を掛け合わせる。

```python
import math
from collections import Counter

def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

def modified_precision(candidate, references, n):
    cand_counts = ngram_counts(candidate, n)
    max_ref = Counter()
    for ref in references:
        max_ref |= ngram_counts(ref, n)
    overlap = {ng: min(count, max_ref[ng]) for ng, count in cand_counts.items()}
    return sum(overlap.values()), max(1, sum(cand_counts.values()))

def bleu(candidate, references, max_n=4):
    candidate = candidate.split()
    references = [r.split() for r in references]
    precisions = []
    for n in range(1, max_n + 1):
        overlap, total = modified_precision(candidate, references, n)
        precisions.append(overlap / total)
    if min(precisions) == 0:
        return 0.0
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
    ref_lens = [len(r) for r in references]
    closest_ref = min(ref_lens, key=lambda r: (abs(r - len(candidate)), r))
    brevity_penalty = 1.0 if len(candidate) > closest_ref else math.exp(1 - closest_ref / len(candidate))
    return brevity_penalty * geo_mean
```

## 長所

- 実装が簡単で高速。翻訳ベンチマークの比較指標として定番。
- 複数の参照文を用意すれば言い換えへの耐性が上がる。

## 注意点

- 同義語や語順の違いに弱く、意味が同じでもスコアが低く出る場合がある。
- 長文要約などでは人手評価との相関が低くなることがある。
- 日本語のように形態素境界が曖昧な言語では、分かち書きなどでトークン化してから計算すると安定する。

