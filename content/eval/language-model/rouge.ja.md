---
title: ROUGE
pre: "4.6.2"
weight: 2
---

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a family of metrics that measure overlap between candidate and reference summaries. ROUGE-1/2 count unigram/bigram recall, ROUGE-L uses the longest common subsequence (LCS), and ROUGE-W weights consecutive matches.

## Typical usage

- ROUGE-1/ROUGE-2: recall of 1-gram and 2-gram tokens.
- ROUGE-L: based on the length of the LCS divided by the reference length.
- ROUGE-Lsum: sentence-level LCS averaged across reference sentences.

`python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
candidate = "the cat sat on the mat"
reference = "a cat was sitting on the mat"
print(scorer.score(reference, candidate))
`

## Strengths and caveats

- **Strengths**: correlates well with human summary recall; supports multiple variants.
- **Caveats**: ignores semantics; rewards lexical overlap even when the meaning differs.
