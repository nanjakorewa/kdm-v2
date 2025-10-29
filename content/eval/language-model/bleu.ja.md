---
title: BLEU metric
pre: "4.6.1"
weight: 1
---

BLEU (Bilingual Evaluation Understudy) is an n-gram based metric widely used for machine translation and summarisation. It counts overlapping n-grams between a candidate and references, then applies a brevity penalty so overly short outputs are discouraged.

## Calculation steps

1. Compute precision for each n-gram length (typically 1–4).
2. Take the log-average of the precisions (usually equal weights).
3. Multiply by a brevity penalty when the candidate is shorter than references.

`python
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
`

## Strengths and caveats

- **Strengths**: simple and fast; still a de-facto benchmark metric.
- **Caveats**: insensitive to synonyms or paraphrases; long-form generation quality correlates poorly with BLEU.
