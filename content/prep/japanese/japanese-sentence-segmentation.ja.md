---
title: "日本語文を安全に分割する"
pre: "3.7.10 "
weight: 10
title_suffix: "句点・感嘆符・括弧を考慮したルールベース分割"
---

句点が省略された文章や絵文字を含むメッセージを扱うとき、単純な `text.split("。")` では文の境界がずれがちです。句点・疑問符・括弧のペアを意識したルールで分割すると安定します。

```python
import re

SENTENCE_END = re.compile(r"([。！？!?]+)(?=[^\\)」』】》”』』】》」]|\Z)")

def split_sentences(text: str) -> list[str]:
    parts = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            parts.append(sentence)
        start = end
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


sample = "今日は打ち合わせ！(本当に？)気を付けてね…。了解しました👍"
for i, sentence in enumerate(split_sentences(sample), start=1):
    print(i, sentence)
```

### 運用のポイント
- 括弧内の疑問符で分割したくない場合は、`SENTENCE_END` の除外文字クラスを追加して調整します。
- Slack などで見かける改行を伴わない長文には、句点が無くても 20〜30 文字ごとに強制的に分割するフォールバックを用意すると読みやすくなります。
- 高精度が求められる場合は、日本語に対応した文分割モデル（KUROSHIO、spaCy ja_ginza の `senter` など）を選定し、今回のルールベース分割を簡易版として併用するのがおすすめです。
